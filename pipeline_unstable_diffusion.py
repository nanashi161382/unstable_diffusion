# @title UnstableDiffusionPipeline
# See the following web page for the usage.
# https://github.com/nanashi161382/unstable_diffusion/tree/main
from diffusers import StableDiffusionInpaintPipelineLegacy
from diffusers import DDIMScheduler
import inspect
import IPython
from IPython.display import display
import numpy as np
import PIL
import torch
from torch import autocast
from typing import Optional, List, Union, Callable, Tuple


class PipelineType:
    def __init__(self, rand_seed: Optional[int], use_new_timesteps: bool = True):
        """
        Args:
            rand_seed (`int`, *optional*):
                A random seed for [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
            use_new_timesteps (`bool`, *optional*):
                Switch the implementation of GetTimestepsWithStrength().
        """
        self._rand_seed = rand_seed
        self._generator = None
        self._use_new_timesteps = use_new_timesteps

    def InitializeGenerator(self, pipe):
        if not self._rand_seed:
            return None
        if not self._generator:
            self._generator = torch.Generator(device=pipe.device.type)
        self._generator.manual_seed(self._rand_seed)
        print(f"Setting random seed to {self._rand_seed}")

    def GetGenerator(self):
        return self._generator

    def Rand(self, shape, device, dtype):
        generator = self.GetGenerator()
        if device.type == "mps":
            # randn does not work reproducibly on mps
            return torch.randn(
                shape,
                generator=generator,
                device="cpu",
                dtype=dtype,
            ).to(device)
        else:
            return torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )

    def GetTimestepsWithStrength(self, pipe, num_inference_steps, strength):
        # Probably the new version is fine, but keep the old version for a while.
        if self._use_new_timesteps:
            return self.GetTimestepsWithStrengthNew(pipe, num_inference_steps, strength)
        else:
            return self.GetTimestepsWithStrengthOld(pipe, num_inference_steps, strength)

    # Modified a copy from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    # Note: A bit skeptical about the correctness of the computation of `latent_timestep`
    def GetTimestepsWithStrengthNew(self, pipe, num_inference_steps, strength):
        # get the original timestep using init_timestep
        offset = pipe.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = pipe.scheduler.timesteps[t_start:].to(pipe.device)

        latent_timestep = timesteps[:1].repeat(1)  # is `repeat(1)` meaningful?

        return timesteps, latent_timestep

    # Old version of computing latent_timestep before refactoring.
    def GetTimestepsWithStrengthOld(self, pipe, num_inference_steps, strength):
        # get the original timestep using init_timestep
        offset = pipe.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = pipe.scheduler.timesteps[t_start:].to(pipe.device)

        # add noise to latents using the timesteps
        latent_timestep = pipe.scheduler.timesteps[-init_timestep]
        latent_timestep = torch.tensor(
            [latent_timestep], device=pipe.device
        )  # meaningless?

        return timesteps, latent_timestep


class Txt2Img(PipelineType):
    def __init__(
        self,
        size: Tuple[int, int] = (512, 512),
        latents: Optional[torch.FloatTensor] = None,
        rand_seed: Optional[int] = None,
    ):
        """
        Args:
            size (`(int, int)`, *optional*, defaults to (512, 512))
                The (width, height) pair in pixels of the generated image.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
            rand_seed (`int`, *optional*):
                A random seed for [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
        """
        PipelineType.__init__(self, rand_seed)
        width, height = size
        self._width = width
        self._height = height
        self._latents = latents

    def GetInitialLatentsAndTimesteps(self, pipe, dtype, num_inference_steps: int):
        scale_factor = pipe.vae_scale_factor()
        generator = self.GetGenerator()

        if self._height % scale_factor != 0 or self._width % scale_factor != 0:
            print(
                f"`width` and `height` have to be divisible by {scale_factor}. "
                "Automatically rounded."
            )

        # get the initial random noise unless the user supplied it
        latents_shape = (
            1,
            pipe.unet.in_channels,
            self._height // scale_factor,
            self._width // scale_factor,
        )
        if self._latents:  # user supplied latents
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )
            latents = latents.to(pipe.device)
        else:
            latents = self.Rand(latents_shape, pipe.device, dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * pipe.scheduler.init_noise_sigma

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps = pipe.scheduler.timesteps.to(pipe.device)

        return latents, timesteps, None


class Img2Img(PipelineType):
    def __init__(
        self,
        init_image: PIL.Image.Image,
        strength: float = 0.8,
        rand_seed: Optional[int] = None,
        use_new_timesteps: bool = True,
    ):
        """
        Args:
            init_image `PIL.Image.Image`:
                `Image`, or tensor representing an image batch, that will be used as the starting point for the process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            rand_seed (`int`, *optional*):
                A random seed for [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
            use_new_timesteps (`bool`, *optional*):
                Switch the implementation of GetTimestepsWithStrength().
        """
        PipelineType.__init__(self, rand_seed, use_new_timesteps)
        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )
        self._strength = strength
        self._init_image = init_image

    def GetInitialLatentsAndTimesteps(self, pipe, dtype, num_inference_steps: int):
        generator = self.GetGenerator()

        init_latents = pipe.image_model.Encode(self._init_image, dtype, generator)
        noise = self.Rand(init_latents.shape, pipe.device, dtype)
        timesteps, latent_timestep = self.GetTimestepsWithStrength(
            pipe, num_inference_steps, self._strength
        )

        init_latents_noise = pipe.scheduler.add_noise(
            init_latents, noise, latent_timestep
        )

        return init_latents_noise, timesteps, None


class Inpaint(PipelineType):
    def __init__(
        self,
        init_image: PIL.Image.Image,
        mask_image: PIL.Image.Image,
        strength: float = 0.8,
        rand_seed: Optional[int] = None,
        use_new_timesteps: bool = True,
    ):
        """
        Args:
            init_image `PIL.Image.Image`:
                `Image`, or tensor representing an image batch, that will be used as the starting point for the process. This is the image whose masked region will be inpainted.
            mask_image `PIL.Image.Image`:
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength` is 1, the denoising process will be run on the masked area for the full number of iterations specified in `num_inference_steps`. `init_image` will be used as a reference for the masked area, adding more noise to that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
            rand_seed (`int`, *optional*):
                A random seed for [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
            use_new_timesteps (`bool`, *optional*):
                Switch the implementation of GetTimestepsWithStrength().
        """
        PipelineType.__init__(self, rand_seed, use_new_timesteps)
        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )
        self._strength = strength
        self._init_image = init_image
        self._mask_image = mask_image

    def GetInitialLatentsAndTimesteps(self, pipe, dtype, num_inference_steps: int):
        generator = self.GetGenerator()

        init_latents = pipe.image_model.Encode(self._init_image, dtype, generator)
        mask = pipe.image_model.PreprocessMask(self._mask_image, dtype)
        noise = self.Rand(init_latents.shape, pipe.device, dtype)
        apply_mask = self.ApplyMask(pipe, init_latents, mask, noise)

        # check sizes
        if not mask.shape == init_latents.shape:
            raise ValueError("The mask and init_image should be the same size!")

        timesteps, latent_timestep = self.GetTimestepsWithStrength(
            pipe, num_inference_steps, self._strength
        )

        # add noise to latents using the timesteps
        init_latents_noise = apply_mask.AddNoise(latent_timestep)

        return init_latents_noise, timesteps, apply_mask

    class ApplyMask:
        def __init__(self, pipe, init_latents, mask, noise):
            self._pipe = pipe
            self._init_latents = init_latents
            self._mask = mask
            self._noise = noise

        def AddNoise(self, ts):
            return self._pipe.scheduler.add_noise(self._init_latents, self._noise, ts)

        def __call__(self, latents, t):
            init_latents_noise = self.AddNoise(torch.tensor([t]))
            return (init_latents_noise * self._mask) + (latents * (1 - self._mask))


class ImageModel:
    def __init__(self, pipe: UnstableDiffusionPipeline):
        self._pipe = pipe

    def Preprocess(self, image: PIL.Image.Image):
        w, h = image.size
        # Shouldn't this be consistent with vae_scale_factor?
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    def Encode(
        self, image: PIL.Image.Image, dtype, generator: Optional[torch.Generator]
    ):
        image = self.Preprocess(image).to(device=self._pipe.device, dtype=dtype)

        # encode the init image into latents and scale the latents
        latent_dist = self._pipe.vae.encode(image).latent_dist
        latents = latent_dist.sample(generator=generator)
        latents = 0.18215 * latents

        # expand init_latents for batch_size
        return torch.cat([latents], dim=0)  # meaningless?

    def Decode(self, latents):
        latents = 1 / 0.18215 * latents
        image = self._pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return self._pipe._pipe.numpy_to_pil(image)

    def PreprocessMask(self, mask: PIL.Image.Image, dtype):
        scale_factor = self._pipe.vae_scale_factor()
        # preprocess mask
        mask = mask.convert("L")
        w, h = mask.size
        # Shouldn't this be consistent with vae_scale_factor?
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        mask = mask.resize(
            (w // scale_factor, h // scale_factor), resample=PIL.Image.NEAREST
        )
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask).to(device=self._pipe.device, dtype=dtype)
        return torch.cat([mask])  # meaningless?


class UnstableDiffusionPipeline:
    def __init__(self, devicetype: str = "cuda"):
        self._devicetype_str = devicetype

    def Connect(self, dataset: str, auth_token: Optional[str] = None):
        self._dataset = dataset
        self._auth_token = auth_token

        extra_args = {
            # Saving memory
            "torch_dtype": torch.float32,
            "revision": "fp16",
        }
        if auth_token:
            extra_args["use_auth_token"] = auth_token

        if dataset == "hakurei/waifu-diffusion":
            extra_args["scheduler"] = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        elif dataset == "Linaqruf/anything-v3.0":
            del extra_args["revision"]
        elif dataset == "naclbit/trinart_stable_diffusion_v2":
            del extra_args["torch_dtype"]
            extra_args["revision"] = "diffusers-60k"

        # Prepare the StableDiffusion pipeline.
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            dataset, **extra_args
        ).to(self._devicetype_str)
        return self.SetPipeline(pipe)

    def SetPipeline(self, pipe):
        self._pipe = pipe
        self.tokenizer = self._pipe.tokenizer
        self.device = self._pipe.device
        self.unet = self._pipe.unet
        self.scheduler = self._pipe.scheduler
        self.vae = self._pipe.vae
        self.image_model = ImageModel(self)
        return self

    def progress_bar(self, *input, **kwargs):
        return self._pipe.progress_bar(*input, **kwargs)

    def vae_scale_factor(self):
        if "vae_scale_factor" in dir(self._pipe):
            print(
                "vae_scale_factor is available in StableDiffusionInpaintPipelineLegacy."
            )
            return self._pipe.vae_scale_factor
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def EncodeText(self, text: str, max_length):
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, max_length:])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, :max_length]
        text_embeddings = self._pipe.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=None,
        )[0]

        # duplicate text embeddings for each generation per prompt
        return text_embeddings.repeat_interleave(1, dim=0)  # meaningless?

    def GetTextEmbeddings(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        do_classifier_free_guidance: bool,
    ):
        # get prompt text embeddings
        text_embeddings = self.EncodeText(prompt, self.tokenizer.model_max_length)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            uncond_embeddings = self.EncodeText(
                negative_prompt, self.tokenizer.model_max_length
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        pipeline_type: Union[Txt2Img, Img2Img, Inpaint],
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            pipeline_type: (`Txt2Img` or `Img2Img` or `Inpaint`)
                The pipeline execution type.
            prompt (`str`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
            pipeline_type:
                Type of pipeline.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to [`schedulers.DDIMScheduler`], will be ignored for others.
        Returns:
            generated images
        """
        with autocast(self._devicetype_str):
            pipeline_type.InitializeGenerator(self)

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            text_embeddings = self.GetTextEmbeddings(
                prompt, negative_prompt, do_classifier_free_guidance
            )

            # set timesteps and initial latents
            self.scheduler.set_timesteps(num_inference_steps)
            (
                latents,
                timesteps,
                apply_mask,
            ) = pipeline_type.GetInitialLatentsAndTimesteps(
                self, text_embeddings.dtype, num_inference_steps
            )

            # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self._pipe.prepare_extra_step_kwargs(
                pipeline_type.GetGenerator(), eta
            )

            for i, t in enumerate(self.progress_bar(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # masking
                if apply_mask:
                    latents = apply_mask(latents, t)

            return self.image_model.Decode(latents)
