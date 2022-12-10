# @title LayeredDiffusionPipeline
# See the following web page for the usage.
# https://github.com/nanashi161382/unstable_diffusion/tree/main
from diffusers import (
    StableDiffusionInpaintPipelineLegacy,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from IPython.display import display
import numpy as np
import PIL
import re
import torch
from torch import autocast
from typing import Any, Optional, List, Union, Callable, Tuple, Dict


#
# -- Debug functions --
#
# Debug level:
#   0: Warn
#   1: Info  <-  Default
#   3: Debug
#   5: Trace
#
# Should raise an exception for Error.
debug_level = 1


def SetDebugLevel(level: int):
    global debug_level
    debug_level = level


def Debug(level: int, obj):
    if level <= debug_level:
        display(obj)


#
# -- Type: Generator --
#
Generator = Optional[torch.Generator]


#
# -- Share the computation target throughout the pipeline. --
#
class SharedTarget:
    def __init__(self, device, dtype):
        if not device:
            raise ValueError(f"device should not be None.")
        if not dtype:
            raise ValueError(f"dtype should not be None.")
        self.dict = {
            "device": device,
            "dtype": dtype,
        }

    def device(self):
        return self.dict["device"]

    def device_type(self):
        return self.device().type

    def dtype(self):
        return self.dict["dtype"]


#
# --  TextModel - a wrapper to the CLIP tokenizer and the CLIP text encoder --
#
class TextModel:
    def __init__(self, tokenizer, text_encoder, device):
        """
        Args:
            tokenizer:
            text_encoder:
            device:
        """
        self._tokenizer = tokenizer
        self._text_encoder = text_encoder
        self.target = SharedTarget(device, text_encoder.config.torch_dtype)

    def max_length(self):
        return self._tokenizer.model_max_length

    def num_tokens(self, text: str):
        tokens = self._tokenizer(text)
        return len(tokens.input_ids) - 2

    def tokenize(self, text: str):
        max_length = self.max_length()
        return self._tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def decode_tokens(self, tokens):
        return self._tokenizer.batch_decode(tokens)

    def EncodeText(self, text: str, fail_on_truncation: bool = True):
        max_length = self.max_length()
        text_inputs = self.tokenize(text)
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        if text_input_ids.shape[-1] > max_length:
            removed_text = self.decode_tokens(text_input_ids[:, max_length:])
            if fail_on_truncation:
                raise ValueError(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )
            text_input_ids = text_input_ids[:, :max_length]
        embeddings = self._text_encoder(
            text_input_ids.to(self.target.device()),
            attention_mask=None,
        )

        num_tokens = 0
        for m in attention_mask[0]:
            if m.item() > 0:
                num_tokens += 1
        return embeddings[0], embeddings[1], num_tokens


#
# -- ImageModel - a wrapper to VAE
#
class ImageModel:
    def __init__(self, vae, vae_scale_factor):
        """
        Args:
            vae:
            vae_scale_factor:
        """
        self._vae = vae
        self._vae_scale_factor = vae_scale_factor
        self._num_channels = 4  # Should match with U-Net.

    def vae_scale_factor(self):
        return self._vae_scale_factor

    def num_channels(self):
        return self._num_channels

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
        self, image: PIL.Image.Image, generator: Generator, target: SharedTarget
    ):
        image = self.Preprocess(image).to(**target.dict)

        # encode the init image into latents and scale the latents
        latent_dist = self._vae.encode(image).latent_dist
        latents = latent_dist.sample(generator=generator)
        latents = 0.18215 * latents

        return latents

    def Decode(self, latents):
        latents = 1 / 0.18215 * latents
        image = self._vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return DiffusionPipeline.numpy_to_pil(image)

    def PreprocessMask(self, mask: PIL.Image.Image, target: SharedTarget):
        scale_factor = self.vae_scale_factor()
        # preprocess mask
        mask = mask.convert("L")
        w, h = mask.size
        # Shouldn't this be consistent with vae_scale_factor?
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        mask = mask.resize(
            (w // scale_factor, h // scale_factor), resample=PIL.Image.NEAREST
        )
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (self._num_channels, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask).to(**target.dict)
        return mask


#
# -- Encoding is a class that defines how to encode a prompt --
#
class StandardEncoding:
    def Encode(self, text_model: TextModel, text: str):
        return text_model.EncodeText(text)[0]


class PromptParser:
    "Prompt parser for ShiftEncoding"

    class Prompt:
        class Chunks:
            def __init__(
                self,
                texts: List[str],
                weights: List[float],
                token_nums: List[int],
            ):
                self.texts = texts
                self.weights = weights
                self.token_nums = token_nums

            def __len__(self):
                return len(self.texts)

            def Reverse(self):
                self.texts.reverse()
                self.weights.reverse()
                self.token_nums.reverse()

        def __init__(
            self,
            anchored,  # : PromptParser.Chunks,
            unanchored,  #: PromptParser.Chunks,
        ):
            self.anchored = anchored
            self.unanchored = unanchored

        def Reverse(self):
            self.unanchored.Reverse()

        def ShiftRange(self):
            i = len(self.unanchored)
            return i if i > 0 else 1

        def Shift(self, i: int, rotate: bool):
            anc = self.anchored
            unanc = self.unanchored
            if rotate:
                return self.Chunks(
                    anc.texts + unanc.texts[i:] + unanc.texts[:i],
                    anc.weights + unanc.weights[i:] + unanc.weights[:i],
                    anc.token_nums + unanc.token_nums[i:] + unanc.token_nums[:i],
                )
            else:
                return self.Chunks(
                    anc.texts + unanc.texts[i:],
                    anc.weights + unanc.weights[i:],
                    anc.token_nums + unanc.token_nums[i:],
                )

    def GetAnnotation(self, text: str):
        words = []
        weight = 1.0
        repeat = 1
        for word in [x.strip() for x in text.split(" ")]:
            if not word:
                continue
            if word[0] == ":":
                if len(word) == 1:
                    raise ValueError(f"Don't put any spaces after :")
                weight = float(word[1:])
            elif word[0] == "*":
                if len(word) == 1:
                    raise ValueError(f"Don't put any spaces after *")
                repeat = int(word[1:])
            else:
                words.append(word)
        if weight < 0.0:
            raise ValueError(f"Invalid weight: {weight}")
        if repeat < 1:
            raise ValueError(f"Invalid repeat: {repeat}")
        return " ".join(words), weight, repeat

    def ParseClause(self, text_model: TextModel, text: str):
        if not text:
            return self.Prompt.Chunks([], [], [])

        chunks = [x.strip() for x in text.split(",")]
        annotations = [self.GetAnnotation(chunk) for chunk in chunks if chunk]
        texts = []
        weights = []
        token_nums = []
        for text, weight, repeat in annotations:
            num_tokens = text_model.num_tokens(text)
            for i in range(repeat):
                texts.append(text)
                weights.append(weight)
                token_nums.append(num_tokens)
        return self.Prompt.Chunks(texts, weights, token_nums)

    def Parse(self, text_model: TextModel, text: str):
        clauses = [x.strip() for x in re.split(">>>+", text)]
        if len(clauses) > 2:
            raise ValueError(
                f'">>>" should appear at most once in a prompt, '
                f"but appeared {len(clauses) - 1} times: {text}"
            )
        elif len(clauses) == 2:
            anc, unanc = clauses
        else:
            anc = ""
            unanc = text
        return self.Prompt(
            self.ParseClause(text_model, anc), self.ParseClause(text_model, unanc)
        )


class BaseAccumulator:
    def __init__(self, target: SharedTarget):
        self.unweighted_mean = None
        self.target = target

    def AddOrInit(self, v, x):
        if v is None:
            return x
        else:
            return v + x

    def Add(self, unweighted_mean):
        self.unweighted_mean = self.AddOrInit(self.unweighted_mean, unweighted_mean)

    def Average(self, chunk_len):
        chunk_len = torch.tensor(chunk_len).to(self.target.dtype())
        self.unweighted_mean /= chunk_len

    def AdjustMean(self, states):
        new_mean = states.float().mean(axis=[-2, -1]).to(self.target.dtype())
        states *= (self.unweighted_mean / new_mean).unsqueeze(-1)
        return torch.cat([torch.unsqueeze(states, 0)], dim=0).to(self.target.device())


class ParallelAccumulator(BaseAccumulator):
    def __init__(self, target: SharedTarget):
        super().__init__(target)
        self.states = None

    def Add(self, states, unweighted_mean):
        super().Add(unweighted_mean)
        self.states = self.AddOrInit(self.states, states)

    def Average(self, chunk_len):
        super().Average(chunk_len)
        self.states /= chunk_len

    def ToStates(self):
        return self.states


class ShiftEncoding(StandardEncoding):
    def __init__(
        self,
        reverse: bool = True,
        rotate: bool = True,
        convolve: Union[bool, List[float]] = False,
    ):
        self._reverse = reverse
        self._rotate = rotate
        if not isinstance(convolve, list):
            if convolve:
                convolve = [0.7, 0.23, 0.07]
            else:
                convolve = [1.0]
        self._convolve = np.array(convolve)

    def GetWeights(self, text_model: TextModel, chunks):
        weights = [1.0]
        for w, n in zip(chunks.weights, chunks.token_nums):
            weights += [w] * n
        weights = (
            np.convolve(np.array(weights) - 1.0, self._convolve, mode="full") + 1.0
        )
        Debug(3, weights)
        if len(weights) >= text_model.max_length():
            weights = weights[: text_model.max_length()]
        weights = np.concatenate(
            [
                weights,
                np.array([1.0] * (text_model.max_length() - len(weights))),
            ]
        )
        return weights

    def Encode(self, text_model: TextModel, text: str):
        target = text_model.target
        parser = PromptParser()
        prompt = parser.Parse(text_model, text)
        proc = ParallelAccumulator(target)

        def run():
            shift_range = prompt.ShiftRange()
            for i in range(shift_range):
                chunks = prompt.Shift(i, self._rotate)
                shifted_text = " ".join(chunks.texts)
                weights = self.GetWeights(text_model, chunks)
                Debug(3, shifted_text)

                embeddings = text_model.EncodeText(shifted_text, False)
                unweighted_mean = (
                    embeddings[0][0].float().mean(axis=[-2, -1]).to(**target.dict)
                )
                weights = torch.tensor(weights).to(**target.dict).unsqueeze(-1)
                full = embeddings[0][0] * weights
                proc.Add(full, unweighted_mean)
            return shift_range

        n = run()
        if self._reverse:
            prompt.Reverse()
            n += run()
        proc.Average(n)

        new_states = proc.ToStates()
        return proc.AdjustMean(new_states)


#
# -- Type: PromptType --
#
# A prompt can be
#  * str  --  a prompt text
#  * (str, Encoding)  --  a prompt text & its encoding method
#  * [(int, str) or (int, str, Encoding)]  --  a list of prompt texts & their encoding methods
#    combined with the ratio of the starting position against the total steps.
PromptType = Union[
    str,
    Tuple[str, StandardEncoding],
    List[Union[Tuple[float, str], Tuple[float, str, StandardEncoding]]],
]


#
# -- Text embedding utility to combine multiple embeddings --
#
class TextEmbeddings:
    def __init__(self, text_embedding):
        self._text_embeddings = [text_embedding]
        self._expiry = [100]

    def AddNext(self, start: float, next_embedding):
        if (start <= 0.0) or (start > 1.0):
            raise ValueError(f"`start` must be between 0 and 1.")
        self._expiry[-1] = start
        self._expiry.append(100)
        self._text_embeddings.append(next_embedding)

    def Get(self, progress: float):
        for k, e in enumerate(self._expiry):
            if progress < e:
                return self._text_embeddings[k]
        Debug(0, f"TextEmbeddings: unexpected progress: {progress:.2f}")
        Debug(0, "returning the last text embeddings.")
        return self._text_embeddings[-1]

    @classmethod
    def Create(
        cls,
        input: PromptType,
        text_model: TextModel,
        default_encoding: StandardEncoding,
    ):
        if not input:
            return cls(default_encoding.Encode(text_model, ""))
        elif isinstance(input, str):
            return cls(default_encoding.Encode(text_model, input))
        elif isinstance(input, tuple):
            return cls(input[1].Encode(text_model, input[0]))
        elif not isinstance(input, list):
            raise ValueError(
                "input should be `str`, `(str, encoding)`, or a list of `(int, str)` and/or "
                "`(int, str, encoding)`."
            )
        embeddings = None
        for p in input:
            if len(p) < 2:
                raise ValueError("input should contain at least `start` and `prompt`.")
            start = p[0]
            prompt = p[1]
            if len(p) > 2:
                encoding = input[2]
            else:
                encoding = default_encoding
            if embeddings:
                embeddings.AddNext(start, encoding.Encode(text_model, prompt))
            else:
                if start > 0.0:
                    embeddings = cls(encoding.Encode(text_model, ""))
                    embeddings.AddNext(start, encoding.Encode(text_model, prompt))
                else:
                    embeddings = cls(encoding.Encode(text_model, prompt))
        return embeddings


#
# -- Scheduler - a wrapper to the diffuser library's scheduler --
#
class Scheduler:
    def Connect(self, dataset):
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            dataset, subfolder="scheduler"
        )
        self.have_max_timesteps = False
        self._rand_seed = None
        self._generator = None
        return self

    def CopyFrom(self, another):
        self.scheduler = another.scheduler
        self.have_max_timesteps = another.have_max_timesteps
        self._rand_seed = another._rand_seed
        self._generator = another._generator
        return self

    def Get(self):
        return self.scheduler

    def ResetGenerator(self, target: SharedTarget, rand_seed: Optional[int]):
        self._rand_seed = rand_seed
        if rand_seed and (not self._generator):
            self._generator = torch.Generator(device=target.device())
        self._generator.manual_seed(rand_seed)
        Debug(1, f"Setting random seed to {self._rand_seed}")

    def generator(self):
        if not self._rand_seed:
            return None
        return self._generator

    def SetTimesteps(self, num_inference_steps):
        # Some schedulers like DDIM and PNDM doesn't seem to handle longer inference steps
        # than those in training.
        if self.have_max_timesteps and (len(self.scheduler) < num_inference_steps):
            num_inference_steps = len(self.scheduler)
        self.num_inference_steps = num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps)
        return num_inference_steps

    def PrepareExtraStepKwargs(self, pipe, eta):
        self._extra_step_kwargs = pipe.prepare_extra_step_kwargs(self.generator(), eta)

    def Step(self, residuals, timestep, original_latents):
        if len(residuals) == 1:
            new_latents = self.scheduler.step(
                residuals[0], timestep, original_latents[0], **self._extra_step_kwargs
            ).prev_sample
            return [new_latents]
        residual_cat = torch.cat(residuals)
        original_latents_cat = torch.cat([original_latents] * len(residuals))
        new_latents_cat = self.scheduler.step(
            residual_cat, timestep, original_latents_cat, **self._extra_step_kwargs
        ).prev_sample
        return new_latents_cat.chunk(len(residuals))

    def InspectLatents(self, model_output, timestep, sample):
        """
        This is to inspect the noise (= model output) during image generation.
        Calling the scheduler to recover latents from noise changing the scheduler's internal state
        and thus affects the image generation.
        So this reimplement the function so as not to change the internal state by copying and
        modifying DPMSolverMultistepScheduler.step() available at
        https://github.com/huggingface/diffusers/blob/769f0be8fb41daca9f3cbcffcfd0dbf01cc194b8/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L428
        The implementation depends on scheduler, so this function only supports
        DPMSolverMultistepScheduler.
        """
        if not isinstance(self.scheduler, DPMSolverMultistepScheduler):
            raise ValueError("Unsupported scheduler for inspection.")

        timestep = timestep.to(self.scheduler.timesteps.device)
        step_index = (self.scheduler.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.scheduler.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = (
            0
            if step_index == len(self.scheduler.timesteps) - 1
            else self.scheduler.timesteps[step_index + 1]
        )
        lower_order_final = (
            (step_index == len(self.scheduler.timesteps) - 1)
            and self.scheduler.config.lower_order_final
            and len(self.scheduler.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.scheduler.timesteps) - 2)
            and self.scheduler.config.lower_order_final
            and len(self.scheduler.timesteps) < 15
        )
        model_output = self.scheduler.convert_model_output(
            model_output, timestep, sample
        )
        if (
            self.scheduler.config.solver_order == 1
            or self.scheduler.lower_order_nums < 1
            or lower_order_final
        ):
            return self.scheduler.dpm_solver_first_order_update(
                model_output, timestep, prev_timestep, sample
            )
        elif (
            self.scheduler.config.solver_order == 2
            or self.scheduler.lower_order_nums < 2
            or lower_order_second
        ):
            timestep_list = [self.scheduler.timesteps[step_index - 1], timestep]
            return self.scheduler.multistep_dpm_solver_second_order_update(
                self.scheduler.model_outputs, timestep_list, prev_timestep, sample
            )
        else:
            timestep_list = [
                self.scheduler.timesteps[step_index - 2],
                self.scheduler.timesteps[step_index - 1],
                timestep,
            ]
            return self.scheduler.multistep_dpm_solver_third_order_update(
                self.scheduler.model_outputs, timestep_list, prev_timestep, sample
            )


#
# -- Initializer --
#
class Initializer:
    def InitializeWithScheduler(self, scheduler: Scheduler):
        raise NotImplementedError()

    def GetLatents(
        self,
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
    ):
        raise NotImplementedError()

    def GetTimesteps(self, scheduler: Scheduler, target: SharedTarget):
        raise NotImplementedError()

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        raise NotImplementedError()

    def NextLatents(self, scheduler: Scheduler, latents, timesteps, step: int):
        raise NotImplementedError()

    @classmethod
    def RandForShape(cls, shape, generator: Generator, target: SharedTarget):
        if target.device_type() == "mps":
            # randn does not work reproducibly on mps
            return torch.randn(
                shape,
                generator=generator,
                device="cpu",
                dtype=taret.dtype(),
            ).to(target.device())
        else:
            return torch.randn(
                shape,
                generator=generator,
                **target.dict,
            )


class Randomly(Initializer):
    def __init__(
        self,
        num_steps: int,
        size: Tuple[int, int],
        symmetric: Optional[bool] = False,
    ):
        """
        Args:
            size (`(int, int)`, *optional*, defaults to (512, 512))
                The (width, height) pair in pixels of the generated image.
        """
        self._requested_num_steps = num_steps
        self._size = size
        self._symmetric = symmetric

    def InitializeWithScheduler(self, scheduler: Scheduler):
        self._actual_num_steps = scheduler.SetTimesteps(self._requested_num_steps)

    def GetLatents(
        self,
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
    ):
        latents_shape = self.GetLatentsShape(image_model)

        if not self._symmetric:
            latents = self.RandForShape(latents_shape, generator, target)
            return latents

        # Making symmetric latents.
        # `(latents + latents.flip(-1)) / 2.0` didn't work.
        width = latents_shape[-1]
        half_width = int((width + 1) / 2)
        half_shape = latents_shape[:-1] + (half_width,)
        left = self.Rand(generator, half_shape, target)
        right = left.flip(-1)
        extra_width = (half_width * 2) - width
        if extra_width > 0:
            right = right[:, :, :, extra_width:]
        return torch.cat([left, right], dim=-1)

    def GetTimesteps(self, scheduler: Scheduler, target: SharedTarget):
        return scheduler.Get().timesteps.to(
            device=target.device()
        )  # dtype should be `int`

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        return latents * scheduler.Get().init_noise_sigma

    def NextLatents(self, scheduler: Scheduler, latents, timesteps, step: int):
        # do nothing
        return latents

    def GetLatentsShape(self, image_model: ImageModel):
        batch_size = 1
        num_channels = image_model.num_channels()
        scale_factor = image_model.vae_scale_factor()
        width, height = self._size
        if height % scale_factor != 0 or width % scale_factor != 0:
            Debug(
                1,
                f"`width` and `height` have to be divisible by {scale_factor}. "
                "Automatically rounded.",
            )

        # get the initial random noise unless the user supplied it
        latents_shape = (
            batch_size,
            num_channels,
            height // scale_factor,
            width // scale_factor,
        )
        return latents_shape


class ByLatents(Initializer):
    def __init__(
        self,
        num_steps: int,
        latents: torch.FloatTensor,
    ):
        """
        Args:
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
        """
        self._requested_num_steps = num_steps
        self._latents = latents

    def InitializeWithScheduler(self, scheduler: Scheduler):
        self._actual_num_steps = scheduler.SetTimesteps(self._requested_num_steps)

    def GetLatents(
        self,
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
    ):
        return self._latents.to(**target.dict)

    def GetTimesteps(self, scheduler: Scheduler, target: SharedTarget):
        return scheduler.Get().timesteps.to(
            device=target.device()
        )  # dtype should be `int`

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        return latents * scheduler.Get().init_noise_sigma

    def NextLatents(self, scheduler: Scheduler, latents, timesteps, step: int):
        # do nothing
        return latents


class ByImage(Initializer):
    def __init__(
        self,
        num_steps: int,
        image: PIL.Image.Image,
        strength: float = 0.8,
        keep_until: float = 0.0,
    ):
        """
        Args:
            image `PIL.Image.Image`:
                `Image`, or tensor representing an image batch, that will be used as the starting point for the process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            keep_until (`float`, *optional*, defaults to 0.0):
                0 if we use the image only initially.
                1 if we keep the image in the final output
                Choose between 0 and 1 to keep the image until the middle of the generation.
        """
        self._image = image
        if (strength <= 0.0) or (strength > 1.0):
            raise ValueError(f"strength must be between 0 and 1. given {strength}.")
        self._strength = strength
        self._requested_num_steps = -(-num_steps // strength)  # round up
        if (fix_until < 0.0) or (fix_until > 1.0):
            raise ValueError(f"strength must be between 0 and 1. given {strength}.")
        self._keep_until = keep_until

    def InitializeWithScheduler(self, scheduler: Scheduler):
        self._actual_total_num_steps = scheduler.SetTimesteps(self._requested_num_steps)
        self._actual_num_steps = int(self._actual_total_num_steps * self._strength)
        self._steps_to_keep = int(self._actual_num_steps * self._keep_until)

    def GetLatents(
        self,
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
    ):
        latents = image_model.Encode(self._image, generator, target)
        self._initial_latents = latents.clone()
        self._noise = self.RandForShape(latents.shape, generator, target)
        return latents

    def GetTimesteps(self, scheduler: Scheduler, target: SharedTarget):
        return (
            scheduler.Get()
            .timesteps[-self._actual_num_steps :]
            .to(device=target.device())  # dtype should be `int`
        )

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        initial_timestep = timesteps[:-1]
        return scheduler.Get().add_noise(
            self._initial_latents, self._noise, initial_timestep
        )

    def NextLatents(self, scheduler: Scheduler, latents, timesteps, step: int):
        if step >= self._steps_to_keep:
            # do nothing
            return latents
        # Return the initial latents if this is the last step.
        if step >= len(timesteps) - 1:
            return self._initial_latents
        # Otherwise add noise to the initial latents.
        next_timestep = timesteps[step + 1 : step + 2]
        return scheduler.Get().add_noise(
            self._initial_latents, self._noise, next_timestep
        )


#
# -- Prompts --
#
class Prompts:
    def __init__(
        self,
        prompt_dict: Dict[str, PromptType],
        text_model: TextModel,
        default_encoding: StandardEncoding,
    ):
        if not prompt_dict:
            raise ValueError("`prompt_dict` must not be empty.")
        self._prompt_dict = prompt_dict
        self._text_model = text_model
        self._default_encoding = default_encoding
        self._key_index = {}
        self._prompts = []

    def Check(self, key: str):
        if key in self._key_index:
            # Already checked
            return
        elif key in self._prompt_dict:
            Debug(3, f"Checked prompt: {key}")
            self._key_index[key] = len(self._prompts)
            self._prompts.append(
                TextEmbeddings.Create(
                    self._prompt_dict[key], self._text_model, self._default_encoding
                )
            )
            return
        elif key == "":
            Debug(3, f"Checked null prompt")
            self._key_index[key] = len(self._prompts)
            self._prompts.append(
                TextEmbeddings.Create("", self._text_model, self._default_encoding)
            )
            return
        else:
            raise ValueError(f"Prompt key is not found: {key}")

    def GetFromList(self, key: str, lst: List[Any]):
        return lst[self._key_index[key]]

    def PredictResiduals(self, scheduler, unet, latents, timestep, progress: float):
        model_input = torch.cat([latents] * len(self._prompts))
        model_input = scheduler.Get().scale_model_input(model_input, timestep)
        text_embeddings = torch.cat([te.Get(progress) for te in self._prompts])
        residuals = unet(
            model_input, timestep, encoder_hidden_states=text_embeddings
        ).sample
        return residuals.chunk(len(self._prompts))


#
# -- Layer --
#
class BaseLayer:
    def __init__(
        self,
        prompt_name: Optional[str] = None,
        negative_prompt_name: Optional[str] = None,
        cfg_scale: float = 7.5,
    ):
        if not prompt_name:
            prompt_name = ""
        self._prompt_name = prompt_name
        if not negative_prompt_name:
            negative_prompt_name = ""
        self._negative_prompt_name = negative_prompt_name
        if cfg_scale < 1.0:
            raise ValueError(f"cfg_scale must be 1.0 or bigger: {cfg_scale}")
        self._cfg_scale = cfg_scale
        self._do_cfg = cfg_scale > 1.0

    def _Initialize(self, prompts: Prompts, target: SharedTarget):
        self._prompts = prompts
        prompts.Check(self._prompt_name)
        if self._do_cfg:
            prompts.Check(self._negative_prompt_name)

    def GetResidual(self, residuals):
        residual_cond = self._prompts.GetFromList(self._prompt_name, residuals)
        if not self._do_cfg:
            return residual_cond
        residual_uncond = self._prompts.GetFromList(
            self._negative_prompt_name, residuals
        )
        residual = residual_uncond + self._cfg_scale * (residual_cond - residual_uncond)
        return residual


class BackgroundLayer(BaseLayer):
    def __init__(
        self,
        initialize: Initializer,
        prompt_name: Optional[str] = None,
        negative_prompt_name: Optional[str] = None,
        cfg_scale: float = 7.5,
    ):
        super().__init__(prompt_name, negative_prompt_name, cfg_scale)
        self._initializer = initialize

    def Initialize(
        self,
        scheduler: Scheduler,
        image_model: ImageModel,
        prompts: Prompts,
        target: SharedTarget,
    ):
        super()._Initialize(prompts, target)
        self._initializer.InitializeWithScheduler(scheduler)
        latents = self._initializer.GetLatents(
            image_model, scheduler.generator(), target
        )
        timesteps = self._initializer.GetTimesteps(scheduler, target)
        self._initializer.InitializeLatents(scheduler, latents, timesteps)
        return latents, timesteps

    def NextLatents(self, scheduler: Scheduler, latents, timesteps, step: int):
        return self._initializer.NextLatents(scheduler, latents, timesteps, step)


class Layer(BaseLayer):
    def __init__(
        self,
        mask: PIL.Image.Image,
        prompt_name: Optional[str] = None,
        negative_prompt_name: Optional[str] = None,
        cfg_scale: float = 7.5,
    ):
        super().__init__(prompt_name, negative_prompt_name, cfg_scale)
        self._mask_image = mask

    def Initialize(
        self, image_model: ImageModel, prompts: Prompts, target: SharedTarget
    ):
        super()._Initialize(prompts, target)
        self._mask = image_model.PreprocessMask(self._mask_image, target)

    def Merge(self, mine, other):
        return (other * self._mask) + (mine * (1.0 - self._mask))


#
# -- Main pipeline --
#
class LayeredDiffusionPipeline:
    def GetRevision(self, dataset: str, revision: Optional[str] = None):
        # revision is a git branch name assigned to the model repository.

        # Always return the provided revision if any.
        if revision:
            return revision

        # We always have the "main" branch.
        default_revision = "main"
        # There may be other branches for smaller memory footprint
        recommended_revision = {
            "stabilityai/stable-diffusion-2": "fp16",
            "CompVis/stable-diffusion-v1-4": "fp16",
            "runwayml/stable-diffusion-v1-5": "fp16",
            "hakurei/waifu-diffusion": "fp16",
            "naclbit/trinart_stable_diffusion_v2": "diffusers-60k",
        }

        return recommended_revision.get(dataset, default_revision)

    def Connect(
        self,
        dataset: str,
        revision: Optional[str] = None,
        auth_token: Optional[str] = None,
        use_xformers: bool = False,
        device_type: str = "cuda",
    ):
        extra_args = {
            "torch_dtype": torch.float32,  # This may be needed to avoid OOM.
            "revision": self.GetRevision(dataset, revision),
        }
        if auth_token:
            extra_args["use_auth_token"] = auth_token

        # use DPM-Solver++ scheduler
        self.scheduler = Scheduler().Connect(dataset)
        extra_args["scheduler"] = self.scheduler.Get()

        # Prepare the StableDiffusion pipeline.
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            dataset, **extra_args
        ).to(device_type)

        # Options for efficient execution
        pipe.enable_attention_slicing()
        if use_xformers and (self._devicetype_str == "cuda"):
            pipe.enable_xformers_memory_efficient_attention()

        self._SetPipeline(pipe)
        return self

    def CopyFrom(self, another):
        self.scheduler = Scheduler().CopyFrom(another.scheduler)
        self._SetPipeline(another.pipe)
        return self

    def _SetPipeline(self, pipe):
        self.pipe = pipe
        self.unet = pipe.unet
        self.text_model = TextModel(pipe.tokenizer, pipe.text_encoder, pipe.device)
        self.image_model = ImageModel(pipe.vae, pipe.vae_scale_factor)

    def ResetGenerator(self, rand_seed: Optional[int] = None):
        self.scheduler.ResetGenerator(self.text_model.target, rand_seed)

    @torch.no_grad()
    def __call__(
        self,
        prompts: Dict[str, PromptType],
        initialize: Initializer,
        default_encoding: Optional[StandardEncoding] = None,
        prompt_name: Optional[str] = None,
        negative_prompt_name: Optional[str] = None,
        cfg_scale: float = 7.5,
        layers: Optional[List[Layer]] = None,
        eta: float = 0.0,
    ):
        if not default_encoding:
            default_encoding = ShiftEncoding()
        if not prompt_name:
            prompt_name = ""
        if not negative_prompt_name:
            negative_prompt_name = ""
        if not layers:
            layers = []
        bglayer = BackgroundLayer(
            initialize, prompt_name, negative_prompt_name, cfg_scale
        )
        all_layers = [bglayer] + layers
        prompts = Prompts(prompts, self.text_model, default_encoding)

        target = self.text_model.target
        with autocast(target.device_type()):
            if layers:
                for l in reversed(layers):
                    l.Initialize(self.image_model, prompts, target)
            latents, timesteps = bglayer.Initialize(
                self.scheduler, self.image_model, prompts, target
            )
            self.scheduler.PrepareExtraStepKwargs(self.pipe, eta)

            for step, ts in enumerate(self.pipe.progress_bar(timesteps)):
                progress = float(step) / len(timesteps)
                residuals_for_prompts = prompts.PredictResiduals(
                    self.scheduler, self.unet, latents, ts, progress
                )
                residuals_for_layers = [
                    l.GetResidual(residuals_for_prompts) for l in all_layers
                ]
                latents_for_layers = self.scheduler.Step(
                    residuals_for_layers, ts, latents
                )
                next_latents = bglayer.NextLatents(
                    self.scheduler, latents_for_layers[0], timesteps, step
                )
                if layers:
                    for layer, latents in zip(
                        reversed(layers), reversed(latents_for_layers[1:])
                    ):
                        next_latents = layer.Merge(latents, next_latents)
                latents = next_latents

            return self.image_model.Decode(latents)
