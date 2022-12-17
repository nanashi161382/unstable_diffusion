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
import random
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


def ShouldDebug(level: int):
    return level <= debug_level


def Debug(level: int, obj):
    if ShouldDebug(level):
        display(obj)


#
# -- Type: Generator --
#
Generator = Optional[torch.Generator]


def OpenImage(filename):
    if filename:
        image = PIL.Image.open(filename).convert("RGB")
        if ShouldDebug(1):
            print(f"{image.size} - {filename}")
            display(image.resize((128, 128), resample=PIL.Image.LANCZOS))
        return image
    else:
        return None


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
        self,
        image: PIL.Image.Image,
        generator: Generator,
        target: SharedTarget,
        deterministic: bool = False,
    ):
        image = self.Preprocess(image).to(**target.dict)

        # encode the init image into latents and scale the latents
        latent_dist = self._vae.encode(image).latent_dist
        if deterministic:
            latents = latent_dist.mode()
        else:
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
# -- LatentMask --
#
class LatentMask:
    def __init__(self, mask_by: Union[float, str, PIL.Image.Image]):
        self._mask_by = mask_by

    def Initialize(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        target: SharedTarget,
    ):
        mask = self._mask_by
        if isinstance(mask, str):
            mask = OpenImage(mask)
        if isinstance(mask, PIL.Image.Image):
            if size and (mask.size != size):
                Debug(1, f"Resize mask image from {mask.size} to {size}.")
                mask = mask.resize(size)
            self._mask = image_model.PreprocessMask(mask, target)
        else:
            self._mask = 1.0 - mask  # make the scale same as the mask image.

    def Merge(self, black, white):
        return (black * self._mask) + (white * (1.0 - self._mask))


MaskType = Union[float, str, PIL.Image.Image, LatentMask]


class ComboOf(LatentMask):
    def __init__(
        self,
        masks: List[MaskType],
        # white = 1.0 and black = 0.0 in the arguments of the callable below.
        transformed_by: Callable[[List[Union[float, torch.Tensor]]], LatentMask],
    ):
        self._f = transformed_by
        self._masks = [m if isinstance(m, LatentMask) else LatentMask(m) for m in masks]

    def Initialize(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        target: SharedTarget,
    ):
        for m in self._masks:
            m.Initialize(size, image_model, target)
        self._mask = 1.0 - self._f([1.0 - m._mask for m in self._masks])


def ScaledMask(mask: MaskType, scale: float = 1.0, offset: float = 0.0):
    return ComboOf([mask], transformed_by=(lambda masks: (masks[0] * scale) + offset))


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
        if not rand_seed:
            return
        Debug(1, f"Setting random seed to {self._rand_seed}")
        if rand_seed and (not self._generator):
            self._generator = torch.Generator(device=target.device())
        self._generator.manual_seed(rand_seed)

    def generator(self):
        if not self._rand_seed:
            return None
        return self._generator

    def SetTimesteps(self, num_inference_steps):
        # Some schedulers like DDIM and PNDM doesn't seem to handle longer inference steps
        # than those in training.
        if self.have_max_timesteps and (len(self.scheduler) < num_inference_steps):
            num_inference_steps = len(self.scheduler)
        self.scheduler.set_timesteps(num_inference_steps)

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
    def GetStrength(self):
        return 1.0

    def GetLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
        vae_encoder_adjust: float,
    ):
        raise NotImplementedError()

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        raise NotImplementedError()

    def OverwriteLatents(
        self,
        scheduler: Scheduler,
        latents,
        timesteps,
        current_step: int,
        remaining: float,
        force_overwrite: bool,
    ):
        # do nothing
        return latents, False

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
        symmetric: Optional[bool] = False,
    ):
        """
        Args:
            size (`(int, int)`, *optional*, defaults to (512, 512))
                The (width, height) pair in pixels of the generated image.
        """
        self._symmetric = symmetric

    def GetLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
        vae_encoder_adjust: float,
    ):
        if not size:
            raise ValueError("`size` is mandatory to initialize `Randomly`.")

        latents_shape = self.GetLatentsShape(image_model, size)

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

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        return latents * scheduler.Get().init_noise_sigma

    def GetLatentsShape(self, image_model: ImageModel, size: Tuple[int, int]):
        batch_size = 1
        num_channels = image_model.num_channels()
        scale_factor = image_model.vae_scale_factor()
        width, height = size
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
        latents: torch.FloatTensor,
    ):
        """
        Args:
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
        """
        self._latents = latents

    def GetLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
        vae_encoder_adjust: float,
    ):
        if size:
            Debug(0, "`size` is ignored when initializing `ByLatents`.")
        return self._latents.to(**target.dict)

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        return latents * scheduler.Get().init_noise_sigma


class ByImage(Initializer):
    def __init__(
        self,
        image: Union[str, PIL.Image.Image],
        strength: float = 0.8,
    ):
        """
        Args:
            image `PIL.Image.Image`:
                `Image`, or tensor representing an image batch, that will be used as the starting point for the process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
        """
        self._image = image
        if (strength <= 0.0) or (strength > 1.0):
            raise ValueError(f"strength must be between 0 and 1. given {strength}.")
        self._strength = strength

    def GetStrength(self):
        return self._strength

    def GetLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
        vae_encoder_adjust: float,
    ):
        image = self._image
        if isinstance(image, str):
            image = OpenImage(image)
        if size and (image.size != size):
            Debug(1, f"Resize image from {image.size} to {size}.")
            image = image.resize(size, resample=PIL.Image.LANCZOS)
        latents = image_model.Encode(image, generator, target) * vae_encoder_adjust
        self._initial_latents = latents
        self._noise = self.RandForShape(latents.shape, generator, target)
        return latents

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        initial_timestep = timesteps[:1]
        latents = scheduler.Get().add_noise(
            self._initial_latents, self._noise, initial_timestep
        )
        return latents

    def OverwriteLatents(
        self,
        scheduler: Scheduler,
        latents,
        timesteps,
        current_step: int,
        remaining: float,
        force_overwrite: bool,
    ):
        Debug(3, f"remaining {remaining:.2f}, strength {self._strength:.2f}")
        if (not force_overwrite) and (remaining < self._strength):
            # do nothing if no need to keep the initial latents
            return latents, False

        # Return the initial latents without noise if this is the last step.
        if remaining <= 0.0:
            Debug(3, "Last step with keeping initial image.")
            return self._initial_latents, True
        # Otherwise add noise to the initial latents.
        Debug(3, f"Add noise for step {current_step}.")
        next_timestep = timesteps[current_step + 1 : current_step + 2]
        return (
            scheduler.Get().add_noise(
                self._initial_latents, self._noise, next_timestep
            ),
            True,
        )


class ByBothOf(Initializer):
    def __init__(
        self,
        black: Initializer,
        white: Initializer,
        mask_by: MaskType,
    ):
        self._both = [black, white]
        if isinstance(mask_by, LatentMask):
            self._mask = mask_by
        else:
            self._mask = LatentMask(mask_by)

    def GetStrength(self):
        return max(g.GetStrength() for g in self._both)

    def GetLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
        vae_encoder_adjust: float,
    ):
        self._mask.Initialize(size, image_model, target)
        return [
            g.GetLatents(size, image_model, generator, target, vae_encoder_adjust)
            for g in self._both
        ]

    def InitializeLatents(self, scheduler: Scheduler, latents, timesteps):
        latents = [
            g.InitializeLatents(scheduler, l, timesteps)
            for g, l in zip(self._both, latents)
        ]
        return self._mask.Merge(*latents)

    def OverwriteLatents(
        self,
        scheduler: Scheduler,
        latents,
        timesteps,
        current_step: int,
        remaining: float,
        force_overwrite: bool,
    ):
        new_latents, changed = zip(
            *[
                g.OverwriteLatents(
                    scheduler,
                    latents,
                    timesteps,
                    current_step,
                    remaining,
                    force_overwrite,
                )
                for g in self._both
            ]
        )
        if any(changed):
            return self._mask.Merge(*new_latents), True
        else:
            # Return the original latents if nothing chnaged.
            return latents, False


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
        vae_encoder_adjust: Optional[float] = None,
    ):
        # The default image generation = no prompts, cfg_scale = 1.0
        super().__init__(cfg_scale=1.0)
        self._initializer = initialize
        if not vae_encoder_adjust:
            self._vae_encoder_adjust = 1.0
        else:
            self._vae_encoder_adjust = vae_encoder_adjust

    def Initialize(
        self,
        num_steps: int,
        size: Optional[Tuple[int, int]],
        scheduler: Scheduler,
        image_model: ImageModel,
        prompts: Prompts,
        target: SharedTarget,
    ):
        super()._Initialize(prompts, target)
        max_strength = self._initializer.GetStrength()
        # compute num_steps_to_request to match the effective number of iterations with num_steps.
        num_steps_to_request = int(-(-int(num_steps) // max_strength))  # round up
        scheduler.SetTimesteps(num_steps_to_request)
        latents = self._initializer.GetLatents(
            size, image_model, scheduler.generator(), target, self._vae_encoder_adjust
        )
        timesteps, total_num_steps = self.GetTimesteps(scheduler, max_strength, target)
        Debug(
            3, f"total_num_steps: {total_num_steps}, actual num_steps: {len(timesteps)}"
        )
        self._total_num_steps = total_num_steps
        latents = self._initializer.InitializeLatents(scheduler, latents, timesteps)
        return latents, timesteps

    def GetTimesteps(self, scheduler: Scheduler, strength: float, target: SharedTarget):
        # dtype should be `int`
        timesteps = scheduler.Get().timesteps.to(device=target.device())
        total_num_steps = len(timesteps)
        if strength > 1.0:
            raise ValueError(f"invalid strength: {strength}")
        if strength == 1.0:
            return timesteps, total_num_steps
        actual_num_steps = int(total_num_steps * strength)
        return timesteps[-actual_num_steps:], total_num_steps

    def OverwriteLatents(
        self,
        scheduler: Scheduler,
        latents,
        timesteps,
        current_step: int,
        force_overwrite: bool = False,
    ):
        remaining = float(len(timesteps) - current_step - 1) / self._total_num_steps
        return self._initializer.OverwriteLatents(
            scheduler, latents, timesteps, current_step, remaining, force_overwrite
        )[0]


class Layer(BaseLayer):
    def __init__(
        self,
        prompt_name: Optional[str] = None,
        negative_prompt_name: Optional[str] = None,
        cfg_scale: float = 7.5,
        mask_by: MaskType = 1.0,
        skip_until: float = 0.0,
    ):
        super().__init__(prompt_name, negative_prompt_name, cfg_scale)
        if isinstance(mask_by, LatentMask):
            self._mask = mask_by
        else:
            self._mask = LatentMask(mask_by)
        if (skip_until < 0.0) or (1.0 < skip_until):
            raise ValueError(f"skip_until must be between 0 and 1. actual {skip_until}")
        self._skip_until = skip_until

    def Initialize(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        prompts: Prompts,
        target: SharedTarget,
    ):
        super()._Initialize(prompts, target)
        self._mask.Initialize(size, image_model, target)

    def Merge(self, other, mine, progress):
        if progress <= self._skip_until:
            return other
        # other = black, mine = white
        return self._mask.Merge(other, mine)


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
        self._dataset = dataset
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
        self._dataset = another._dataset
        self.scheduler = Scheduler().CopyFrom(another.scheduler)
        self._SetPipeline(another.pipe)
        return self

    def _SetPipeline(self, pipe):
        self.pipe = pipe
        self.unet = pipe.unet
        self.text_model = TextModel(pipe.tokenizer, pipe.text_encoder, pipe.device)
        self.image_model = ImageModel(pipe.vae, pipe.vae_scale_factor)

    def _ResetGenerator(self, rand_seed: Optional[int] = None):
        if not rand_seed:
            rand_seed = random.SystemRandom().randint(1, 4294967295)
        self._rand_seed = rand_seed
        self.scheduler.ResetGenerator(self.text_model.target, rand_seed)

    def GetRandSeed(self):
        return self._rand_seed

    @torch.no_grad()
    def __call__(
        self,
        num_steps: int,
        prompts: Dict[str, PromptType],
        initialize: Initializer,
        layers: List[Layer],
        size: Optional[Tuple[int, int]] = None,
        default_encoding: Optional[StandardEncoding] = None,
        rand_seed: Optional[int] = None,
        eta: float = 0.0,
        vae_encoder_adjust: Optional[float] = None,
    ):
        self._ResetGenerator(rand_seed)

        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0. actual: {num_steps}")
        if not default_encoding:
            default_encoding = ShiftEncoding()
        if not layers:
            raise ValueError("layers should contain at least 1 layer.")

        # Anything v3.0's VAE has a degradation problem in its encoder.
        # Multiplying the adjustment factor of 1.25 to the encoder output mitigates
        # the problem to a lower level.
        if not vae_encoder_adjust:
            if self._dataset == "Linaqruf/anything-v3.0":
                vae_encoder_adjust = 1.25

        bglayer = BackgroundLayer(initialize, vae_encoder_adjust)
        all_layers = [bglayer] + layers
        prompts = Prompts(prompts, self.text_model, default_encoding)

        target = self.text_model.target
        with autocast(target.device_type()):
            if layers:
                for l in reversed(layers):
                    l.Initialize(size, self.image_model, prompts, target)
            latents, timesteps = bglayer.Initialize(
                num_steps, size, self.scheduler, self.image_model, prompts, target
            )
            self.scheduler.PrepareExtraStepKwargs(self.pipe, eta)

            for i, ts in enumerate(self.pipe.progress_bar(timesteps)):
                progress = float(i + 1) / len(timesteps)
                Debug(3, f"progress: {progress} <= {i + 1} / {len(timesteps)}")
                residuals_for_prompts = prompts.PredictResiduals(
                    self.scheduler, self.unet, latents, ts, progress
                )
                residuals_for_layers = [
                    l.GetResidual(residuals_for_prompts) for l in all_layers
                ]

                # TODO: implement residual inspection
                # latents_debug = self.scheduler.InspectLatents(
                #    residuals_for_layers[0], ts, latents
                # )
                # Debug(3, self.image_model.Decode(latents_debug)[0])

                latents_for_layers = self.scheduler.Step(
                    residuals_for_layers, ts, latents
                )
                next_latents = latents_for_layers[0]
                next_latents = bglayer.OverwriteLatents(  # unconditionally
                    self.scheduler, next_latents, timesteps, i, force_overwrite=True
                )
                if layers:
                    for layer, latents in zip(layers, latents_for_layers[1:]):
                        next_latents = layer.Merge(next_latents, latents, progress)
                next_latents = bglayer.OverwriteLatents(  # conditioned by strength
                    self.scheduler, next_latents, timesteps, i
                )
                latents = next_latents

            return self.image_model.Decode(latents)
