# @title LayeredDiffusionPipeline
# See the following web page for the usage.
# https://github.com/nanashi161382/unstable_diffusion/tree/main
import accelerate
import copy
import cv2
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
)
from diffusers.loaders import TextualInversionLoaderMixin
import diffusers.pipelines.stable_diffusion.convert_from_ckpt as convert_from_ckpt
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
    # load_pipeline_from_original_stable_diffusion_ckpt,
)
from IPython.display import display
import numpy as np
import PIL
from PIL import ImageDraw, ImageFont
import random
import re
from safetensors import safe_open
import torch
import torch.nn as nn
from torch import autocast
from transformers import (
    AutoProcessor,
    CLIPPreTrainedModel,
    CLIPTextConfig,
    CLIPVisionModelWithProjection,
)
from transformers.activations import ACT2FN
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPMLP
from typing import Any, Optional, List, Union, Callable, Tuple, Dict


#
# -- Debug functions --
#
# Debug level:
#   0: Warn
#   1: Info  <-  Default
#   2: Temporary
#   3: Debug
#   4: Debug in loop
#   5: Trace
#
# Should raise an exception for Error.
debug_level = 1


def SetDebugLevel(level: int):
    global debug_level
    debug_level = level


def ShouldDebug(level: int):
    return level <= debug_level


def Debug(level: int, title: str, obj=None):
    if ShouldDebug(level):
        if title:
            print(title)
        if obj is not None:
            display(obj)


#
# -- Type: Generator --
#
Generator = Optional[torch.Generator]

XXS_SIZE = 96
XS_SIZE = 128
S_SIZE = 256
M_SIZE = 512

FONTNAME = "LiberationMono-Regular.ttf"  # This font is available on Google Colab


def ResizeImage(image, height):
    w, h = image.size
    ratio = float(height) / h
    width = round(ratio * w)
    return image.resize((width, height), resample=PIL.Image.LANCZOS)


def ConcatImages(images, titles):
    max_height = 0
    total_width = 0
    for img in images:
        w, h = img.size
        max_height = max(h, max_height)
        total_width += w
    output = PIL.Image.new("RGB", (total_width, max_height + 20), "white")
    draw = ImageDraw.Draw(output)
    font = ImageFont.truetype(FONTNAME, size=16)
    current_x = 0
    for img, title in zip(images, titles):
        w, h = img.size
        draw.text((current_x, 0), title, "black", font=font)
        output.paste(img, (current_x, 20))
        current_x += w
    return output


def CreateMaskImage(image):
    sz = image.size
    bg = PIL.Image.new("RGB", sz, "black")
    fg = PIL.Image.new("RGB", sz, "white")
    bg.paste(fg, (0, 0), image)
    return bg.convert("L")


def CreateRgbImage(image, background="white"):
    sz = image.size
    bg = PIL.Image.new("RGB", sz, background)
    bg.paste(image, (0, 0))
    return bg


def OpenImage(filename):
    if filename:
        image = PIL.Image.open(filename)
        if any(a in image.mode for a in "Aa"):
            # image with alpha
            mask = CreateMaskImage(image)
            image = CreateRgbImage(image)
            Debug(
                1,
                f"{image.size} - {filename}",
                ConcatImages(
                    [ResizeImage(image, XXS_SIZE), ResizeImage(mask, XXS_SIZE)],
                    ["image", "mask"],
                ),
            )
            return image, mask
        else:
            Debug(1, f"{image.size} - {filename}", ResizeImage(image, XXS_SIZE))
            return image.convert("RGB"), None
    else:
        return None, None


def MaskImage(image, mask, background="white"):
    sz = image.size
    bg = PIL.Image.new("RGB", sz, background)
    bg.paste(image, (0, 0), mask)
    return bg


def OpenImageWithBackground(filename, background="white"):
    image, mask = OpenImage(filename)
    if not image:
        return None
    elif not mask:
        return image
    else:
        return MaskImage(image, mask, background)


#
# -- Canny --
#
def GradationCanny(degree=1.0):
    def f(image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, PIL.Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        thresholds = [16, 32, 64, 128, 256, 512, 1024]
        weights = []
        for k in range(1, len(thresholds) + 1):
            weights.append((k / len(thresholds)) ** degree)
        canny_image = 0
        for threshold, weight in zip(thresholds, weights):
            canny_image = np.maximum(
                canny_image, cv2.Canny(image, threshold, threshold) * weight
            )
        return PIL.Image.fromarray(canny_image)

    return f


#
# -- RangeMap --
#
class RangeMap:
    """
    RangeMap is a class that assigns a value to each consecutive time segment.
    The value can be anything including a number and a string.
    The time segment is a range of the number between 1.0 and 0.0 in the decreasing order.
    This class is used for `strength`, `cfg_scale`, `prompt` and `negative_prompt`.
    Examples:
      * strength = using(0.0, until=0.8).then(0.6, until=0.5).lastly(1.0)
      * strength = using(0.6, after=0.8, until=0.5)
      * strength = 0.8
      * prompt = using("1girl", until=0.8).then("1girl, lowres")
      * prompt = using(("lowres", StandardEncoding), after=0.8)
      * prompt = "1girl"
    """

    def __init__(self, value, until):
        self._start = None
        self._end = None
        if (value is None) and (until is None):
            self._segments = []
        else:
            self._segments = [(value, until)]

    def then(self, value, until=None):
        if value is None:
            raise ValueError(f"value must not be None.")
        self._segments.append((value, until))
        return self

    def lastly(self, value):
        return self.then(value)

    def Finalize(self, start, end, value_finalizer=None):
        if (start is None) or (end is None):
            raise ValueError(f"Finalize requires non-null `start` and `end` arguments.")
        self._start = start
        self._end = end
        if value_finalizer:
            self._segments = [
                (None if value is None else value_finalizer(value), until)
                for value, until in self._segments
            ]
        return self

    def Get(self, remaining, debug_label):
        for k, (value, until) in enumerate(self._segments):
            if (until is None) or (remaining > until):
                if value is None:
                    Debug(
                        4,
                        f"RangeMap at {remaining:.2f} for {debug_label}: start",
                        self._start if ShouldDebug(5) else None,
                    )
                    return self._start
                else:
                    Debug(
                        4,
                        f"RangeMap at {remaining:.2f} for {debug_label}: {k}",
                        value if ShouldDebug(5) else None,
                    )
                    return value
        Debug(
            4,
            f"RangeMap at {remaining:.2f} for {debug_label}: end",
            self._end if ShouldDebug(5) else None,
        )
        if remaining >= 1.0:
            return self._start
        else:
            return self._end

    def GetFirst(self):
        if len(self._segments) == 0:
            if self._end == self._start:
                return self._end, 0.0
            else:
                return self._start, 1.0
        value, until = self._segments[0]
        if (value is None) or (value == self._start):
            if len(self._segments) == 1:
                return self._end, until
            else:
                return self._segments[1][0], until
        else:
            return value, 1.0

    def IsConstant(self):
        if len(self._segments) == 0:
            return True, self._end
        if len(self._segments) > 1:
            return False, None
        value, until = self._segments[0]
        if (until is None) or (until < 0.0):
            return True, self._start if value is None else value
        elif until >= 1.0:
            return True, self._end
        return False, None

    def IsFinalized(self):
        return self._start is not None

    def __str__(self):
        if self.IsFinalized():
            return f"RangeMap: start = {self._start}, end = {self._end}, segments = {str(self._segments)}"
        else:
            return f"RangeMap: not finalized, segments = {str(self._segments)}"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    @classmethod
    def CreateFromSegments(cls, segments):
        if not segments:
            return cls(None, None)

        rm = cls(*segments[0])
        for s in segments[1:]:
            rm.then(*s)
        return rm

    @classmethod
    def CreateWithUntil(cls, until):
        return cls(None, until=float(until))

    @classmethod
    def CreateWithValue(cls, value):
        return cls(value, until=None)

    @classmethod
    def CreateAndFinalize(cls, arg, is_value, start, end, value_finalizer=None):
        if arg is None:
            rm = cls(None, None)
        elif isinstance(arg, list):
            rm = cls.CreateFromSegments(arg)
        elif isinstance(arg, cls):
            rm = arg
        elif is_value:
            rm = cls.CreateWithValue(arg)
        else:
            rm = cls.CreateWithUntil(arg)
        Debug(3, "Finalize RangeMap for input:", arg)
        rm.Finalize(start, end, value_finalizer)
        return rm


def using(value, after=None, until=None):
    if value is None:
        raise ValueError(f"value must not be None.")
    elif after is None:
        return RangeMap(value, until=until)
    else:
        return RangeMap(None, until=after).then(value, until=until)


#
# -- Share the computation target throughout the pipeline. --
#
class SharedTarget:
    def __init__(self, device, dtype):
        Debug(1, f"SharedTarget is created for {device}, {dtype}")
        if not device:
            raise ValueError(f"device should not be None.")
        if not dtype:
            raise ValueError(f"dtype should not be None.")
        if isinstance(device, str):
            device = torch.device(device)
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

    def __str__(self):
        return f"({self.device_type()}, {self.dtype()})"


#
# -- CLIPTextDeprojector
#
class CLIPTextDeprojectorConfig(CLIPTextConfig):
    model_type = "clip_text_deprojector_model"
    default_ensemble_size = 1
    default_relative_to_null = False
    default_relative_to_prev = False
    default_apply_mlp_to_input = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ensemble_size = kwargs.get(
            "ensemble_size", self.__class__.default_ensemble_size
        )
        self.relative_to_null = kwargs.get(
            "relative_to_null", self.__class__.default_relative_to_null
        )
        self.relative_to_prev = kwargs.get(
            "relative_to_prev", self.__class__.default_relative_to_prev
        )
        self.apply_mlp_to_input = kwargs.get(
            "apply_mlp_to_input", self.__class__.default_apply_mlp_to_input
        )


class CLIPTextDeprojectorBase(CLIPPreTrainedModel):
    config_class = CLIPTextDeprojectorConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextDeprojectorConfig):
        super().__init__(config)

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class CLIPTextDeprojector(CLIPTextDeprojectorBase):
    default_fuse = 0.0

    def __init__(self, config: CLIPTextDeprojectorConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.register_buffer("sos_embed", torch.zeros([embed_dim]))
        self.register_buffer("null_embed", torch.zeros([embed_dim]))

        self.projection = nn.Linear(config.projection_dim, embed_dim, bias=False)
        for param in self.projection.parameters():
            param.requires_grad = False  # Fix the parameter of the projection layer.

        if config.ensemble_size == 1:
            # Treat config.ensemble_size == 1 specially due to a historical reason.
            if self.config.apply_mlp_to_input:
                self.mlp_to_input = CLIPMLP(config)
            self.position_embedding = nn.Embedding(
                config.max_position_embeddings, embed_dim
            )
            self.encoder_layer = CLIPEncoderLayer(config)
            self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            if self.config.apply_mlp_to_input:
                self.mlp_to_input = nn.ModuleList(
                    [CLIPMLP(config) for _ in range(config.ensemble_size)]
                )
            self.position_embedding = nn.ModuleList(
                [
                    nn.Embedding(config.max_position_embeddings, embed_dim)
                    for _ in range(config.ensemble_size)
                ]
            )
            self.encoder_layer = nn.ModuleList(
                [CLIPEncoderLayer(config) for _ in range(config.ensemble_size)]
            )
            self.final_layer_norm = nn.ModuleList(
                [
                    nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
                    for _ in range(config.ensemble_size)
                ]
            )

    def ConstructInput(self, embeds, prev_output, fn_to_input, from_projected=False):
        seq_len = self.config.max_position_embeddings
        bsz, embed_dim = embeds.size()
        device = embeds.device
        if prev_output is None:
            prev_len = 0
        else:
            prev_len = prev_output.size()[1]

        null_embeds = self.null_embed.reshape([1, 1, embed_dim]).repeat([bsz, 1, 1])

        if from_projected:
            embeds = self.projection(embeds)
        embeds = embeds.unsqueeze(1)
        if self.config.relative_to_null:
            embeds = embeds - null_embeds

        if fn_to_input:
            result = [fn_to_input(embeds), embeds]
        else:
            result = [null_embeds, embeds]

        result_len = 2
        if prev_len > 1:
            prev_output = prev_output[:, 1:, :]
            if self.config.relative_to_prev:
                prev_output = prev_output - torch.cat(
                    [
                        null_embeds,
                        prev_output[:, :-1, :],
                    ],
                    dim=1,
                )
            elif self.config.relative_to_null:
                prev_output = prev_output - null_embeds.repeat([1, prev_len - 1, 1])
            result.append(prev_output)
            result_len += prev_len - 1

        len_to_fill = seq_len - result_len
        if len_to_fill > 0:
            result.append(torch.zeros([bsz, len_to_fill, embed_dim]).to(device))
        elif len_to_fill < 0:
            result[-1] = result[-1][:, :len_to_fill, :]

        result = torch.cat(result, dim=1)
        return result

    def AddPrevOutput(self, layer_output, prev_output):
        bsz, seq_len, embed_dim = layer_output.size()
        device = layer_output.device
        if prev_output is None:
            prev_len = 0
        else:
            prev_len = prev_output.size()[1]

        base_output = [self.null_embed.reshape([1, 1, embed_dim]).repeat([bsz, 2, 1])]

        len_to_fill = seq_len - prev_len - 1
        if prev_len < 2:
            base_output.append(torch.zeros([bsz, seq_len - 2, embed_dim]).to(device))
        elif len_to_fill > 0:
            base_output.append(prev_output[:, 1:, :])
            base_output.append(torch.zeros([bsz, len_to_fill, embed_dim]).to(device))
        elif len_to_fill == 0:
            base_output.append(prev_output[:, 1:, :])
        else:
            base_output.append(prev_output[:, 1:len_to_fill, :])

        return layer_output + torch.cat(base_output, dim=1)

    def forward(self, embeds, prev_outputs, **kwargs):
        if isinstance(self.encoder_layer, nn.ModuleList):
            if self.config.apply_mlp_to_input:
                mlp_to_input_fn = self.mlp_to_input
            position_embedding_fn = self.position_embedding
            encoder_layer_fn = self.encoder_layer
            final_layer_norm_fn = self.final_layer_norm
        else:
            if self.config.apply_mlp_to_input:
                mlp_to_input_fn = [self.mlp_to_input]
            position_embedding_fn = [self.position_embedding]
            encoder_layer_fn = [self.encoder_layer]
            final_layer_norm_fn = [self.final_layer_norm]

        if self.config.apply_mlp_to_input:
            hidden_states = [
                self.ConstructInput(embeds, p, fn, **kwargs)
                for p, fn in zip(prev_outputs, mlp_to_input_fn)
            ]
        else:
            hidden_states = [
                self.ConstructInput(embeds, p, None, **kwargs) for p in prev_outputs
            ]
        bsz, seq_len, embed_dim = hidden_states[0].size()

        attention_mask = None
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states[0].dtype
        ).to(hidden_states[0].device)

        position_embeddings = [fn(self.position_ids) for fn in position_embedding_fn]
        hidden_states = [hs + pe for hs, pe in zip(hidden_states, position_embeddings)]

        layer_outputs = [
            fn(hs, attention_mask, causal_attention_mask)[0]
            for fn, hs in zip(encoder_layer_fn, hidden_states)
        ]
        if self.config.relative_to_prev:
            layer_outputs = [
                self.AddPrevOutput(lo, po)
                for lo, po in zip(layer_outputs, prev_outputs)
            ]
        output = [fn(lo) for fn, lo in zip(final_layer_norm_fn, layer_outputs)]

        sos_embeds = self.sos_embed.reshape([1, 1, embed_dim]).repeat([bsz, 1, 1])
        return [torch.cat([sos_embeds, o[:, 1:]], dim=1) for o in output]

    def Inference(self, embeds, **kwargs):
        self.eval()
        ensemble_len = self.config.ensemble_size
        seq_len = self.config.max_position_embeddings

        result = None
        result_len = 1
        while True:
            inference_input = self.PrepareInferenceInput(result, ensemble_len, **kwargs)
            result = self(embeds, inference_input, **kwargs)
            result_len += 1
            if result_len >= seq_len:
                break
            result = [r[:, :result_len, :] for r in result]

        if ensemble_len == 1:
            return result[0]
        else:
            return sum(result) / ensemble_len

    def PrepareInferenceInput(self, prev_output, ensemble_len, fuse=None, **kwargs):
        if not isinstance(prev_output, list):
            return [prev_output] * ensemble_len
        if ensemble_len == 1:
            return prev_output

        if fuse is None:
            fuse = self.__class__.default_fuse
        if fuse == 0.0:
            return prev_output

        first = [o[:-1, :] for o in prev_output]
        last = [o[-1:, :] for o in prev_output]
        avg_last = sum(last) / ensemble_len
        last = [(1.0 - fuse) * last_i + fuse * avg_last for last_i in last]
        return [torch.cat([f, l], dim=0) for f, l in zip(first, last)]

    @classmethod
    def from_units(cls, models):
        if any(isinstance(model.encoder_layer, nn.ModuleList) for model in models):
            raise ValueError(f"The length of all `models` must be 1.")
        new_config = copy.deepcopy(models[0].config)
        new_config.ensemble_size = len(models)
        new_model = cls(new_config)
        new_model.position_ids = models[0].position_ids
        new_model.sos_embed = models[0].sos_embed
        new_model.null_embed = models[0].null_embed

        def copy_params(a, b):
            for pa, pb in zip(a.parameters(), b.parameters()):
                pa.data = pb.data

        copy_params(new_model.projection, models[0].projection)
        if new_config.ensemble_size == 1:
            if new_config.apply_mlp_to_input:
                copy_params(new_model.mlp_to_input, models[0].mlp_to_input)
            copy_params(new_model.position_embedding, models[0].position_embedding)
            copy_params(new_model.encoder_layer, models[0].encoder_layer)
            copy_params(new_model.final_layer_norm, models[0].final_layer_norm)
        else:
            for i, model in enumerate(models):
                if new_config.apply_mlp_to_input:
                    copy_params(new_model.mlp_to_input[i], models[i].mlp_to_input)
                copy_params(new_model.position_embedding[i], model.position_embedding)
                copy_params(new_model.encoder_layer[i], model.encoder_layer)
                copy_params(new_model.final_layer_norm[i], model.final_layer_norm)
        return new_model


class CLIPTextDeprojectorMerge:
    def __init__(self, models: List[CLIPTextDeprojector]):
        if not models:
            raise ValueError(f"At least one model must be provided.")
        model = CLIPTextDeprojector(models[0].config)
        state_dict = model.state_dict()
        for key in state_dict:
            state_sum = sum(m.state_dict()[key] for m in models)
            state_dict[key] = state_sum / len(models)
        model.load_state_dict(state_dict)
        self.model = model

    def Inference(self, embeds, **kwargs):
        return self.model.Inference(embeds, **kwargs)


class CLIPTextDeprojectorEnsemble2:
    def __init__(self, models: List[CLIPTextDeprojector]):
        if not models:
            raise ValueError(f"At least one model must be provided.")
        self.models = models

    def call(self, embeds, prev_output, **kwargs):
        result = sum(model(embeds, prev_output, **kwargs) for model in self.models)
        return result / len(self.models)

    def Inference(self, embeds, **kwargs):
        for model in self.models:
            model.eval()
        seq_len = self.models[0].config.max_position_embeddings
        result_len = 2
        result = self.call(embeds, None, **kwargs)[:, :result_len, :]
        while True:
            result_len += 1
            result = self.call(embeds, result, **kwargs)[:, :result_len, :]
            if result_len == seq_len:
                return result


#
# --  TextModel - a wrapper to the CLIP tokenizer and the CLIP text encoder --
#
class TextModel(TextualInversionLoaderMixin):
    def __init__(self, tokenizer, text_encoder, device):
        """
        Args:
            tokenizer:
            text_encoder:
            device:
        """
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        dtype = text_encoder.config.torch_dtype
        if not dtype:
            Debug(1, f"Text encoder doesn't have dtype. Setting float32 as default.")
            dtype = torch.float32
        self.target = SharedTarget(device, dtype)
        self.mask_after_eos = 0.0
        self.random_after_eos = 0.0
        self.null_emb_after_eos = 0.0

        # deprecated
        self.deprojector = None
        self.use_deprojector = False

    def SetMaskAfterEOS(self, mask):
        """
        Experimental. Remove later.
        """
        self.mask_after_eos = mask

    def SetRandomAfterEOS(self, mask):
        """
        Experimental. Remove later.
        """
        self.random_after_eos = mask

    def SetNullEmbAfterEOS(self, mask):
        """
        Experimental. Remove later.
        """
        self.null_emb_after_eos = mask

    def SetDeprojector(self, deprojector):
        # deprecated
        self.deprojector = deprojector

    def UseDeprojector(self, to_use: bool = True):
        # deprecated
        self.use_deprojector = to_use

    def DontUseDeprojector(self):
        # deprecated
        self.UseDeprojector(False)

    def max_length(self):
        return self.tokenizer.model_max_length

    def hidden_size(self):
        return self.text_encoder.config.hidden_size

    def num_tokens(self, text: str):
        """
        Returns:
            number of tokens including SOT and EOT
        """
        tokens = self.tokenizer(text)
        return len(tokens.input_ids)

    def tokenize(self, text: str):
        max_length = self.max_length()
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="pt",  # PyTorch
        )

    def decode_tokens(self, tokens):
        return self.tokenizer.batch_decode(tokens)

    def EncodeText(self, text: str, fail_on_truncation: bool):
        """
        Returns:
            Tuple of (
                last_hidden_state, (= embeddings for UNet)
                truncated: bool,  (= whether tokens are truncated)
                pooled_output,  (= EOT embedding)
                num_tokens,  (= number of tokens including SOT and EOT)
            )
        """
        max_length = self.max_length()
        new_text = self.maybe_convert_prompt(text, self.tokenizer)
        if new_text != text:
            Debug(1, f"Prompt is converted by Textual Inversion: {new_text}")
            text = new_text
        text_inputs = self.tokenize(text)
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        truncated = False

        if text_inputs.num_truncated_tokens > 0:
            removed_text = self.decode_tokens(tokenized.overflowing_tokens)
            if fail_on_truncation:
                raise ValueError(
                    f"The following part of your input was truncated because CLIP can only"
                    f" handle sequences up to {max_length} tokens: {removed_text}"
                )
            else:
                Debug(
                    5,
                    f"The following part of your input was truncated because CLIP can only"
                    f" handle sequences up to {max_length} tokens: {removed_text}",
                )
                truncated = True
        embeddings = self.text_encoder(
            text_input_ids.to(self.target.device()),
            attention_mask=None,
        )
        last_hidden_state = embeddings[0]
        pooled_output = embeddings[1]

        # deprecated
        if self.deprojector and self.use_deprojector:
            last_hidden_state = self.Deprojection(pooled_output, from_projected=False)

        num_tokens = 0
        for m in attention_mask[0]:
            if m.item() > 0:
                num_tokens += 1

        __ones = None
        __mask = None

        def GetOnes():
            nonlocal __ones
            if __ones is None:
                __ones = torch.ones([1, 77, 768]).to(self.target.device())
            return __ones

        def GetMask():
            nonlocal __mask
            if __mask is None:
                __mask = GetOnes() - (
                    attention_mask.reshape(1, 77, 1)
                    .repeat(1, 1, 768)
                    .to(self.target.device())
                )
            return __mask

        if self.random_after_eos != 0.0:
            mask = self.random_after_eos * GetMask()
            noise = torch.normal(0, 1, [1, 77, 768]).to(self.target.device())
            last_hidden_state = (GetOnes() - mask) * last_hidden_state + mask * noise

        if self.null_emb_after_eos != 0.0:
            global singleton_null_emb
            if singleton_null_emb is None:
                singleton_null_emb = self.text_encoder(
                    self.tokenize("").input_ids.to(self.target.device()),
                    attention_mask=None,
                )[1]
                singleton_null_emb = singleton_null_emb.reshape(1, 1, 768).repeat(
                    1, 77, 1
                )
            mask = self.null_emb_after_eos * GetMask()
            last_hidden_state = (
                GetOnes() - mask
            ) * last_hidden_state + mask * singleton_null_emb

        if self.mask_after_eos != 0.0:
            coeff = (
                (
                    torch.ones(attention_mask.shape) * (1.0 - self.mask_after_eos)
                    + attention_mask * self.mask_after_eos
                )
                .reshape(1, 77, 1)
                .repeat(1, 1, 768)
                .to(self.target.device())
            )
            last_hidden_state = coeff * last_hidden_state

        return last_hidden_state, truncated, pooled_output, num_tokens

    def GetConstant(self, value: float):
        mat = [
            [value for _ in range(self.hidden_size())] for _ in range(self.max_length())
        ]
        return torch.tensor([mat]).to(**self.target.dict)

    def Deprojection(self, embeds, **kwargs):
        if not self.deprojector:
            raise ValueError(f"Deprojector must be set before calling Deprojection().")
        return self.deprojector.Inference(embeds, **kwargs)


singleton_null_emb = None


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

    def Preprocess(self, image: PIL.Image.Image, target: SharedTarget):
        w, h = image.size
        # Shouldn't this be consistent with vae_scale_factor?
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        return torch.from_numpy(image).to(**target.dict)

    def Encode(
        self,
        image: PIL.Image.Image,
        generator: Generator,
        target: SharedTarget,
        deterministic: bool = False,
    ):
        image = self.Preprocess(image, target)
        image = 2.0 * image - 1.0

        # encode the init image into latents and scale the latents
        latent_dist = self._vae.encode(image).latent_dist
        if deterministic:
            latents = latent_dist.mode()
        else:
            latents = latent_dist.sample(generator=generator)
        latents = (
            0.18215 * latents
        )  # TODO: use vae.config.scaling_factor instead of 0.18215

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
    @classmethod
    def CreateIfNeeded(cls, mask_by):
        if isinstance(mask_by, cls):
            return mask_by
        else:
            return cls(mask_by)

    def __init__(self, mask_by: Union[float, str, PIL.Image.Image], inverse=False):
        self._mask_by = mask_by
        self._inverse = inverse

    def Initialize(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        target: SharedTarget,
    ):
        mask = self._mask_by
        if isinstance(mask, str):
            # TODO: use alpha if it has alpha
            mask = OpenImageWithBackground(mask, background="black")
        if isinstance(mask, PIL.Image.Image):
            if size and (mask.size != size):
                Debug(1, f"Resize mask image from {mask.size} to {size}.")
                mask = mask.resize(size)
            self._mask = image_model.PreprocessMask(mask, target)
        else:
            self._mask = 1.0 - mask  # make the scale same as the mask image.

        if self._inverse:
            self._mask = 1.0 - self._mask

    def Merge(self, black, white, level: float = 1.0):
        inverted_mask = 1.0 - self._mask  # make white = 1.0 & black = 0.0
        inverted_mask *= level
        return self.ApplyInvertedMask(inverted_mask, black=black, white=white)

    # TODO: consider integrating with UnionMask() below.
    def UnionWithLevel(self, other, level: float = 1.0):
        my_inverted_mask = level * (1.0 - self._mask)  # white = 1.0 & black = 0.0
        union_mask = other * (1.0 - my_inverted_mask)  # white = 0.0 & black = 1.0
        return union_mask

    @classmethod
    def ApplyMask(cls, mask, black, white):
        return (black * mask) + (white * (1.0 - mask))

    @classmethod
    def ApplyInvertedMask(cls, inverted_mask, black, white):
        return cls.ApplyMask(inverted_mask, black=white, white=black)


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


def ScaledMask(mask: MaskType, end: float = 1.0, start: float = 0.0):
    def transform(masks):
        mask = masks[0]
        return mask * end + (1.0 - mask) * start

    return ComboOf([mask], transformed_by=transform)


def UnionMask(masks: List[MaskType]):
    def transform(masks):
        mask = 1.0
        for m in masks:
            mask *= 1.0 - m
        return 1.0 - mask

    return ComboOf(masks, transformed_by=transform)


def IntersectMask(masks: List[MaskType]):
    def transform(masks):
        mask = 1.0
        for m in masks:
            mask *= m
        return mask

    return ComboOf(masks, transformed_by=transform)


#
# -- GeneralizedStrength --
#


StrengthType = Optional[Union[float, RangeMap]]


class GeneralizedStrength:
    def __init__(self, strength: StrengthType, is_increasing: bool, debug_label: str):
        self.is_increasing = is_increasing
        if is_increasing:
            start, end = 0.0, 1.0
        else:
            start, end = 1.0, 0.0
        self._strength = RangeMap.CreateAndFinalize(strength, False, start, end)
        self._debug_label = f"GeneralizedStrength[{debug_label}]"
        Debug(3, f"{self._debug_label}: {str(self._strength)}")

    def Merge(self, mask: LatentMask, other, mine, remaining: float):
        level = self.GetCurrentLevel(remaining)
        return mask.Merge(black=other, white=mine, level=level)

    def Add(self, mask: LatentMask, other, mine, remaining: float):
        level = self.GetCurrentLevel(remaining)
        return other + mask.Merge(black=0.0, white=mine, level=level)

    def UnionMask(self, other_mask, my_mask, remaining: float):
        level = self.GetCurrentLevel(remaining)
        return my_mask.UnionWithLevel(other_mask, level=level)

    def GetCurrentLevel(self, remaining: float) -> float:
        return self._strength.Get(remaining, debug_label=self._debug_label)

    def First(self):
        return self._strength.GetFirst()


#
# -- Encoding is a class that defines how to encode a prompt --
#
class StandardEncoding:
    def __init__(self, text: Optional[str] = None):
        self._text = text

    def IsTerminated(self):
        return True

    def EncodeSelf(
        self, text_model: TextModel, default_encoding
    ):  #: StandardEncoding):
        text = self._text
        if text is None:
            Debug(
                0,
                "The default text is not set for the encoding. A null string is used instead.",
            )
            text = ""
        return self.Encode(text_model, text, default_encoding)

    def Encode(
        self, text_model: TextModel, text: str, default_encoding  #: StandardEncoding
    ):
        emb = text_model.EncodeText(text, fail_on_truncation=True)[0]
        return emb


class ConstEncoding(StandardEncoding):
    def __init__(self, value: float = 0.0):
        self.value = value

    def EncodeSelf(self, text_model: TextModel, default_encoding: StandardEncoding):
        emb = text_model.GetConstant(self.value)
        return emb

    def Encode(
        self, text_model: TextModel, text: str, default_encoding: StandardEncoding
    ):
        Debug(0, f"A prompt is ignored for ConstEncoding: {text}")
        return self.EncodeSelf(text_model)


class _FoldEncoding(StandardEncoding):
    "This is currently an experimental feature."

    def __init__(self, text: Optional[str] = None):
        super().__init__(text)
        self.encoding = ShiftEncoding()

    def _Average(self, embs):
        Debug(1, f"Take the average of {len(embs)} embeddings.")
        s = embs[0]
        for e in embs[1:]:
            s += e
        return s / len(embs)

    def Encode(
        self, text_model: TextModel, text: str, default_encoding: StandardEncoding
    ):
        Debug(1, f"Before folding: {text}")
        folded_texts = [s for s in (s.strip() for s in text.split(",")) if s]
        Debug(1, "Folded Texts:", folded_texts)
        if not folded_texts:
            return self.encoding.Encode(text_model, text, default_encoding)
        embs = [
            self.encoding.Encode(text_model, s, default_encoding) for s in folded_texts
        ]
        return self._Average(embs)


class _AverageEncoding(StandardEncoding):
    "This is currently an experimental feature."

    def __init__(self, encodings: List[StandardEncoding]):
        if not encodings:
            raise ValueError(f"Encodings must not be empty.")
        self.encodings = encodings

    def IsTerminated(self):
        return all(e.IsTerminated() for e in self.encodings)

    def EncodeSelf(self, text_model: TextModel, default_encoding: StandardEncoding):
        embs = [e.EncodeSelf(text_model, default_encoding) for e in self.encodings]
        return self._Average(embs)

    def Encode(
        self, text_model: TextModel, text: str, default_encoding: StandardEncoding
    ):
        embs = [e.Encode(text_model, text, default_encoding) for e in self.encodings]
        return self._Average(embs)

    def _Average(self, embs):
        Debug(1, f"Take the average of {len(embs)} embeddings.")
        s = embs[0]
        for e in embs[1:]:
            s += e
        return s / len(embs)


class ScaledEncoding(StandardEncoding):
    def __init__(
        self,
        scale: Union[float, Tuple[float, float]],
        encoding: Optional[StandardEncoding] = None,
    ):
        if isinstance(scale, tuple):
            self._scale = scale
            self._scale_tensor = None  # compute later
        else:
            self._scale = (scale, scale)
            self._scale_tensor = scale
        self._encoding = encoding

    def IsTerminated(self):
        if self._encoding is None:
            return False
        return self._encoding.IsTerminated()

    def GetScale(self, text_model):
        if self._scale_tensor is None:
            scale = [self._scale[0]] + (
                [self._scale[1]] * (text_model.max_length() - 1)
            )
            self._scale_tensor = (
                torch.tensor(scale).to(**text_model.target.dict).unsqueeze(-1)
            )
        return self._scale_tensor

    def EncodeSelf(self, text_model: TextModel, default_encoding: StandardEncoding):
        encoding = self._encoding
        if encoding is None:
            encoding = default_encoding
        emb = encoding.EncodeSelf(text_model, default_encoding)
        return emb * self.GetScale(text_model)

    def Encode(
        self, text_model: TextModel, text: str, default_encoding: StandardEncoding
    ):
        encoding = self._encoding
        if encoding is None:
            encoding = default_encoding
        emb = encoding.Encode(text_model, text, default_encoding)
        return emb * self.GetScale(text_model)


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
            anchored,  #: PromptParser.Chunks,
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
            num_tokens = text_model.num_tokens(text) - 2
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


class Accumulator:
    def __init__(self, target: SharedTarget):
        self.states = None

    def AddOrInit(self, v, x):
        if v is None:
            return x
        else:
            return v + x

    def Add(self, states):
        self.states = self.AddOrInit(self.states, states)

    def Average(self, chunk_len):
        self.states /= chunk_len

    def ToStates(self):
        return self.states


class ShiftEncoding(StandardEncoding):
    def __init__(
        self,
        text: Optional[str] = None,
        normalize_weight: bool = False,
        reverse: bool = True,
        rotate: bool = True,
        convolve: Union[bool, List[float]] = False,
    ):
        super().__init__(text)
        self._normalize_weight = normalize_weight
        self._reverse = reverse
        self._rotate = rotate
        if not isinstance(convolve, list):
            if convolve:
                convolve = [0.7, 0.23, 0.07]
            else:
                convolve = [1.0]
        self._convolve = np.array(convolve)

    def GetWeights(self, text_model: TextModel, chunks):
        weights = [1.0]  # SOT
        for w, n in zip(chunks.weights, chunks.token_nums):
            weights += [w] * n
        if len(weights) >= text_model.max_length() - 1:
            weights = weights[: text_model.max_length() - 1]
        weights += [1.0]  # EOT
        orig_len = len(weights)
        weights += weights[-1:] * (text_model.max_length() - orig_len)
        if self._normalize_weight:
            weight_sum = sum(weights)
            if weight_sum == 0.0:
                raise ValueError(
                    f"unable to normalize weights if the sum is 0.0: {weights}"
                )
            coeff = len(weights) / weight_sum  # inverse of average weight
            weights = [coeff * w for w in weights]
        weights = (
            np.convolve(np.array(weights) - 1.0, self._convolve, mode="full") + 1.0
        )
        Debug(3, "weights:", weights[:orig_len])
        target = text_model.target
        return torch.tensor(weights).to(**target.dict).unsqueeze(-1)

    def Encode(
        self, text_model: TextModel, text: str, default_encoding: StandardEncoding
    ):
        target = text_model.target
        parser = PromptParser()
        prompt = parser.Parse(text_model, text)
        accum = Accumulator(target)
        truncated = False

        def run():
            nonlocal truncated
            shift_range = prompt.ShiftRange()
            for i in range(shift_range):
                chunks = prompt.Shift(i, self._rotate)
                shifted_text = " ".join(chunks.texts)
                Debug(3, f"shifted_text: {shifted_text}")
                weights = self.GetWeights(text_model, chunks)
                encode_result = text_model.EncodeText(
                    shifted_text, fail_on_truncation=False
                )
                truncated = truncated or encode_result[1]
                embeddings = encode_result[0][0]
                full = embeddings * weights
                accum.Add(full)
            return shift_range

        n = run()
        if self._reverse:
            prompt.Reverse()
            n += run()
        accum.Average(n)
        if truncated:
            Debug(1, f"ShiftEncoding: text is truncated: {text}")

        new_states = accum.ToStates()
        return torch.cat([torch.unsqueeze(new_states, 0)], dim=0).to(target.device())


class ImageEncoding(StandardEncoding):
    def __init__(
        self,
        model: CLIPVisionModelWithProjection,
        processor: AutoProcessor,
        base_img_emb: torch.Tensor,
        image: Optional[Union[str, PIL.Image.Image]] = None,
    ):
        super().__init__(image)
        self.model = model
        self.processor = processor
        self.base_img_emb = base_img_emb.unsqueeze(0)

    def Encode(
        self,
        text_model: TextModel,
        image: Union[str, PIL.Image.Image],
        default_encoding: StandardEncoding,
    ):
        Debug(2, image)
        if isinstance(image, str):
            image = OpenImageWithBackground(image)
        input = self.processor(images=image, return_tensor="pt")
        # Debug(2, input.keys(), input["pixel_values"])
        output = self.model(
            pixel_values=torch.tensor(input["pixel_values"]).to(
                text_model.target.device()
            )
        )
        # Debug(2, output.image_embeds.shape)
        base_emb = text_model.EncodeText("", fail_on_truncation=True)[2]
        embeds = output.image_embeds - self.base_img_emb + base_emb
        return text_model.Deprojection(embeds, from_projected=True)


class PooledEncoding(StandardEncoding):
    def __init__(
        self,
        normalize: bool = False,
        diff_only: bool = False,
        text: Optional[str] = None,
    ):
        super().__init__(text)
        self.normalize = normalize
        self.diff_only = diff_only

    def Encode(
        self, text_model: TextModel, text: str, default_encoding: StandardEncoding
    ):
        chunks = [s for s in (s.strip() for s in text.split(",")) if s]
        embeds = []
        for chunk in chunks:
            e = text_model.EncodeText(chunk, fail_on_truncation=True)[2][0]
            embeds.append(e)
        if self.diff_only:
            base_emb = text_model.EncodeText("", fail_on_truncation=True)[2][0]
            embeds = [e - base_emb for e in embeds]
            merged_embeds = sum(embeds) + base_emb
        else:
            merged_embeds = sum(embeds)
        if self.normalize:
            coeff = (
                sum(torch.norm(e) for e in embeds)
                / len(embeds)
                / torch.norm(merged_embeds)
            )
            merged_embeds *= coeff
        return text_model.Deprojection(merged_embeds.unsqueeze(0), from_projected=False)


#
# -- Type: PromptType --
#
# A prompt can be
#  * str  --  a prompt text
#  * Encoding  -- an encoding without a promppt text
#  * (str, Encoding)  --  a prompt text & its encoding method
#  * following the RangeMap format.
PromptType = Optional[
    Union[
        str,
        StandardEncoding,
        Tuple[str, StandardEncoding],
        RangeMap,
    ]
]

ZeroEmbedding = ConstEncoding(0.0)
# -0.11 is the approximate average of embedding vector elements.
UnitEmbedding = ConstEncoding(-0.11)


#
# -- Text embedding utility to combine multiple embeddings --
#
class TextEmbeddings:
    def __init__(self, text_embeddings: RangeMap):
        self._text_embeddings = text_embeddings

    def GetEmbedding(self, remaining: float):
        emb = self._text_embeddings.Get(remaining, debug_label="TextEmbeddings")
        return emb

    @classmethod
    def Create(
        cls,
        input: PromptType,
        text_model: TextModel,
        disable_deprojector: bool,
        default_encoding: StandardEncoding,
    ):
        if not default_encoding.IsTerminated():
            raise ValueError("Default encoding must be terminated.")

        def finalizer(arg):
            if isinstance(arg, str):
                return default_encoding.Encode(text_model, arg, default_encoding)
            elif isinstance(arg, StandardEncoding):
                return arg.EncodeSelf(text_model, default_encoding)
            elif isinstance(arg, tuple):
                return arg[1].Encode(text_model, arg[0], default_encoding)
            else:
                return arg

        use_deprojector = text_model.use_deprojector
        if disable_deprojector:
            text_model.DontUseDeprojector()
        default_embedding = default_encoding.Encode(text_model, "", default_encoding)
        if input == "":
            input = None
        embeddings = RangeMap.CreateAndFinalize(
            input, True, default_embedding, default_embedding, finalizer
        )
        text_model.UseDeprojector(use_deprojector)
        return cls(embeddings)


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

    def Wrap(self, scheduler_type: str, scheduler):
        self.scheduler = scheduler
        if scheduler_type == "dpm":
            self.have_max_timesteps = False
        else:
            Debug(0, f"Unknown scheduler type: {scheduler_type}")
            self.have_max_timesteps = True
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
        residual_cat = torch.cat(residuals)
        original_latents_cat = torch.cat(original_latents)
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
    def __init__(self, mask_by: MaskType, strength: StrengthType):
        self.mask = LatentMask.CreateIfNeeded(mask_by)
        self.strength = GeneralizedStrength(
            strength, is_increasing=False, debug_label="Initializer"
        )
        self.is_whole_image = mask_by == 1.0

    def IsWholeImage(self):
        return self.is_whole_image

    def GetStrength(self):
        return self.strength

    def GetSize(self):
        raise NotImplementedError("Subclass should implement this.")

    def InitializeMask(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        target: SharedTarget,
    ):
        self.mask.Initialize(size, image_model, target)

    def OverwriteLatents(
        self,
        scheduler: Scheduler,
        latents,
        timesteps,
        step_index: int,
    ):
        # do nothing
        return latents, False

    def Merge(self, other, mine, remaining, force_overwrite: bool):
        if force_overwrite:
            return self.mask.Merge(black=other, white=mine, level=1.0)
        else:
            return self.strength.Merge(
                mask=self.mask, other=other, mine=mine, remaining=remaining
            )

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
        mask_by: MaskType = 1.0,
        strength: StrengthType = None,
    ):
        """
        Args:
            size (`(int, int)`, *optional*, defaults to (512, 512))
                The (width, height) pair in pixels of the generated image.
        """
        super().__init__(mask_by, strength)
        self._symmetric = symmetric

    def InitializeLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        target: SharedTarget,
        vae_encoder_adjust: float,
        scheduler: Scheduler,
        timesteps,
    ):
        self.InitializeMask(size, image_model, target)
        latents = self._GetLatents(
            size, image_model, scheduler.generator(), target, vae_encoder_adjust
        )
        self._size = size
        return latents * scheduler.Get().init_noise_sigma

    def GetSize(self):
        return self._size

    def _GetLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        generator: Generator,
        target: SharedTarget,
        vae_encoder_adjust: float,
    ):
        if not size:
            raise ValueError("`size` is mandatory to initialize `Randomly`.")

        latents_shape = self._GetLatentsShape(image_model, size)

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

    def _GetLatentsShape(self, image_model: ImageModel, size: Tuple[int, int]):
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
        mask_by: MaskType = 1.0,
        strength: StrengthType = None,
    ):
        """
        Args:
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied random `generator`.
        """
        super().__init__(mask_by, strength)
        self._latents = latents

    def InitializeLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        target: SharedTarget,
        vae_encoder_adjust: float,
        scheduler: Scheduler,
        timesteps,
    ):
        scale_factor = image_model.vae_scale_factor()
        _, _, h, w = latents.shape
        self._size = (w * scale_factor, h * scale_factor)
        if size and (size != self._size):
            raise ValueError(
                f"Latent size should match the given image size: {self._size}"
            )

        self.InitializeMask(size, image_model, target)
        latents = self._latents.to(**target.dict)
        return latents * scheduler.Get().init_noise_sigma

    def GetSize(self):
        return self._size


class ByImage(Initializer):
    def __init__(
        self,
        image: Union[str, PIL.Image.Image],
        mask_by: MaskType = 1.0,
        strength: StrengthType = 0.8,
    ):
        """
        Args:
            image `PIL.Image.Image`:
                `Image`, or tensor representing an image batch, that will be used as the starting point for the process.
            strength (`StrengthType`, *optional*, defaults to 0.8):
                TODO: update the description below.
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
        """
        if isinstance(image, str):
            image, mask = OpenImage(image)
            if not image:
                raise ValueError(f"image must not be empty.")
            if mask:
                mask_by = IntersectMask([mask_by, mask])
        self._image = image
        super().__init__(mask_by, strength)

    def InitializeLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        target: SharedTarget,
        vae_encoder_adjust: float,
        scheduler: Scheduler,
        timesteps,
    ):
        self.InitializeMask(size, image_model, target)

        initial_timestep = timesteps[:1]
        generator = scheduler.generator()

        image = self._image
        if size and (image.size != size):
            Debug(1, f"Resize image from {image.size} to {size}.")
            image = image.resize(size, resample=PIL.Image.LANCZOS)
        self._size = image.size

        latents = image_model.Encode(image, generator, target) * vae_encoder_adjust
        self._initial_latents = latents
        self._noise = self.RandForShape(latents.shape, generator, target)

        return scheduler.Get().add_noise(
            self._initial_latents, self._noise, initial_timestep
        )

    def GetSize(self):
        return self._size

    def OverwriteLatents(
        self,
        scheduler: Scheduler,
        latents,
        timesteps,
        step_index: int,
    ):
        # Return the initial latents without noise if this is the last step.
        if step_index + 1 >= len(timesteps):
            Debug(4, "Last step with keeping initial image.")
            return self._initial_latents, True
        # Otherwise add noise to the initial latents.
        Debug(4, f"Add noise for step {step_index + 1}.")
        next_timestep = timesteps[step_index + 1 : step_index + 2]
        return (
            scheduler.Get().add_noise(
                self._initial_latents, self._noise, next_timestep
            ),
            True,
        )


class InitializerList:
    def __init__(self, initializers: Optional[Union[Initializer, List[Initializer]]]):
        if not initializers:
            initializers = []
        elif not isinstance(initializers, list):
            initializers = [initializers]

        max_strength = 0.0
        its_level = 0.0
        reverse_index = -1
        for i, init in enumerate(reversed(initializers)):
            value, until = init.GetStrength().First()
            if (
                init.IsWholeImage()
                and (until > max_strength)
                or ((its_level < 1.0) and (until == max_strength))
            ):
                max_strength = until
                its_level = value
                reverse_index = i
        if (reverse_index < 0) or (its_level < 1.0):
            # no initializers, or none of them cover the whole image.
            # or it needs another preceding initializer.
            Debug(1, f"Insert the random initializer to the head.")
            initializers = [Randomly()] + initializers
            max_strength = 1.0
        else:
            Debug(1, f"Take only last {reverse_index + 1} initializers.")
            initializers = initializers[-(reverse_index + 1) :]
        self.initializers = initializers
        self.max_strength = max_strength

    def GetMaxStrength(self):
        return self.max_strength

    def InitializeLatents(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        target: SharedTarget,
        vae_encoder_adjust: float,
        scheduler: Scheduler,
        timesteps,
    ):
        latents_list = [
            (
                init.InitializeLatents(
                    size,
                    image_model,
                    target,
                    vae_encoder_adjust,
                    scheduler,
                    timesteps,
                ),
                init,
            )
            for init in self.initializers
        ]
        init_latents = latents_list[0][0]
        init_size = latents_list[0][1].GetSize()
        for latents, init in latents_list[1:]:
            if init_size != init.GetSize():
                raise ValueError(
                    f"Image size of initializers are not the same: {init_size} vs {init.GetSize()}"
                )
            init_latents = init.Merge(
                other=init_latents, mine=latents, remaining=1.0, force_overwrite=True
            )
        self._size = init_size
        return init_latents

    def GetSize(self):
        return self._size

    def OverwriteLatents(
        self,
        scheduler: Scheduler,
        latents,
        timesteps,
        step_index: int,
        remaining: float,
        force_overwrite: bool,
    ):
        for init in self.initializers:
            new_latents, changed = init.OverwriteLatents(
                scheduler,
                latents,
                timesteps,
                step_index,
            )
            if changed:
                latents = init.Merge(
                    other=latents,
                    mine=new_latents,
                    remaining=remaining,
                    force_overwrite=force_overwrite,
                )
        return latents


#
# -- Prompts --
#
class Prompts:
    def __init__(
        self,
        text_model: TextModel,
        default_encoding: StandardEncoding,
        negative_prompt_scale: Union[float, Tuple[float, float]],
    ):
        self._text_model = text_model
        self._default_encoding = default_encoding
        if (negative_prompt_scale == 1.0) or (negative_prompt_scale == (1.0, 1.0)):
            self._default_negative_encoding = default_encoding
        else:
            self._default_negative_encoding = ScaledEncoding(
                scale=negative_prompt_scale, encoding=default_encoding
            )
        self._prompts = []

    def __len__(self):
        return len(self._prompts)

    def Add(self, prompt: PromptType, is_negative: bool) -> int:
        Debug(3, "Add prompt:", prompt)
        index = len(self._prompts)
        default_encoding = (
            self._default_negative_encoding if is_negative else self._default_encoding
        )
        self._prompts.append(
            TextEmbeddings.Create(
                prompt, self._text_model, is_negative, default_encoding
            )
        )
        return index

    def PredictResiduals(
        self, scheduler, unet, controlnet, latents, timestep, remaining: float
    ):
        model_input = torch.cat(latents)
        model_input = scheduler.Get().scale_model_input(model_input, timestep)
        text_embeddings = torch.cat(
            [te.GetEmbedding(remaining) for te in self._prompts]
        )
        if controlnet:
            down_block_residuals, mid_block_residuals = controlnet(
                model_input, timestep, text_embeddings
            )
        else:
            down_block_residuals, mid_block_residuals = None, None
        residuals = unet(
            model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_residuals,
            mid_block_additional_residual=mid_block_residuals,
        ).sample
        return residuals.chunk(len(self._prompts))


#
# -- Layer --
#
class DecomposedResidual:
    def __init__(self, cond, uncond, scale):
        self.cond = cond
        self.uncond = uncond
        self.scale = scale

    @classmethod
    def CreateForLayer(cls, cond, uncond, scale):
        # CreateForLayer() is for normal Layer.
        return DecomposedResidual(
            cond * scale,
            uncond * scale,
            scale=scale,
        )

    @classmethod
    def CreateForSPLayer(cls, cond):
        # CreateForSPLayer() is for SPLayer.
        return DecomposedResidual(
            cond,
            None,
            scale=None,
        )

    @classmethod
    def CreateForBGLayer(cls, shape, target):
        # CreateForBGLayer() is for BackgroundLayer.
        return DecomposedResidual(
            torch.zeros(shape).to(**target.dict),
            torch.zeros(shape).to(**target.dict),
            scale=0.0,
        )

    def IsForBG(self):
        return (not isinstance(self.scale, torch.Tensor)) and (self.scale == 0.0)

    def AddOrMerge(
        self, mine, strength: GeneralizedStrength, mask, remaining: float, to_add: bool
    ):
        if mine.uncond is None:
            raise ValueError(
                f"AddOrMerge() works only for DecomposedResidual with uncond."
            )

        def _do(a, b):
            if to_add:
                return strength.Add(mask=mask, other=a, mine=b, remaining=remaining)
            else:
                return strength.Merge(mask=mask, other=a, mine=b, remaining=remaining)

        return DecomposedResidual(
            cond=_do(self.cond, mine.cond),
            uncond=_do(self.uncond, mine.uncond),
            scale=_do(self.scale, mine.scale),
        )

    def AddOrMergeForCond(
        self,
        mine,
        strength: GeneralizedStrength,
        mask,
        remaining: float,
        to_add: bool,
    ):
        if mine.uncond is not None:
            raise ValueError(
                f"AddOrMergeForCond() works only for DecomposedResidual without uncond."
            )

        def _do(other, my_cond, same: bool):
            if to_add:
                # TODO: multiply rather than add if same == True
                return strength.Add(
                    mask=mask,
                    other=other,
                    mine=my_cond,
                    remaining=remaining,
                )
            elif same:
                # Merge but two args are same
                return other
            else:
                return strength.Merge(
                    mask=mask,
                    other=other,
                    mine=my_cond,
                    remaining=remaining,
                )

        if self.IsForBG():
            return DecomposedResidual(
                cond=_do(self.cond, mine.cond, False),
                uncond=_do(self.uncond, self.uncond, True),
                scale=_do(self.scale, 1.0, False),
            )
        else:
            return DecomposedResidual(
                cond=_do(self.cond, mine.cond * self.scale, False),
                uncond=_do(self.uncond, self.uncond, True),
                scale=_do(self.scale, self.scale, True),
            )

    def AddOrMergeForUncond(
        self,
        mine,
        strength: GeneralizedStrength,
        mask,
        remaining: float,
        to_add: bool,
    ):
        if mine.uncond is not None:
            raise ValueError(
                f"AddOrMergeForUncond() works only for DecomposedResidual without uncond."
            )
        if self.IsForBG():
            raise ValueError(
                f"AddOrMergeForUncond() must not be called for the BackgroundLayer."
            )

        if to_add:
            uncond = strength.Add(
                mask=mask,
                other=self.uncond,
                mine=mine.cond * self.scale,
                remaining=remaining,
            )
            new_scale = strength.Add(
                mask=mask,
                other=self.scale,
                mine=self.scale,
                remaining=remaining,
            )
            # scale back to the original scale
            uncond = uncond * scale / new_scale
        else:
            uncond = strength.Merge(
                mask=mask,
                other=self.uncond,
                mine=mine.cond * self.scale,
                remaining=remaining,
            )

        return DecomposedResidual(cond=self.cond, uncond=uncond, scale=self.scale)

    def Compose(self):
        if not isinstance(self.scale, torch.Tensor):
            if self.scale <= 1.0:
                return self.cond
            else:
                uncond = self.uncond * (self.scale - 1.0) / self.scale
                res = self.cond - uncond
                return res

        le_one = torch.le(self.scale, 1.0)
        coeff = torch.where(le_one, 0.0, (self.scale - 1.0) / self.scale)
        uncond = self.uncond * coeff
        res = self.cond - uncond
        return res


class BackgroundLayer:
    def __init__(
        self,
        initializers: Optional[Union[Initializer, List[Initializer]]],
        vae_encoder_adjust: Optional[float] = None,
    ):
        self._initializers = InitializerList(initializers)
        if not vae_encoder_adjust:
            self._vae_encoder_adjust = 1.0
        else:
            self._vae_encoder_adjust = vae_encoder_adjust

    def IsDistinct(self):
        return False

    def GetSize(self):
        return self._initializers.GetSize()

    def Initialize(
        self,
        num_steps: int,
        size: Optional[Tuple[int, int]],
        scheduler: Scheduler,
        image_model: ImageModel,
        prompts: Prompts,
        target: SharedTarget,
    ):
        Debug(3, "Initialize BackgroundLayer.")
        self._indices = []
        max_strength = self._initializers.GetMaxStrength()
        # compute num_steps_to_request to match the effective number of iterations with num_steps.
        num_steps_to_request = int(-(-int(num_steps) // max_strength))  # round up
        scheduler.SetTimesteps(num_steps_to_request)
        timesteps, total_num_steps = self.GetTimesteps(scheduler, max_strength, target)
        Debug(
            3, f"total_num_steps: {total_num_steps}, actual num_steps: {len(timesteps)}"
        )
        self._total_num_steps = total_num_steps
        latents = self._initializers.InitializeLatents(
            size,
            image_model,
            target,
            self._vae_encoder_adjust,
            scheduler,
            timesteps,
        )
        return latents, timesteps

    # TODO: move to scheduler
    @classmethod
    def GetTimesteps(cls, scheduler: Scheduler, strength: float, target: SharedTarget):
        # dtype should be `int`
        timesteps = scheduler.Get().timesteps.to(device=target.device())
        total_num_steps = len(timesteps)
        if strength > 1.0:
            raise ValueError(f"invalid strength: {strength}")
        if strength == 1.0:
            return timesteps, total_num_steps
        actual_num_steps = int(total_num_steps * strength)
        return timesteps[-actual_num_steps:], total_num_steps

    # TODO: move to scheduler
    def GetRemaining(self, timesteps, step_index):
        return float(len(timesteps) - step_index - 1) / self._total_num_steps

    def OverwriteLatents(
        self,
        scheduler: Scheduler,
        latents,
        timesteps,
        step_index: int,
        force_overwrite: bool = False,
    ):
        remaining = self.GetRemaining(timesteps, step_index)
        return self._initializers.OverwriteLatents(
            scheduler, latents, timesteps, step_index, remaining, force_overwrite
        )

    def GetResidual(self, residuals, remaining, decomposed, target: SharedTarget):
        if decomposed:
            return DecomposedResidual.CreateForBGLayer(residuals[0].shape, target)
        else:
            return torch.zeros_like(residuals[0]).to(**target.dict)


class BaseLayer:
    def __init__(
        self,
        prompt: PromptType,
        negative_prompt: PromptType = None,
        to_add: bool = False,  # default: merge mode
    ):
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self.to_add = to_add

    def _Initialize(
        self, prompts: Prompts, decomposed: bool, do_cfg: bool, is_sp_negative: bool
    ):
        self.decomposed = decomposed
        self._do_cfg = do_cfg
        self._prompts = prompts
        self._indices = [prompts.Add(self._prompt, is_negative=is_sp_negative)]
        if self._do_cfg:
            self._indices.append(prompts.Add(self._negative_prompt, is_negative=True))

    def GetResidualAt(self, residuals, index: int):
        return residuals[self._indices[index]]


class Layer(BaseLayer):
    def __init__(
        self,
        prompt: PromptType = None,
        negative_prompt: PromptType = None,
        cfg_scale: Union[
            float, RangeMap
        ] = 4.0,  # follows https://huggingface.co/stabilityai/stable-diffusion-2
        mask_by: MaskType = 1.0,
        strength: StrengthType = None,
        is_distinct: bool = False,  # False: common layer, True: distinct layer
        disabled: bool = False,
        to_add: bool = False,  # default: merge mode
    ):
        if cfg_scale is None:
            raise ValueError(f"cfg_scale must not be None.")
        default_cfg_scale = 4.0
        cfg_scale = RangeMap.CreateAndFinalize(
            cfg_scale, True, default_cfg_scale, default_cfg_scale
        )
        self._cfg_scale = cfg_scale
        super().__init__(
            prompt,
            negative_prompt,
            to_add=to_add,
        )
        self._mask = LatentMask.CreateIfNeeded(mask_by)
        self._strength = GeneralizedStrength(
            strength, is_increasing=True, debug_label="Layer"
        )
        self._is_distinct = is_distinct
        self.disabled = disabled

    def IsDistinct(self):
        return self._is_distinct

    def Initialize(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        prompts: Prompts,
        target: SharedTarget,
        decomposed: bool,
    ):
        Debug(3, "Initialize Layer.")
        if decomposed:
            # If decomposed, always do CFG.
            do_cfg = True
        else:
            if self.to_add:
                # For an adding Layer, always do CFG.
                do_cfg = True
            else:
                cfg_is_const, const_cfg_scale = self._cfg_scale.IsConstant()
                cfg_scale_is_always_one = cfg_is_const and (const_cfg_scale == 1.0)
                do_cfg = not cfg_scale_is_always_one
                if not do_cfg:
                    Debug(
                        3,
                        "CFG is disabled for a merging Layer whose cfg_scale is always one.",
                    )
        super()._Initialize(
            prompts, decomposed=decomposed, do_cfg=do_cfg, is_sp_negative=False
        )
        self._mask.Initialize(size, image_model, target)

    def GetCfgScale(self, remaining):
        cfg_scale = self._cfg_scale.Get(remaining, debug_label="CFG Scale")

        if self.to_add:
            if cfg_scale < 0.0:
                raise ValueError(
                    f"cfg_scale must be 0.0 or bigger for an adding Layer: {cfg_scale}"
                )
            elif cfg_scale == 0.0:
                Debug(4, f"Effectively ignore an adding Layer if cfg_scale = 0.")
        else:
            if cfg_scale < 1.0:
                raise ValueError(
                    f"cfg_scale must be 1.0 or bigger for a merging layer: {cfg_scale}"
                )
            elif cfg_scale == 1.0:
                Debug(4, f"Effectively ignore a merging layer if cfg_scale = 1.")

        return cfg_scale

    def GetResidual(self, residuals, remaining, decomposed, target: SharedTarget):
        if decomposed:
            residual_cond = self.GetResidualAt(residuals, 0)
            residual_uncond = self.GetResidualAt(residuals, 1)
            return DecomposedResidual.CreateForLayer(
                cond=residual_cond,
                uncond=residual_uncond,
                scale=self.GetCfgScale(remaining),
            )
        else:
            residual_cond = self.GetResidualAt(residuals, 0)
            if not self._do_cfg:
                return residual_cond

            residual_uncond = self.GetResidualAt(residuals, 1)
            residual = self.GetCfgScale(remaining) * (residual_cond - residual_uncond)
            if not self.to_add:
                residual = residual_uncond + residual
            return residual

    def Merge(self, other, mine, remaining: float, decomposed: bool = False):
        if decomposed:
            return other.AddOrMerge(
                mine, self._strength, self._mask, remaining, self.to_add
            )

        if self.to_add:
            return self._strength.Add(
                mask=self._mask, other=other, mine=mine, remaining=remaining
            )
        else:
            return self._strength.Merge(
                mask=self._mask, other=other, mine=mine, remaining=remaining
            )

    def UnionMask(self, other_mask, remaining: float):
        return self._strength.UnionMask(
            other_mask=other_mask, my_mask=self._mask, remaining=remaining
        )


class SPLayer(BaseLayer):  # Single-Prompt Layer
    def __init__(
        self,
        prompt: PromptType = None,
        negative_prompt: PromptType = None,
        mask_by: MaskType = 1.0,
        strength: StrengthType = None,
        to_add: bool = False,  # default: merge mode
        disabled: bool = False,
    ):
        if (prompt is None) and (negative_prompt is None):
            raise ValueError(f"Either prompt or negative_prompt must be set.")
        if (prompt is not None) and (negative_prompt is not None):
            raise ValueError(
                f"Both prompt and negative_prompt must not be set simultaneously."
            )
        if prompt is not None:
            super().__init__(prompt, to_add=to_add)
            self._for_positive = True
        else:
            super().__init__(negative_prompt, to_add=to_add)
            self._for_positive = False

        self._mask = LatentMask.CreateIfNeeded(mask_by)
        self._strength = GeneralizedStrength(
            strength, is_increasing=True, debug_label="SPLayer"
        )
        self.disabled = disabled

    def IsDistinct(self):
        return False

    def Initialize(
        self,
        size: Optional[Tuple[int, int]],
        image_model: ImageModel,
        prompts: Prompts,
        target: SharedTarget,
        decomposed: bool,
    ):
        Debug(3, "Initialize SPLayer.")
        super()._Initialize(
            prompts,
            decomposed=decomposed,
            do_cfg=False,
            is_sp_negative=(not self._for_positive),
        )
        self._mask.Initialize(size, image_model, target)

    def GetResidual(self, residuals, remaining, decomposed, target: SharedTarget):
        residual_cond = self.GetResidualAt(residuals, 0)
        return DecomposedResidual.CreateForSPLayer(residual_cond)

    def Merge(self, other, mine, remaining: float, decomposed: bool = False):
        if not decomposed:
            raise NotImplementedError(f"SPLayer is only for DecomposedResidual.")
        if self._for_positive:
            return other.AddOrMergeForCond(
                mine, self._strength, self._mask, remaining, self.to_add
            )
        else:
            return other.AddOrMergeForUncond(
                mine, self._strength, self._mask, remaining, self.to_add
            )

    def UnionMask(self, other_mask, remaining: float):
        return self._strength.UnionMask(
            other_mask=other_mask, my_mask=self._mask, remaining=remaining
        )


#
# -- Sideloader --
#
class CkptVAESideloader:
    def __init__(self, vae_path, device_type):
        self._enabled = bool(vae_path)
        if not self._enabled:
            return
        vae_pt_suffix = ".pt"
        vae_ckpt_suffix = ".ckpt"
        vae_safetensors_suffix = ".safetensors"

        is_safetensors = (
            vae_path[-len(vae_safetensors_suffix) :] == vae_safetensors_suffix
        )
        if (
            is_safetensors
            or (vae_path[-len(vae_pt_suffix) :] == vae_pt_suffix)
            or (vae_path[-len(vae_ckpt_suffix) :] == vae_ckpt_suffix)
        ):
            # VAE pt file
            if is_safetensors:
                pt_ckpt = {}
                with safe_open(vae_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        pt_ckpt[key] = f.get_tensor(key)
            else:
                pt_ckpt = torch.load(vae_path, map_location=device_type)
            if "state_dict" in pt_ckpt.keys():
                pt_ckpt = pt_ckpt.get("state_dict")

            sd_ckpt = {}
            for key in pt_ckpt.keys():
                sd_ckpt["first_stage_model." + key] = pt_ckpt.get(key)
            self._ckpt = sd_ckpt
            self._should_convert = True
        else:
            # diffusers VAE dir
            self._ckpt = torch.load(
                vae_path + "/diffusion_pytorch_model.bin", map_location=device_type
            )
            self._should_convert = False

    def Initialize(self):
        self._original_function = convert_from_ckpt.convert_ldm_vae_checkpoint
        convert_from_ckpt.convert_ldm_vae_checkpoint = (
            lambda orig_ckpt, config: self.convert_ldm_vae_checkpoint(config, orig_ckpt)
        )
        return self

    def convert_ldm_vae_checkpoint(self, config, orig_ckpt):
        Debug(1, "VAE sideloader is invoked.")
        if not self._enabled:
            # Call the original function with the original checkpoint data.
            return self._original_function(orig_ckpt, config)
        if not self._should_convert:
            return self._ckpt
        return self._original_function(self._ckpt, config)

    def Finalize(self):
        convert_from_ckpt.convert_ldm_vae_checkpoint = self._original_function


class TextualInversion:
    def __init__(self):
        self.tokens = []
        self.paths = []
        self.kwargs_list = []

    def Add(self, token: str, path: str, **kwargs):
        if not token:
            raise ValueError(f"`token` must not be empty.")
        if not path:
            raise ValueError(f"`path` must not be empty.")
        self.tokens.append(token)
        self.paths.append(path)
        self.kwargs_list.append(kwargs)
        return self

    def Keywords(self):
        return self.tokens

    def Apply(self, mixin):
        for token, path, kwargs in zip(self.tokens, self.paths, self.kwargs_list):
            mixin.load_textual_inversion(path, token, **kwargs)


class LoraSideloader:
    def __init__(
        self,
        path: str,
        alpha: float = 1.0,
        text_encoder_alpha: Optional[float] = None,
    ):
        if not path:
            raise ValueError(f"`path` must not be empty.")
        self.path = path
        self.unet_alpha = alpha
        if text_encoder_alpha is None:
            self.te_alpha = alpha
        else:
            self.te_alpha = text_encoder_alpha

    def Load(self, device_type):
        if not (self.te_alpha or self.unet_alpha):
            return

        safetensors_suffix = ".safetensors"
        is_safetensors = self.path[-len(safetensors_suffix) :] == safetensors_suffix
        if is_safetensors:
            ckpt = {}
            with safe_open(self.path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    ckpt[key] = f.get_tensor(key)
        else:
            ckpt = torch.load(self.path, map_location=device_type)

        unet = {}
        te = {}
        for key, value in ckpt.items():
            Debug(5, f"lora key = {key}")
            if key.startswith("lora_unet_"):
                self.AddToModel(key[len("lora_unet_") :], value, unet)
            elif key.startswith("lora_te_"):
                self.AddToModel(key[len("lora_te_") :], value, te)
            else:
                Debug(0, f"unknown lora key: {key}", value)
        self.unet = unet
        self.te = te

    def AddToModel(self, key, value, model):
        processor_key, subkey = key.split(".", 1)
        processor_key = self.ConvertProcessorKey(processor_key)
        Debug(5, f"processor_key => sub_key: {processor_key} => {subkey}")
        model.setdefault(processor_key, {})[subkey] = value

    def ConvertProcessorKey(self, key):
        # Input Examples
        # UNet:
        #  * down_blocks_0_attentions_0_proj_in
        #  * down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k
        #  * down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj
        # TextEncoder:
        #  * text_model_encoder_layers_0_mlp_fc1
        #  * text_model_encoder_layers_0_self_attn_k_proj
        phrases = [
            "down_blocks",
            "mid_block",
            "up_blocks",
            "transformer_blocks",
            "proj_in",
            "proj_out",
            "to_k",
            "to_q",
            "to_v",
            "to_out",
            "text_model",
            "self_attn",
            "k_proj",
            "q_proj",
            "v_proj",
            "out_proj",
        ]

        key = key.replace("_", ".")
        for ph in phrases:
            key = key.replace(ph.replace("_", "."), ph)

        return key

    def GetAdditionalWeight(self, weight_dict, alpha):
        # Ignore weight_dict["alpha"] as it is only for training and not for inference.
        down = weight_dict["lora_down.weight"]
        up = weight_dict["lora_up.weight"]
        need_squeeze = len(up.shape) == 4
        if need_squeeze:
            up = up.squeeze(3).squeeze(2)
            down = down.squeeze(3).squeeze(2)
        weights = alpha * torch.mm(up.to(torch.float32), down.to(torch.float32))
        if need_squeeze:
            weights = weights.unsqueeze(2).unsqueeze(3)
        return weights

    def Apply(self, text_encoder, unet):
        Debug(1, f"LoraSideloader is invoked.")
        if self.te_alpha:
            Debug(3, f"Update Text Encoder model with LoRA.")
            self.ApplyModel(self.te, self.te_alpha, text_encoder)
        if self.unet_alpha:
            Debug(3, f"Update UNet model with LoRA.")
            self.ApplyModel(self.unet, self.unet_alpha, unet)

    def ApplyModel(self, model_dict, alpha, root):
        for key, weights in model_dict.items():
            Debug(5, f"apply to key = {key}")
            path = key.split(".")
            layer = root
            for name in path:
                try:
                    layer = layer.__getattr__(name)
                    Debug(5, f"go down to layer {name}")
                except Exception as e:
                    Debug(0, f"ERROR: {type(e)} occurred: {e}")
                    Debug(0, f"key = {key}")
                    Debug(0, f"name = {name}")
                    continue
            if "weight" in dir(layer):
                layer.weight.data += self.GetAdditionalWeight(weights, alpha)
            else:
                Debug(0, f"ERROR: Skipping layer witout weight: {key}")


class LoRA:
    def __init__(self):
        self._lora = []

    def Add(
        self,
        path: str,
        alpha: float = 1.0,
        text_encoder_alpha: Optional[float] = None,
    ):
        self._lora.append(LoraSideloader(path, alpha, text_encoder_alpha))
        return self

    def Apply(self, device_type, text_encoder, unet):
        for loader in self._lora:
            loader.Load(device_type)
            loader.Apply(text_encoder, unet)


#
# -- ControlNet --
#
class ControlNet:
    def __init__(self, post_scale: float = 1.0, pre_scale: float = 1.0):
        self._master_post_scale = post_scale
        self._master_pre_scale = pre_scale
        self._models = []
        self._images = []
        self._post_scales = []
        self._pre_scales = []
        self._detectors = []
        self._detector_args = []

    def Add(
        self,
        model: ControlNetModel,
        image: Union[str, PIL.Image.Image],
        post_scale: float = 1.0,
        pre_scale: float = 1.0,
        detector=None,
    ):
        if not model:
            raise ValueError(f"`model` must be set.")
        if not image:
            raise ValueError(f"`image` must be set.")
        self._models.append(model)
        self._images.append(image)
        self._post_scales.append(post_scale * self._master_post_scale)
        self._pre_scales.append(pre_scale * self._master_pre_scale)

        if isinstance(detector, tuple) or isinstance(detector, list):
            self._detectors.append(detector[0])
            self._detector_args.append(detector[1:])
        else:
            self._detectors.append(detector)
            self._detector_args.append([])
        return self

    def Initialize(
        self,
        size: Tuple[int, int],
        image_model: ImageModel,
        prompts: Prompts,
        target: SharedTarget,
    ):
        Debug(1, "ControlNet is initialized.")

        def GetImageInput(image, pre_scale, detector, arg):
            if isinstance(image, str):
                image = OpenImageWithBackground(image, background="black")
            if image.size != size:
                image = image.resize(size, resample=PIL.Image.LANCZOS)
            if detector:
                image = detector(image, *arg).convert("RGB")
            orig_image = image
            image = image_model.Preprocess(image, target) * pre_scale
            return torch.cat([image] * len(prompts)), orig_image

        self._image_inputs = [
            GetImageInput(image, pre_scale, detector, arg)
            for image, pre_scale, detector, arg in zip(
                self._images, self._pre_scales, self._detectors, self._detector_args
            )
        ]

        # self._offload_hooks = [
        #     accelerate.cpu_offload_with_hook(model, target.device())[1]
        #     for model in self._models
        # ]
        for model in self._models:
            model.to(target.device())

    def GetImages(self):
        return [x[1] for x in self._image_inputs]

    def __call__(self, model_input, timestep, text_embeddings):
        for i, (model, (image, _), post_scale) in enumerate(
            zip(self._models, self._image_inputs, self._post_scales)
        ):
            dbr, mbr = model(
                model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=image,
                conditioning_scale=post_scale,
                return_dict=False,
            )
            if i == 0:
                down_block_residuals, mid_block_residuals = dbr, mbr
            else:
                down_block_residuals = [
                    sum(x) for x in zip(down_block_residuals, dbr, strict=True)
                ]
                mid_block_residuals += mbr
        return down_block_residuals, mid_block_residuals


#
# -- Main pipeline --
#
class LayeredDiffusionPipeline:
    def GetRevision(self, model_name: str, revision: Optional[str] = None):
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

        return recommended_revision.get(model_name, default_revision)

    @torch.no_grad()
    def Connect(
        self,
        model_name: str,
        cache_path: Optional[str] = None,
        revision: Optional[str] = None,
        auth_token: Optional[str] = None,
        use_xformers: bool = True,
        device_type: str = "cuda",
    ):
        self._model_name = model_name
        self._device_type = device_type

        if cache_path:
            dataset = cache_path
        else:
            dataset = model_name

        extra_args = {
            "torch_dtype": torch.float32,
            "revision": self.GetRevision(model_name, revision),
        }
        if auth_token:
            extra_args["use_auth_token"] = auth_token

        # use DPM-Solver++ scheduler
        self.scheduler = Scheduler().Connect(dataset)
        extra_args["scheduler"] = self.scheduler.Get()

        # Prepare the StableDiffusion pipeline.
        pipe = StableDiffusionPipeline.from_pretrained(dataset, **extra_args)

        self._SetOptions(pipe, use_xformers, device_type)
        self._SetPipeline(pipe)
        return self

    @torch.no_grad()
    def ConnectCkpt(
        self,
        model_name: str,
        checkpoint_path: str,
        original_config_file: str = None,  # YAML file only if necessary
        vae_path: Optional[str] = None,
        embeddings: Optional[TextualInversion] = None,
        lora: Optional[LoRA] = None,
        use_xformers: bool = True,
        device_type: str = "cuda",
        scheduler_type: str = "dpm",  # use DPM-Solver++ scheduler
        extract_ema: bool = True,  # ignored if not applicable
        # After here, I copied the argument list from download_from_original_stable_diffusion_ckpt
        # without checking if they are compatible with this library.
        image_size: int = 512,
        prediction_type: str = None,
        model_type: str = None,
        num_in_channels: Optional[int] = None,
        upcast_attention: Optional[bool] = None,
        stable_unclip: Optional[str] = None,
        stable_unclip_prior: Optional[str] = None,
        clip_stats_path: Optional[str] = None,
        controlnet: Optional[bool] = None,
    ):
        self._model_name = model_name
        self._device_type = device_type

        from_safetensors = False
        safetensors_suffix = ".safetensors"
        if checkpoint_path[-len(safetensors_suffix) :] == safetensors_suffix:
            from_safetensors = True
        Debug(1, f"checkpoint_path: {checkpoint_path}")
        Debug(3, f"suffix: {checkpoint_path[-len(safetensors_suffix):]}")
        Debug(3, f"from_safetensors: {from_safetensors}")

        vae_sideloader = CkptVAESideloader(vae_path, device_type).Initialize()

        # pipe = load_pipeline_from_original_stable_diffusion_ckpt(
        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path=checkpoint_path,
            original_config_file=original_config_file,
            image_size=image_size,
            prediction_type=prediction_type,
            model_type=model_type,
            extract_ema=extract_ema,
            scheduler_type=scheduler_type,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            from_safetensors=from_safetensors,
            device=device_type,
            stable_unclip=stable_unclip,
            stable_unclip_prior=stable_unclip_prior,
            clip_stats_path=clip_stats_path,
            controlnet=controlnet,
            load_safety_checker=False,
        )
        self.scheduler = Scheduler().Wrap(scheduler_type, pipe.scheduler)

        vae_sideloader.Finalize()

        if lora:
            lora.Apply(device_type, pipe.text_encoder, pipe.unet)
        self._SetOptions(pipe, use_xformers, device_type)
        self._SetPipeline(pipe)
        if embeddings:
            embeddings.Apply(self.text_model)
        return self

    def CopyFrom(self, another):
        self._model_name = another._model_name
        self._device_type = another._device_type
        self.scheduler = Scheduler().CopyFrom(another.scheduler)
        self._SetPipeline(another.pipe)
        another._DeleteFields()
        return self

    def Dispose(self):
        self.pipe.to("cpu")
        self._DeleteFields()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def EnableCPUOffload(self):
        pipe.enable_model_cpu_offload()

    def _DeleteFields(self):
        del self.pipe
        del self.unet
        del self.text_model
        del self.image_model
        del self.scheduler

    def _SetOptions(self, pipe, use_xformers, device_type):
        # Attention slicing doesn't work with LoRA.
        # pipe.enable_attention_slicing()
        if use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        # TODO: this shouldn't be called when using EnableCPUOffload().
        pipe.to(device_type)

    def _SetPipeline(self, pipe):
        self.pipe = pipe
        self.unet = pipe.unet
        self.text_model = TextModel(
            pipe.tokenizer, pipe.text_encoder, self._device_type
        )
        self.image_model = ImageModel(pipe.vae, pipe.vae_scale_factor)

    def _ResetGenerator(self, rand_seed: Optional[int] = None):
        if not rand_seed:
            rand_seed = random.SystemRandom().randint(1, 4294967295)
        self._rand_seed = rand_seed
        self.scheduler.ResetGenerator(self.text_model.target, rand_seed)

    def _GetVaeEncoderAdjust(self):
        # Anything v3.0's VAE has a degradation problem in its encoder.
        # Multiplying the adjustment factor of 1.25 to the encoder output mitigates
        # the problem to a lower level.
        # See https://twitter.com/tomo161382/status/1601962971085041664 for details.
        #
        # TODO: revisit this function considering that now VAE is swappable.
        if self._model_name == "Linaqruf/anything-v3.0":
            return 1.25
        return 1.0

    def GetModelName(self):
        return self._model_name

    def GetShortModelName(self):
        return self._model_name.split("/")[-1]

    def GetRandSeed(self):
        return self._rand_seed

    @torch.no_grad()
    def __call__(
        self,
        num_steps: int,
        iterate: Union[Layer, List[Layer]],
        initialize: Optional[Union[Initializer, List[Initializer]]] = None,
        controlnet: Optional[ControlNet] = None,
        size: Optional[Tuple[int, int]] = None,
        default_encoding: Optional[StandardEncoding] = None,
        negative_prompt_scale: Union[float, Tuple[float, float]] = 1.0,
        use_rmm: bool = True,  # rmm = Residual Merge Method
        use_decomposed_rmm: bool = True,
        rand_seed: Optional[int] = None,
        eta: float = 0.0,
        vae_encoder_adjust: Optional[float] = None,
    ):
        self._ResetGenerator(rand_seed)

        if num_steps <= 0:
            raise ValueError(f"`num_steps` must be > 0. actual: {num_steps}")
        if not default_encoding:
            default_encoding = ShiftEncoding()
        if not iterate:
            raise ValueError("`iterate` should contain at least 1 layer.")
        elif not isinstance(iterate, list):
            layers = [iterate]
        else:
            layers = iterate
        layers = [l for l in layers if not l.disabled]
        if not layers:
            raise ValueError("`iterate` should contain at least 1 enabled layer.")
        if not use_rmm:
            if any(
                l.IsDistinct() or l.to_add or isinstance(l, SPLayer) for l in layers
            ):
                raise ValueError(
                    "Distinct layers, layers to add or SP layers are available only with RMM."
                )

        if not vae_encoder_adjust:
            vae_encoder_adjust = self._GetVaeEncoderAdjust()
        if vae_encoder_adjust != 1.0:
            Debug(1, f"Adjusting VAE Encoder level by {vae_encoder_adjust}")

        bglayer = BackgroundLayer(initialize, vae_encoder_adjust)
        prompts = Prompts(self.text_model, default_encoding, negative_prompt_scale)
        target = self.text_model.target

        if use_rmm:
            mm = self.ResidualMergeMethod(
                self.scheduler,
                self.unet,
                controlnet,
                target,
                bglayer,
                layers,
                prompts,
                decomposed=use_decomposed_rmm,
            )
        else:
            mm = self.LatentMergeMethod(
                self.scheduler, self.unet, controlnet, target, bglayer, layers, prompts
            )

        with autocast(target.device_type()):
            mm.Initialize(self.image_model, num_steps, size, self.pipe, eta)

            for i, ts in enumerate(self.pipe.progress_bar(mm.timesteps)):
                remaining = bglayer.GetRemaining(mm.timesteps, i)
                Debug(
                    3,
                    f"remaining: {remaining:.2f} @ step {i + 1} of {len(mm.timesteps)}",
                )

                mm.Step(i, ts, remaining)

                # TODO: implement residual inspection
                # latents_debug = self.scheduler.InspectLatents(
                #    residuals_for_layers[0], ts, latents
                # )
                # Debug(3, "", self.image_model.Decode(latents_debug)[0])

            result = mm.Result(self.image_model)

        # Offload last model to CPU
        if (
            hasattr(self.pipe, "final_offload_hook")
            and self.pipe.final_offload_hook is not None
        ):
            self.pipe.final_offload_hook.offload()
            torch.cuda.empty_cache()

        return result

    class LatentMergeMethod:
        def __init__(
            self, scheduler, unet, controlnet, target, bglayer, layers, prompts
        ):
            self.scheduler = scheduler
            self.unet = unet
            self.controlnet = controlnet
            self.target = target
            self.bglayer = bglayer
            self.layers = layers
            self.all_layers = [bglayer] + layers
            self.prompts = prompts
            self.decomposed = False
            Debug(1, f"number of all layers = {len(self.all_layers)}")

        def Initialize(self, image_model, num_steps, size, pipe, eta):
            for l in self.layers:
                l.Initialize(
                    size,
                    image_model,
                    self.prompts,
                    self.target,
                    decomposed=self.decomposed,
                )
            latents, timesteps = self.bglayer.Initialize(
                num_steps, size, self.scheduler, image_model, self.prompts, self.target
            )
            self.latents = latents
            self.scheduler.PrepareExtraStepKwargs(pipe, eta)
            self.timesteps = timesteps
            if self.controlnet:
                self.controlnet.Initialize(
                    self.bglayer.GetSize(), image_model, self.prompts, self.target
                )

        def Step(self, i, ts, remaining):
            latents_for_prompts = [self.latents] * len(self.prompts)

            residuals_for_layers = self.UNet(latents_for_prompts, ts, remaining)

            latents_for_layers = self.scheduler.Step(
                residuals_for_layers, ts, [self.latents] * len(self.all_layers)
            )
            next_latents = latents_for_layers[0]
            next_latents = self.bglayer.OverwriteLatents(  # unconditionally
                self.scheduler, next_latents, self.timesteps, i, force_overwrite=True
            )
            for layer, latents in zip(self.layers, latents_for_layers[1:]):
                next_latents = layer.Merge(next_latents, latents, remaining)
            next_latents = self.bglayer.OverwriteLatents(  # conditioned by strength
                self.scheduler, next_latents, self.timesteps, i
            )
            self.latents = next_latents

        def UNet(self, latents_for_prompts, ts, remaining, decomposed=False):
            residuals_for_prompts = self.prompts.PredictResiduals(
                self.scheduler,
                self.unet,
                self.controlnet,
                latents_for_prompts,
                ts,
                remaining,
            )
            residuals_for_layers = [
                l.GetResidual(residuals_for_prompts, remaining, decomposed, self.target)
                for l in self.all_layers
            ]
            return residuals_for_layers

        def ControlNetImagesAndTitles(self):
            if not self.controlnet:
                return [], []
            rs = [
                (ResizeImage(image, S_SIZE), f"controlnet #{i}")
                for i, image in enumerate(self.controlnet.GetImages())
            ]
            return (list(a) for a in zip(*rs))

        def Result(self, image_model):
            return (
                image_model.Decode(self.latents)[0],
                ConcatImages(*self.ControlNetImagesAndTitles()),
            )

    class ResidualMergeMethod(LatentMergeMethod):
        def __init__(
            self,
            scheduler,
            unet,
            controlnet,
            target,
            bglayer,
            layers,
            prompts,
            decomposed,
        ):
            Debug(1, f"Using RMM with decomposed = {decomposed}")
            super().__init__(
                scheduler, unet, controlnet, target, bglayer, layers, prompts
            )
            rmm_index = 0
            rmm_layers = []
            for i, layer in enumerate(self.all_layers):
                if layer.IsDistinct():
                    rmm_index += 1
                    rmm_layers.append((layer, i, rmm_index))
                else:
                    rmm_layers.append((layer, i, 0))
            num_rmm_latents = rmm_index + 1
            Debug(1, f"number of rmm latents = {num_rmm_latents}")
            self.rmm_layers = rmm_layers
            self.num_rmm_latents = num_rmm_latents
            self.decomposed = decomposed  # overwrite

        def Initialize(self, image_model, num_steps, size, pipe, eta):
            Debug(3, f"RMM.Initialize()")
            super().Initialize(image_model, num_steps, size, pipe, eta)
            self.rmm_latents = [self.latents] * self.num_rmm_latents

        def Step(self, i, ts, remaining):
            Debug(4, f"RMM.Step() for step {i + 1}")
            latents_for_prompts = [None] * len(self.prompts)
            for layer, k, rmm_index in self.rmm_layers:
                l = self.rmm_latents[rmm_index]
                for m in layer._indices:
                    latents_for_prompts[m] = l
            for k, l in enumerate(latents_for_prompts):
                if l is None:
                    raise ValueError(f"unexpected error: missing latents for index {k}")

            Debug(4, f"Calling UNet() for step {i + 1}")
            residuals_for_layers = self.UNet(
                latents_for_prompts, ts, remaining, self.decomposed
            )

            latents, rmm_latents = self.RMM(residuals_for_layers, i, ts, remaining)
            self.latents = latents
            self.rmm_latents = rmm_latents

        def Result(self, image_model):
            ctn_images, ctn_titles = self.ControlNetImagesAndTitles()
            # Decode images one by one to avoid OOM
            return (
                image_model.Decode(self.latents)[0],
                ConcatImages(
                    ctn_images
                    + [
                        ResizeImage(image_model.Decode(ex)[0], S_SIZE)
                        for ex in self.rmm_latents
                    ],
                    ctn_titles
                    + [
                        "RMM Image: common" if i == 0 else f"distinct #{i}"
                        for i in range(self.num_rmm_latents)
                    ],
                ),
            )

        def RMM(self, residuals_for_layers, i, ts, remaining):
            Debug(4, f"RMM.RMM() for step {i + 1}")
            residuals_for_bg = residuals_for_layers[0]
            residuals_for_rmm = [residuals_for_bg] * self.num_rmm_latents
            residuals_for_all = residuals_for_bg
            for layer, k, rmm_index in self.rmm_layers[1:]:
                residuals_for_all = layer.Merge(
                    residuals_for_all,
                    residuals_for_layers[k],
                    remaining,
                    self.decomposed,
                )
                for m in range(self.num_rmm_latents):
                    if rmm_index in (0, m):
                        residuals_for_rmm[m] = layer.Merge(
                            residuals_for_rmm[m],
                            residuals_for_layers[k],
                            remaining,
                            self.decomposed,
                        )
            latents_for_rmm = self.scheduler.Step(
                [
                    r.Compose() if self.decomposed else r
                    for r in [residuals_for_bg, residuals_for_all] + residuals_for_rmm
                ],
                ts,
                [self.rmm_latents[0], self.latents] + self.rmm_latents,
            )
            latents = self.SynthesizeLatents(latents_for_rmm, -1, i, remaining)
            rmm_latents = [
                self.SynthesizeLatents(latents_for_rmm, m, i, remaining)
                for m in range(self.num_rmm_latents)
            ]
            return latents, rmm_latents

        def SynthesizeLatents(self, latents_for_rmm, rmm_index, i, remaining):
            Debug(4, f"RMM.SynthesizeLatents({rmm_index}) for step {i + 1}")
            next_latents = latents_for_rmm[0]
            next_latents = self.bglayer.OverwriteLatents(  # unconditionally
                self.scheduler,
                next_latents,
                self.timesteps,
                i,
                force_overwrite=True,
            )
            next_latents = self.UnionLayerMask(
                self.rmm_layers,
                rmm_index,
                remaining,
                black=next_latents,
                white=latents_for_rmm[rmm_index + 2],
            )
            next_latents = self.bglayer.OverwriteLatents(  # conditioned by strength
                self.scheduler, next_latents, self.timesteps, i
            )
            return next_latents

        def UnionLayerMask(self, rmm_layers, m, remaining, black, white):
            Debug(4, f"RMM.UnionLayerMask(): latents {m}, remaining {remaining:.2f}")
            mask = 1.0
            for layer, i, rmm_index in rmm_layers[1:]:
                if (m < 0) or (rmm_index in (0, m)):
                    Debug(4, f"Add layer mask {i}, rmm_index {rmm_index}")
                    mask = layer.UnionMask(other_mask=mask, remaining=remaining)
            return LatentMask.ApplyMask(mask, black=black, white=white)
