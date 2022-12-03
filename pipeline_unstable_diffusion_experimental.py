# @title Experimental code for UnstableDiffusionPipeline
# Code in this file is experimental and may not work with the latest pipeline.
# See the following web page for the usage.
# https://github.com/nanashi161382/unstable_diffusion/tree/main
from pipeline_unstable_diffusion import StandardEncoding

from IPython.display import display
import numpy as np
import re
import torch
from typing import Optional, List, Union, Callable, Tuple


class ExperimentalEncoding(StandardEncoding):
    def __init__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ):
        super().__init__(prompt, negative_prompt)

    class EncodedSegment:
        def __init__(self, embeddings, weight):
            self.weight = torch.tensor(weight).to(embeddings[0].dtype)
            self.unweighted_mean = (
                embeddings[0][0].float().mean(axis=[-2, -1]).to(embeddings[0].dtype)
            )
            self.full = embeddings[0][0] * weight
            self.sot = self.full[0]
            self.eot = embeddings[1][0] * weight
            self.end = self.full[-1]
            effective_length = embeddings[2]
            self.words = self.full[1:effective_length]

        @classmethod
        def Encode(cls, text_model, text, weight, fail_on_truncation: bool = True):
            embeddings = text_model.EncodeText(text, fail_on_truncation)
            return cls(embeddings, weight)

        @classmethod
        def AnnotateAndEncode(cls, text_model, chunk, fail_on_truncation: bool = True):
            phrase, weight, repeat = StandardEncoding.GetAnnotation(chunk)
            display([phrase, weight, repeat])
            return cls.Encode(text_model, phrase, weight, fail_on_truncation), repeat

    class SequentialBase(StandardEncoding.AdjustingBase):
        def __init__(self):
            super().__init__()
            self.sot_state = None
            self.eot_state = None
            self.end_state = None
            self.word_states = []
            self.word_states_len = 0
            self.eot_states = []

        def Add(self, enc, segment):
            super().Add(enc, segment.unweighted_mean)
            self.sot_state = enc.AddOrInit(self.sot_state, segment.sot)
            self.eot_state = enc.AddOrInit(self.eot_state, segment.eot)
            self.end_state = enc.AddOrInit(self.end_state, segment.end)
            self.word_states.append(segment.words)
            self.word_states_len += segment.words.shape[0]
            self.eot_states.append(segment.eot.unsqueeze(0))

        def Average(self, chunk_len):
            super().Average(chunk_len)
            chunk_len = torch.tensor(chunk_len).to(self.unweighted_mean.dtype)
            self.sot_state /= chunk_len
            self.eot_state /= chunk_len
            self.end_state /= chunk_len

    class Parallel(StandardEncoding.AdjustingBase):
        def __init__(self):
            super().__init__()
            self.states = None

        def Add(self, enc, segment):
            super().Add(enc, segment.unweighted_mean)
            self.states = enc.AddOrInit(self.states, segment.full)

        def Average(self, chunk_len):
            super().Average(chunk_len)
            self.states /= chunk_len

        def ToStates(self, *input, **kwargs):
            return self.states


class EotEncoding(ExperimentalEncoding):
    def __init__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        repeat: bool = False,
    ):
        """
        Args:
            prompt (`str`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
        super().__init__(prompt, negative_prompt)
        self._repeat = repeat

    def EncodeText(self, text_model, text: str):
        embeddings = text_model.EncodeText(text)
        full_states = embeddings[0][0]
        eot_state = embeddings[1][0]
        end_state = full_states[-1]

        old_mean = full_states.float().mean(axis=[-2, -1]).to(full_states.dtype)
        if self._repeat:
            new_states = torch.cat(
                (
                    full_states[0].unsqueeze(0),
                    eot_state.repeat((len(full_states) - 2), 1),
                    end_state.unsqueeze(0),
                ),
                dim=0,
            )
        else:
            new_states = torch.cat(
                (
                    full_states[0].unsqueeze(0),
                    eot_state.unsqueeze(0),
                    end_state.repeat((len(full_states) - 2), 1),
                ),
                dim=0,
            )
        new_mean = new_states.float().mean(axis=[-2, -1]).to(new_states.dtype)
        new_states *= (old_mean / new_mean).unsqueeze(-1)
        display(new_states.shape)
        display(new_states)
        return torch.cat([torch.unsqueeze(new_states, 0)], dim=0).to(text_model._device)


class SegmentedEotEncoding(ExperimentalEncoding):
    def __init__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        repeat: bool = False,
    ):
        """
        Args:
            prompt (`str`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
        super().__init__(prompt, negative_prompt)
        self._repeat = repeat

    class Sequential(ExperimentalEncoding.SequentialBase):
        def ToStates(self, max_length, repeat):
            states = (
                [self.sot_state.unsqueeze(0)]
                + (self.eot_states * repeat)
                + [self.eot_state.unsqueeze(0)]
            )
            states = torch.cat(
                states + ([self.end_state.unsqueeze(0)] * (max_length - len(states))),
                dim=0,
            )
            return states

    def EncodeText(self, text_model, text: str):
        chunks = [x.strip() for x in text.split(",")]
        display(chunks)

        proc = self.Sequential()
        chunk_len = 0
        for chunk in chunks:
            segment, repeat = self.EncodedSegment.AnnotateAndEncode(text_model, chunk)
            chunk_len += repeat
            for i in range(repeat):
                proc.Add(self, segment)
        proc.Average(chunk_len)

        eot_repeat = 1
        if self._repeat:
            eot_repeat = int((text_model.max_length() - 3) / len(proc.eot_states))
        new_states = proc.ToStates(text_model.max_length(), eot_repeat)
        return proc.AdjustMean(new_states, text_model._device)


class SegmentedEncoding(ExperimentalEncoding):
    def __init__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        sequential: bool = True,
    ):
        """
        Args:
            prompt (`str`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            sequential (`bool`):
        """
        super().__init__(prompt, negative_prompt)
        self._sequential = sequential

    class Sequential(ExperimentalEncoding.SequentialBase):
        def ToStates(self, max_length):
            states = (
                [self.sot_state.unsqueeze(0)]
                + self.word_states
                + [self.eot_state.unsqueeze(0)]
            )
            states_len = self.word_states_len + 2
            states = torch.cat(
                states + ([self.end_state.unsqueeze(0)] * (max_length - states_len)),
                dim=0,
            )
            return states

    def EncodeText(self, text_model, text: str):
        chunks = [x.strip() for x in text.split(",")]
        display(chunks)

        if self._sequential:
            proc = self.Sequential()
        else:
            proc = self.Parallel()

        chunk_len = 0
        for chunk in chunks:
            segment, repeat = self.EncodedSegment.AnnotateAndEncode(text_model, chunk)
            chunk_len += repeat
            for i in range(repeat):
                proc.Add(self, segment)
        proc.Average(chunk_len)

        new_states = proc.ToStates(text_model.max_length())
        return proc.AdjustMean(new_states, text_model._device)
