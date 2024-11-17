import copy
from collections import deque
import math
from transformers import (
    CLIPPreTrainedModel,
    CLIPTextConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.models.clip.modeling_clip import (
    CLIPEncoderLayer,
    CLIPMLP,
    CLIPAttention,
)
import torch
import torch.nn as nn
from typing import Any, Optional, List, Union, Callable, Tuple, Dict

#
# -- Debug functions --
#
debugger = None


def SetDebugger(new_debugger):
    global debugger
    debugger = new_debugger


def Debug(level: int, title: str, obj=None):
    if debugger:
        debugger(level, title, obj)
    else:
        if title:
            print(title)
        if obj is not None:
            display(obj)


#
# -- Concat model --
#
class CLIPTextDeprojectorConcat:
    def __init__(self, model1, len1, model2, len2):
        self.model1 = model1
        self.len1 = len1
        self.model2 = model2
        self.len2 = len2

    def to(self, *args, **kwargs):
        self.model1.to(*args, **kwargs)
        self.model2.to(*args, **kwargs)
        return self

    def Deproject(self, embeds):
        return self.model1.Deproject(embeds)

    def Inference(self, embeds, from_projected=False, fuse=None, **kwargs):
        result1 = self.model1.Inference(embeds, from_projected, fuse, **kwargs)
        result2 = self.model2.Inference(embeds, from_projected, fuse, **kwargs)
        return torch.cat(
            [result1[:, : self.len1, :], result2[:, 1 : self.len2 + 1, :]], dim=1
        )


#
# -- Base --
#
class CLIPTextDeprojectorModel(CLIPPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_size = config.hidden_size
        self.seq_len = config.max_position_embeddings

    def OffloadTensor(self, *ts):
        for t in ts:
            t.to("cpu")
        torch.cuda.empty_cache()


class CLIPTextDeprojectorBase(CLIPTextDeprojectorModel):
    def __init__(self, config):
        super().__init__(config)

        self.register_buffer(
            "deprojection", torch.empty([self.embed_size, config.projection_dim])
        )
        self.register_buffer("sos_embed", torch.empty([self.embed_size]))

    def to(self, *args, **kwargs):
        self.deprojection.to(*args, **kwargs)
        self.sos_embed.to(*args, **kwargs)
        return self

    def GetSOSEmb(self, bsz, device):
        return self.sos_embed.view([1, 1, -1]).repeat([bsz, 1, 1]).to(device)

    def Deproject(self, embeds):
        return nn.functional.linear(embeds, self.deprojection, None)

    def RunForTraining(self, embeds, final_states, **kwargs):
        raise NotImplementedError()

    def Inference(self, embeds, from_projected=False, **kwargs):
        raise NotImplementedError()


class ElementWiseLinear(CLIPTextDeprojectorModel):
    def __init__(self, config, f_size):
        super().__init__(config)
        self.weight = nn.Parameter(torch.ones(f_size))
        self.bias = nn.Parameter(torch.zeros(f_size))

    def forward(self, x):
        return (self.weight * x) + self.bias


#
# -- VRNN (Vicinity RNN) model --
#
class CLIPTextDeprojectorVRNNConfig(CLIPTextConfig):
    model_type = "clip_text_deprojector_vrnn_model"

    default_residual = False
    default_detach_all = False
    default_detach_s2 = False

    default_m1_norm = False
    default_m1_vicinity = 1
    default_m1_split = False
    default_m1_type = "linear"
    default_m1_f_size = 64
    default_m1_p_size = 3
    default_m1_layers = 1

    default_m2_use_s2 = False
    default_m2_norm = False
    default_m2_vicinity = 1
    default_m2_type = "linear"
    default_m2_f_size = 64
    default_m2_p_size = 3
    default_m2_layers = 1
    default_m2_ex_split = False
    default_m2_ex_type = "same"
    default_m2_ex_f_size = 64
    default_m2_ex_p_size = 3

    default_channel_act = "gelu"
    default_channel_ewl = False
    default_channel_ewl2 = True

    # deprecated
    default_m1_channel = False
    default_m1_mlp = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c = self.__class__

        self.residual = kwargs.get("residual", c.default_residual)
        self.detach_all = kwargs.get("detach_all", c.default_detach_all)
        self.detach_s2 = kwargs.get("detach_s2", c.default_detach_s2)

        self.m1_norm = kwargs.get("m1_norm", c.default_m1_norm)
        self.m1_vicinity = kwargs.get("m1_vicinity", c.default_m1_vicinity)
        self.m1_split = kwargs.get("m1_split", c.default_m1_split)
        self.m1_type = kwargs.get("m1_type", c.default_m1_type)
        self.m1_f_size = kwargs.get("m1_f_size", c.default_m1_f_size)
        self.m1_p_size = kwargs.get("m1_p_size", c.default_m1_p_size)
        self.m1_layers = kwargs.get("m1_layers", c.default_m1_layers)

        self.m2_use_s2 = kwargs.get("m2_use_s2", c.default_m2_use_s2)
        self.m2_norm = kwargs.get("m2_norm", c.default_m2_norm)
        self.m2_vicinity = kwargs.get("m2_vicinity", c.default_m2_vicinity)
        self.m2_type = kwargs.get("m2_type", c.default_m2_type)
        self.m2_f_size = kwargs.get("m2_f_size", c.default_m2_f_size)
        self.m2_p_size = kwargs.get("m2_p_size", c.default_m2_p_size)
        self.m2_layers = kwargs.get("m2_layers", c.default_m2_layers)
        self.m2_ex_split = kwargs.get("m2_ex_split", c.default_m2_ex_split)
        self.m2_ex_type = kwargs.get("m2_ex_type", c.default_m2_ex_type)
        self.m2_ex_f_size = kwargs.get("m2_ex_f_size", c.default_m2_ex_f_size)
        self.m2_ex_p_size = kwargs.get("m2_ex_p_size", c.default_m2_ex_p_size)

        self.channel_act = kwargs.get("channel_act", c.default_channel_act)
        self.channel_ewl = kwargs.get("channel_ewl", c.default_channel_ewl)
        self.channel_ewl2 = kwargs.get("channel_ewl2", c.default_channel_ewl2)

        # deprecated
        self.m1_channel = kwargs.get("m1_channel", c.default_m1_channel)
        self.m1_mlp = kwargs.get("m1_mlp", c.default_m1_mlp)
        if self.m1_channel:
            self.m1_type = "channel"
        elif self.m1_mlp:
            self.m1_type = "mlp"


class CLIPTextDeprojectorVRNN(CLIPTextDeprojectorBase):
    config_class = CLIPTextDeprojectorVRNNConfig

    freeze_final_norm = True

    def __init__(self, config: CLIPTextDeprojectorVRNNConfig):
        super().__init__(config)
        embed_size = self.embed_size
        self.detach_s2 = config.detach_s2
        self.detach_all = config.detach_all

        self.model1 = CLIPTextDeprojectorVRNN_1(config)
        self.model2 = CLIPTextDeprojectorVRNN_2(config)

        self.final_norm = nn.LayerNorm(embed_size, eps=config.layer_norm_eps)
        if self.__class__.freeze_final_norm:
            for p in self.final_norm.parameters():
                p.requires_grad = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model1.to(*args, **kwargs)
        self.model2.to(*args, **kwargs)
        self.final_norm.to(*args, **kwargs)
        return self

    def MayAddWeight(self, k, v, weights, prefix):
        if k.startswith(prefix):
            weights[k[len(prefix) :]] = v

    def LoadWeights(self, weights):
        final_norm_weights = {}
        for k, v in weights.items():
            self.MayAddWeight(k, v, final_norm_weights, "text_model.final_layer_norm.")
        self.final_norm.load_state_dict(final_norm_weights)

    def RunForTraining(self, embeds, final_states, **kwargs):
        return self.Inference(embeds)

    def Inference(self, embeds, from_projected=False, **kwargs):
        device = embeds.device
        bsz = embeds.shape[0]

        if from_projected:
            embeds = self.Deproject(embeds)

        embeds = embeds.unsqueeze(1)
        states = self.model1(embeds)
        states = [self.final_norm(s) for s in states]
        if self.detach_all:
            final_states = states
            states = [s.detach() for s in states]

        s2 = states[0]
        if self.detach_s2 and not self.detach_all:
            s2 = s2.detach()

        while len(states) < (self.seq_len - 1):
            sn = self.model2(s2, states)
            sn = self.final_norm(sn)
            if self.detach_all:
                final_states.append(sn)
                states.append(sn.detach())
            else:
                states.append(sn)

        if not self.detach_all:
            final_states = states
        return torch.cat([self.GetSOSEmb(bsz, device)] + final_states, dim=1).clone()


class CLIPTextDeprojectorVRNN_Model(CLIPTextDeprojectorModel):
    def __init__(
        self,
        config: CLIPTextDeprojectorVRNNConfig,
        norm,
        vicinity,
        type,
        f_size,
        p_size,
        layers,
        ex_type=None,
        ex_f_size=None,
        ex_p_size=None,
    ):
        super().__init__(config)
        self.use_residual = config.residual
        self.use_norm = norm
        self.vicinity = vicinity

        self.type = type
        self.f_size = f_size
        self.p_size = p_size

        if ex_type and (ex_type != "same"):
            self.ex_type = ex_type
            self.ex_f_size = ex_f_size
            self.ex_p_size = ex_p_size
        else:
            self.ex_type = type
            self.ex_f_size = f_size
            self.ex_p_size = p_size

        if layers < 1:
            raise ValueError(f"{layers=} should be more than 0.")
        self.layers = layers
        self.extra_layers = self.layers - 1

    def CreateLayer(self, config, in_size, out_size, extra=False, split=False):
        if extra:
            n_type = self.ex_type
            f_size = self.ex_f_size
            p_size = self.ex_p_size
        else:
            n_type = self.type
            f_size = self.f_size
            p_size = self.p_size

        match n_type:
            case "channel":
                return CLIPTextDeprojectorVRNN_Channelled(
                    config, f_size, in_size, out_size
                )
            case "mlp":
                return CLIPTextDeprojectorVRNN_MLP(
                    config, f_size, in_size, out_size, split=split
                )
            case "lorl":
                return CLIPTextDeprojectorVRNN_LowRankLinear(
                    config, f_size, in_size, out_size
                )
            case "pool":
                return CLIPTextDeprojectorVRNN_PoolMLP(
                    config, p_size, f_size, in_size, out_size
                )
            case _:
                return CLIPTextDeprojectorVRNN_Linear(
                    config, in_size, out_size, split=split
                )

    def RunLayer(self, results, impl, norm, residual):
        if norm:
            results = [norm[i](r) for i, r in enumerate(results)]
        results = impl(results)
        if residual:
            results = [r + res for r, res in zip(results, residual)]
        return results


class CLIPTextDeprojectorVRNN_1(CLIPTextDeprojectorVRNN_Model):
    def __init__(self, config: CLIPTextDeprojectorVRNNConfig):
        super().__init__(
            config,
            config.m1_norm,
            config.m1_vicinity,
            config.m1_type,
            config.m1_f_size,
            config.m1_p_size,
            config.m1_layers,
        )
        if self.use_norm:
            self.norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.impl = self.CreateLayer(config, 1, self.vicinity, split=config.m1_split)

        if self.extra_layers > 0:
            if self.use_norm:
                self.extra_norm = nn.ModuleList(
                    nn.ModuleList(
                        nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
                        for _ in range(self.vicinity)
                    )
                    for _ in range(self.extra_layers)
                )
            self.extra_impl = nn.ModuleList(
                self.CreateLayer(
                    config,
                    self.vicinity,
                    self.vicinity,
                    extra=True,
                    split=config.m1_split,
                )
                for _ in range(self.extra_layers)
            )

    def forward(self, embeds):
        residual = [embeds] * self.vicinity if self.use_residual else None
        norm = [self.norm] if self.use_norm else None
        results = self.RunLayer([embeds], self.impl, norm, residual)

        for i in range(self.extra_layers):
            residual = results if self.use_residual else None
            norm = self.extra_norm[i] if self.use_norm else None
            results = self.RunLayer(results, self.extra_impl[i], norm, residual)

        return results


class CLIPTextDeprojectorVRNN_2(CLIPTextDeprojectorVRNN_Model):
    def __init__(self, config: CLIPTextDeprojectorVRNNConfig):
        super().__init__(
            config,
            config.m2_norm,
            config.m2_vicinity,
            config.m2_type,
            config.m2_f_size,
            config.m2_p_size,
            config.m2_layers,
            config.m2_ex_type,
            config.m2_ex_f_size,
            config.m2_ex_p_size,
        )
        self.use_s2 = config.m2_use_s2

        mult = self.vicinity
        if self.use_s2:
            mult += 1

        self.impl = self.CreateLayer(config, mult, 1)

        if self.use_norm:
            # [0] for the last layer
            # [n] for the (n-1)th layer
            self.norm = nn.ModuleList(
                nn.ModuleList(
                    nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
                    for _ in range(mult)
                )
                for _ in range(self.layers)
            )

        if self.extra_layers > 0:
            self.extra_impl = nn.ModuleList(
                self.CreateLayer(
                    config, mult, mult, extra=True, split=config.m2_ex_split
                )
                for _ in range(self.extra_layers)
            )

    def MakeInput(self, states):
        if self.vicinity <= len(states):
            return states[-self.vicinity :]

        return [
            torch.zeros_like(states[0]) for _ in range(self.vicinity - len(states))
        ] + states

    def forward(self, s2, states):
        states = self.MakeInput(states)
        if self.use_s2:
            states = [s2] + states

        for i in range(self.extra_layers):
            residual = states if self.use_residual else None
            norm = self.norm[i + 1] if self.use_norm else None
            states = self.RunLayer(states, self.extra_impl[i], norm, residual)

        residual = states[-1:] if self.use_residual else None
        norm = self.norm[0] if self.use_norm else None
        return self.RunLayer(states, self.impl, norm, residual)[0]


class CLIPTextDeprojectorVRNN_Linear(CLIPTextDeprojectorModel):
    def __init__(
        self,
        config: CLIPTextDeprojectorVRNNConfig,
        input_mult,
        output_mult,
        split=False,
    ):
        super().__init__(config)
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.split = split

        if split and (input_mult == output_mult):
            self.fc = nn.ModuleList(
                nn.Linear(self.embed_size, self.embed_size) for _ in range(input_mult)
            )
        else:
            self.fc = nn.Linear(
                self.embed_size * input_mult, self.embed_size * output_mult
            )

    def forward(self, states):
        if self.split:
            return [self.fc[i](s) for i, s in enumerate(states)]

        if len(states) == 1:
            states = states[0]
        else:
            states = torch.cat(states, dim=2)

        states = self.fc(states)
        if self.output_mult == 1:
            return [states]
        states = states.view(states.shape[:-1] + (-1, self.embed_size))
        return [states[:, :, i, :] for i in range(self.output_mult)]


class CLIPTextDeprojectorVRNN_Channelled(CLIPTextDeprojectorModel):
    def __init__(
        self,
        config: CLIPTextDeprojectorVRNNConfig,
        f_size,
        input_mult,
        output_mult,
    ):
        super().__init__(config)
        if f_size < 1:
            raise ValueError(f"{f_size=} should be more than 0.")

        self.output_mult = output_mult
        self.use_ewl1 = config.channel_ewl
        self.use_ewl2 = config.channel_ewl and config.channel_ewl2

        self.act = ACT2FN[config.channel_act]
        self.fc0 = nn.ModuleList(
            nn.Linear(1, f_size, bias=False) for _ in range(input_mult)
        )
        self.fc1 = nn.ModuleList(
            nn.Linear(self.embed_size, f_size) for _ in range(input_mult)
        )
        self.fc2 = nn.Linear(f_size * input_mult, output_mult)

        if self.use_ewl1:
            self.ewl1 = nn.ModuleList(
                ElementWiseLinear(config, self.embed_size) for _ in range(input_mult)
            )
        if self.use_ewl2:
            self.ewl2 = nn.ModuleList(
                ElementWiseLinear(config, self.embed_size) for _ in range(input_mult)
            )

    def Channelize(self, i, s):
        s1 = s
        if self.use_ewl1:
            s1 = self.ewl1[i](s1)
        s1 = s1.unsqueeze(-1)
        s1 = self.fc0[i](s1)
        ss = self.fc1[i](s).unsqueeze(-2)
        ss = ss.expand(s1.shape)
        return self.act(s1 + ss)

    def forward(self, states):
        states = torch.cat(
            [self.Channelize(i, s) for i, s in enumerate(states)], dim=-1
        )
        states = self.fc2(states)
        states = states.transpose(-2, -1)
        states = [states[:, :, i, :] for i in range(self.output_mult)]
        if self.use_ewl2:
            states = [self.ewl2[i](s) for i, s in enumerate(states)]
        return states


class CLIPTextDeprojectorVRNN_MLP(CLIPTextDeprojectorModel):
    def __init__(
        self,
        config: CLIPTextDeprojectorVRNNConfig,
        f_size,
        input_mult,
        output_mult,
        split=False,
    ):
        super().__init__(config)
        if f_size < 1:
            raise ValueError(f"{f_size=} should be more than 0.")

        self.act = ACT2FN[config.hidden_act]
        self.o_f_size = -1
        if split and (input_mult == output_mult):
            self.fc1 = nn.ModuleList(
                nn.Linear(self.embed_size, f_size) for _ in range(input_mult)
            )
            self.fc2 = nn.ModuleList(
                nn.Linear(f_size, self.embed_size) for _ in range(input_mult)
            )
        elif split and (input_mult == 1):
            self.fc1 = nn.ModuleList(
                nn.Linear(self.embed_size, f_size * output_mult)
                for _ in range(input_mult)
            )
            self.fc2 = nn.ModuleList(
                nn.Linear(f_size, self.embed_size) for _ in range(output_mult)
            )
        else:
            self.fc1 = nn.ModuleList(
                nn.Linear(self.embed_size, f_size) for _ in range(input_mult)
            )
            self.fc2 = nn.Linear(f_size * input_mult, self.embed_size * output_mult)

        self.split = split
        self.f_size = f_size
        self.input_mult = input_mult
        self.output_mult = output_mult

    def forward(self, states):
        states = [self.fc1[i](s) for i, s in enumerate(states)]
        if self.split and (self.input_mult == self.output_mult):
            states = [self.act(s) for s in states]
            states = [self.fc2[i](s) for i, s in enumerate(states)]
            return states
        elif self.split and (self.input_mult == 1):
            states = self.act(torch.cat(states, dim=-1))
            states = states.view(states.shape[:-1] + (-1, self.f_size))
            states = [self.fc2[i](states[:, :, i, :]) for i in range(self.output_mult)]
            return states

        states = torch.cat(states, dim=-1)
        states = self.act(states)
        states = self.fc2(states)
        if self.output_mult == 1:
            return [states]
        states = states.view(states.shape[:-1] + (-1, self.embed_size))
        return [states[:, :, i, :] for i in range(self.output_mult)]


class CLIPTextDeprojectorVRNN_LowRankLinear(CLIPTextDeprojectorModel):
    def __init__(
        self,
        config: CLIPTextDeprojectorVRNNConfig,
        f_size,
        input_mult,
        output_mult,
    ):
        super().__init__(config)
        if f_size < 1:
            raise ValueError(f"{f_size=} should be more than 0.")

        self.fc1 = nn.ModuleList(
            nn.Linear(self.embed_size, f_size) for _ in range(input_mult)
        )
        self.fc2 = nn.Linear(f_size * input_mult, self.embed_size * output_mult)

        self.output_mult = output_mult

    def forward(self, states):
        states = [self.fc1[i](s) for i, s in enumerate(states)]
        states = torch.cat(states, dim=-1)
        states = self.fc2(states)
        if self.output_mult == 1:
            return [states]
        states = states.view(states.shape[:-1] + (-1, self.embed_size))
        return [states[:, :, i, :] for i in range(self.output_mult)]


class CLIPTextDeprojectorVRNN_PoolMLP(CLIPTextDeprojectorModel):
    def __init__(
        self,
        config: CLIPTextDeprojectorVRNNConfig,
        p_size,
        f_size,
        input_mult,
        output_mult,
    ):
        super().__init__(config)
        if f_size < 1:
            raise ValueError(f"{f_size=} should be more than 0.")

        self.pool = nn.MaxPool1d(p_size)
        self.fc1 = nn.ModuleList(
            nn.Linear(self.embed_size, f_size * p_size) for _ in range(input_mult)
        )
        self.fc2 = nn.Linear(f_size * input_mult, self.embed_size * output_mult)

        self.output_mult = output_mult

    def forward(self, states):
        states = [self.pool(self.fc1[i](s)) for i, s in enumerate(states)]
        states = torch.cat(states, dim=-1)
        states = self.fc2(states)
        if self.output_mult == 1:
            return [states]
        states = states.view(states.shape[:-1] + (-1, self.embed_size))
        return [states[:, :, i, :] for i in range(self.output_mult)]


#
# -- Ensemble --
#
class CLIPTextDeprojectorEnsemble(CLIPTextDeprojectorBase):
    default_fuse = 0.0

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = self.embed_size

    def forward(self, embeds, states):
        raise NotImplementedError()

    def average(self, ls, size):
        if size == 1:
            return ls[0]
        else:
            return sum(ls) / size

    def RunForTraining(self, embeds, final_states, **kwargs):
        if self.config.ensemble_size > 1:
            raise ValueError(
                f"ensemble_size must be 1 for training: {self.config.ensemble_size}"
            )
        bsz, embed_dim = embeds.shape
        if embed_dim != self.config.hidden_size:
            raise ValueError(
                f"Dimension of `embeds` must match the model's dimension."
                f" embeds.shape = {embeds.shape}"
            )
        states = final_states[:, 1:-1, :]  # Ignore SOS token and the last token
        states = self(embeds, [states])[0]

        sos_embeds = self.GetSOSEmb(bsz, states.device)
        result = torch.cat([sos_embeds, states], dim=1)
        return result

    def Inference(self, embeds, from_projected=False, fuse=None, **kwargs):
        if fuse is None:
            fuse = self.__class__.default_fuse

        self.eval()
        device = embeds.device
        bsz, embed_dim = embeds.shape
        ensemble_size = self.config.ensemble_size
        max_seq_len = self.config.max_position_embeddings - 1
        if embed_dim != self.config.hidden_size:
            raise ValueError(
                f"Dimension of `embeds` must match the model's dimension."
                f" embeds.shape = {embeds.shape}"
            )
        if from_projected:
            embeds = self.Deproject(embeds)

        states = [
            torch.empty([bsz, 0, embed_dim], device=device)
            for _ in range(ensemble_size)
        ]
        final_states = []
        context = None
        for i in range(max_seq_len):
            last_states, context = self.InferenceStep(embeds, states, context)

            # Take average for final states
            average_last_state = self.average(last_states, ensemble_size)
            final_states.append(average_last_state)

            # Prepare for next token
            if fuse != 0:
                last_states = [
                    (1.0 - fuse) * last_i + fuse * average_last_state
                    for last_i in last_states
                ]
            next_states = [
                torch.cat([s, ls], dim=1).clone() for s, ls in zip(states, last_states)
            ]
            self.OffloadTensor(*states)
            self.OffloadTensor(*last_states)

            states = next_states

        sos_embeds = self.GetSOSEmb(bsz, device)
        # clone() is needed here to discard intermediate computation results.
        result = torch.cat([sos_embeds] + final_states, dim=1).clone()
        self.OffloadTensor(sos_embeds)
        self.OffloadTensor(*states)
        self.OffloadTensor(*final_states)
        return result

    def InferenceStep(self, embeds, states, context):
        output = self(embeds, states)
        # clone() is needed here to discard intermediate computation results.
        last_states = [s[:, -1:, :].clone() for s in output]
        self.OffloadTensor(*output)
        return last_states, context


#
# -- LSTM model --
#
class CLIPTextDeprojectorLSTMConfig(CLIPTextConfig):
    model_type = "clip_text_deprojector_lstm_model"

    default_ensemble_size = 1
    default_use_model3 = False
    default_num_layers = 2
    default_add_embed_once = False
    default_add_embed_at = -1
    default_use_x_norm = True
    default_combined_x_norm = False
    default_x_residual = True
    default_h0_residual = False
    default_h0_excl_residual = False
    default_x_out_residual = False
    default_states_proj = False
    default_out_proj = False
    default_vicinity_mode = False
    default_additional_mlp = False

    default_lstm_hidden_size = 768 * 0
    default_use_mlp = False

    default_use_model2 = False
    default_lstm_hidden_with_residual = False
    default_mlp_intermediate_dim = 0
    default_use_c_norm = True
    default_use_h_norm = True
    default_use_global_state_norm = False

    default_use_extra = False
    default_use_extra_scale = False
    default_use_extra_norm = False
    default_use_extra_linear = True
    default_use_hidden_0 = True
    default_use_context_0 = False
    default_use_state_0 = False
    default_use_sos = False
    default_use_out_norm = False
    default_use_out_scale = False
    default_use_out_proj = False
    default_use_out_extended_proj = False
    default_use_attention = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ensemble_size = kwargs.get(
            "ensemble_size", self.__class__.default_ensemble_size
        )
        self.use_model3 = kwargs.get("use_model3", self.__class__.default_use_model3)
        self.num_layers = kwargs.get("num_layers", self.__class__.default_num_layers)
        self.add_embed_once = kwargs.get(
            "add_embed_once", self.__class__.default_add_embed_once
        )
        self.add_embed_at = kwargs.get(
            "add_embed_at", self.__class__.default_add_embed_at
        )
        self.use_x_norm = kwargs.get("use_x_norm", self.__class__.default_use_x_norm)
        self.combined_x_norm = kwargs.get(
            "combined_x_norm", self.__class__.default_combined_x_norm
        )
        self.x_residual = kwargs.get("x_residual", self.__class__.default_x_residual)
        self.h0_residual = kwargs.get("h0_residual", self.__class__.default_h0_residual)
        self.h0_excl_residual = kwargs.get(
            "h0_excl_residual", self.__class__.default_h0_excl_residual
        )
        self.x_out_residual = kwargs.get(
            "x_out_residual", self.__class__.default_x_out_residual
        )
        self.states_proj = kwargs.get("states_proj", self.__class__.default_states_proj)
        self.out_proj = kwargs.get("out_proj", self.__class__.default_out_proj)
        self.vicinity_mode = kwargs.get(
            "vicinity_mode", self.__class__.default_vicinity_mode
        )
        self.additional_mlp = kwargs.get(
            "additional_mlp", self.__class__.default_additional_mlp
        )

        self.lstm_hidden_size = kwargs.get(
            "lstm_hidden_size", self.__class__.default_lstm_hidden_size
        )
        self.use_mlp = kwargs.get("use_mlp", self.__class__.default_use_mlp)
        self.mlp_intermediate_dim = kwargs.get(
            "mlp_intermediate_dim", self.__class__.default_mlp_intermediate_dim
        )

        self.use_model2 = kwargs.get("use_model2", self.__class__.default_use_model2)
        self.lstm_hidden_with_residual = kwargs.get(
            "lstm_hidden_with_residual",
            self.__class__.default_lstm_hidden_with_residual,
        )
        self.use_c_norm = kwargs.get("use_c_norm", self.__class__.default_use_c_norm)
        self.use_h_norm = kwargs.get("use_h_norm", self.__class__.default_use_h_norm)
        self.use_global_state_norm = kwargs.get(
            "use_global_state_norm", self.__class__.default_use_global_state_norm
        )

        self.use_extra = kwargs.get("use_extra", self.__class__.default_use_extra)
        self.use_extra_scale = kwargs.get(
            "use_extra_scale", self.__class__.default_use_extra_scale
        )
        self.use_extra_norm = kwargs.get(
            "use_extra_norm", self.__class__.default_use_extra_norm
        )
        self.use_extra_linear = kwargs.get(
            "use_extra_linear", self.__class__.default_use_extra_linear
        )
        self.use_hidden_0 = kwargs.get(
            "use_hidden_0", self.__class__.default_use_hidden_0
        )
        self.use_context_0 = kwargs.get(
            "use_context_0", self.__class__.default_use_context_0
        )
        self.use_state_0 = kwargs.get("use_state_0", self.__class__.default_use_state_0)
        self.use_sos = kwargs.get("use_sos", self.__class__.default_use_sos)
        self.use_out_norm = kwargs.get(
            "use_out_norm", self.__class__.default_use_out_norm
        )
        self.use_out_scale = kwargs.get(
            "use_out_scale", self.__class__.default_use_out_scale
        )
        self.use_out_proj = kwargs.get(
            "use_out_proj", self.__class__.default_use_out_proj
        )
        self.use_out_extended_proj = kwargs.get(
            "use_out_extended_proj", self.__class__.default_use_out_extended_proj
        )
        self.use_attention = kwargs.get(
            "use_attention", self.__class__.default_use_attention
        )


class CLIPTextDeprojectorLSTMModel3Layer(CLIPPreTrainedModel):
    config_class = CLIPTextDeprojectorLSTMConfig

    def __init__(self, config: CLIPTextDeprojectorLSTMConfig, index: int):
        super().__init__(config)
        self.index = index
        self.embed_dim = config.hidden_size
        embed_dim = self.embed_dim
        hidden_dim = config.lstm_hidden_size
        self.use_proj = (hidden_dim > 0) and (not config.vicinity_mode)
        if hidden_dim == 0:
            hidden_dim = embed_dim
        self.long_residual = self.use_proj and (not config.out_proj)
        self.states_proj = self.long_residual and config.states_proj

        if self.use_proj:
            self.h0_linear = nn.Linear(embed_dim, hidden_dim)

        if self.states_proj:
            self.states_linear = nn.Linear(embed_dim, embed_dim)

        if config.vicinity_mode:
            input_dim = embed_dim * 2
        else:
            input_dim = embed_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        if config.use_x_norm:
            if config.combined_x_norm:
                self.x_norm = nn.LayerNorm(input_dim, eps=config.layer_norm_eps)
            else:
                self.x_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.c_norm = nn.LayerNorm(hidden_dim, eps=config.layer_norm_eps)
        self.h_norm = nn.LayerNorm(hidden_dim, eps=config.layer_norm_eps)

        if config.vicinity_mode:
            self.lstm_linear = nn.Linear(hidden_dim, embed_dim)
            self.vicinity_linear = nn.Linear(embed_dim, embed_dim)
        elif config.out_proj:
            self.x_out_linear = nn.Linear(hidden_dim, embed_dim)
            self.lstm_linear = nn.Linear(embed_dim, embed_dim)
        else:
            self.lstm_linear = nn.Linear(hidden_dim, embed_dim)

        add_embed_at = config.add_embed_at
        if add_embed_at < 0:
            add_embed_at += config.num_layers
        self.apply_embed_linear = (not config.add_embed_once) or (index == add_embed_at)
        if self.apply_embed_linear:
            self.embed_linear = nn.Linear(embed_dim, embed_dim)

        self.additional_mlp = config.additional_mlp and (
            config.mlp_intermediate_dim > 0
        )
        if self.additional_mlp:
            if config.use_mlp:
                raise NotImplementedError("additional_mlp with use_mlp")

            self.mlp_linear1 = nn.Linear(embed_dim, config.mlp_intermediate_dim)
            self.mlp_linear2 = nn.Linear(config.mlp_intermediate_dim, embed_dim)
            self.mlp_act = ACT2FN[config.hidden_act]

        if config.use_mlp:
            if config.vicinity_mode:
                raise NotImplementedError("mlp with vicinity_mode")
            if not self.apply_embed_linear:
                raise NotImplementedError("without embed linear")

            if self.states_proj:
                mlp_dim = embed_dim * 3
            else:
                mlp_dim = embed_dim * 2
            self.mlp_linear = nn.Linear(mlp_dim, embed_dim)
            self.mlp_act = ACT2FN[config.hidden_act]

    def ZerosForLSTM(self, embeds, hidden_dim):
        bsz = embeds.shape[0]
        return torch.zeros(
            [bsz, hidden_dim],
            dtype=embeds.dtype,
            layout=embeds.layout,
            device=embeds.device,
        )

    def H0(self, embeds):
        if self.config.vicinity_mode:
            return self.ZerosForLSTM(embeds, self.config.lstm_hidden_size)
        if self.use_proj:
            return self.h0_linear(embeds)
        return embeds

    def C0(self, embeds):
        if self.config.vicinity_mode:
            return self.ZerosForLSTM(embeds, self.config.lstm_hidden_size)
        if self.use_proj:
            return self.ZerosForLSTM(embeds, self.config.lstm_hidden_size)
        return torch.zeros_like(embeds)

    def ApplyLSTM(self, embeds, state_i, h_c_h0):
        h, c, h0 = h_c_h0
        x = state_i
        x_residual = x

        if self.config.use_x_norm and (not self.config.combined_x_norm):
            x = self.x_norm(x)
        if self.config.vicinity_mode:
            x = torch.cat([x, embeds], dim=1)
        if self.config.use_x_norm and (self.config.combined_x_norm):
            x = self.x_norm(x)

        h, c = self.lstm(x, (h, c))
        c = self.c_norm(c)
        h = self.h_norm(h)

        if self.config.x_residual:
            h = h + x_residual
        if self.config.h0_residual:
            h = h + h0
        o = h
        if self.config.out_proj:
            o = self.x_out_linear(o)
        if self.config.h0_excl_residual:
            h = h + h0
        if self.config.x_out_residual:
            o = o + x_residual
        return o.unsqueeze(1), (h, c, h0)

    def forward(self, embeds, states, h_c_h0):
        if h_c_h0 is None:
            h0 = self.H0(embeds)
            h_c_h0 = (h0, self.C0(embeds), h0)

        if self.config.vicinity_mode:
            vicinity = states

        if self.long_residual and (not self.config.vicinity_mode):
            residual = states

        seq_len = states.shape[1]
        if seq_len == 1:
            states, h_c_h0 = self.ApplyLSTM(embeds, states[:, 0, :], h_c_h0)
        else:
            next_states = []
            for i in range(states.shape[1]):
                state_i, h_c_h0 = self.ApplyLSTM(embeds, states[:, i, :], h_c_h0)
                next_states.append(state_i)
            states = torch.cat(next_states, dim=1)

        if not self.config.use_mlp:
            if self.config.vicinity_mode:
                if self.embed_dim == self.config.lstm_hidden_size:
                    states = self.lstm_linear(states) + states
                else:
                    states = self.lstm_linear(states)
                states = states + self.vicinity_linear(vicinity) + vicinity
            else:
                if self.states_proj:
                    residual = self.states_linear(residual)
                elif not self.long_residual:
                    residual = states
                states = self.lstm_linear(states) + residual

            if self.apply_embed_linear:
                embeds = self.embed_linear(embeds) + embeds
                embeds = embeds.unsqueeze(1).repeat([1, seq_len, 1])
                states = states + embeds

            if self.additional_mlp:
                residual = states
                states = self.mlp_linear1(states)
                states = self.mlp_act(states)
                states = self.mlp_linear2(states)
                states = states + residual

            return states, h_c_h0

        # if use_mlp
        embeds = self.embed_linear(embeds) + embeds
        embeds = embeds.unsqueeze(1).repeat([1, seq_len, 1])
        mlp_states = [embeds]

        if self.states_proj:
            mlp_states += [self.lstm_linear(states), self.states_linear(residual)]
        else:
            if not self.long_residual:
                residual = states
            mlp_states += [self.lstm_linear(states)]

        mlp_states = torch.cat(mlp_states, dim=2)
        mlp_states = self.mlp_act(mlp_states)
        states = self.mlp_linear(mlp_states) + residual

        return states, h_c_h0


class CLIPTextDeprojectorLSTMModel3(CLIPPreTrainedModel):
    config_class = CLIPTextDeprojectorLSTMConfig
    freeze_final_norm = True

    def __init__(self, config: CLIPTextDeprojectorLSTMConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        embed_dim = self.embed_dim

        self.layers = nn.ModuleList(
            CLIPTextDeprojectorLSTMModel3Layer(config, i)
            for i in range(config.num_layers)
        )

        self.final_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        if self.__class__.freeze_final_norm:
            for p in self.final_norm.parameters():
                p.requires_grad = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def InitWeights(self):
        pass

    def MayAddWeight(self, k, v, weights, prefix):
        if k.startswith(prefix):
            weights[k[len(prefix) :]] = v

    def LoadWeights(self, weights):
        final_norm_weights = {}
        for k, v in weights.items():
            self.MayAddWeight(k, v, final_norm_weights, "text_model.final_layer_norm.")
        self.final_norm.load_state_dict(final_norm_weights)

    def forward(self, embeds, states, first_state, last_state, cont):
        if cont is None:
            cont = [None] * self.config.num_layers

        if last_state is None:
            # for whole states
            states = torch.cat([first_state, states], dim=1)
        else:
            # only for last_state
            states = last_state

        for i, (layer, h_c) in enumerate(zip(self.layers, cont)):
            states, h_c = layer(embeds, states, h_c)
            cont[i] = h_c

        return self.final_norm(states), cont


class CLIPTextDeprojectorLSTMModel2(CLIPPreTrainedModel):
    config_class = CLIPTextDeprojectorLSTMConfig
    freeze_final_norm = True

    def __init__(self, config: CLIPTextDeprojectorLSTMConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        embed_dim = self.embed_dim
        hidden_dim = config.lstm_hidden_size
        self.use_proj = hidden_dim > 0
        if not self.use_proj:
            hidden_dim = embed_dim

        if config.use_global_state_norm:
            self.global_state_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        if config.use_c_norm:
            self.c_norm = nn.LayerNorm(hidden_dim, eps=config.layer_norm_eps)
        if config.use_h_norm:
            self.h_norm = nn.LayerNorm(hidden_dim, eps=config.layer_norm_eps)

        if self.use_proj:
            self.lstm_in_linear = nn.Linear(embed_dim, hidden_dim)
        self.embed_linear = nn.Linear(embed_dim, embed_dim)
        self.lstm_linear = nn.Linear(hidden_dim, embed_dim)

        if config.mlp_intermediate_dim > 0:
            self.mlp_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            self.mlp_fc1 = nn.Linear(embeds, config.mlp_intermediate_dim)
            self.mlp_fc2 = nn.Linear(config.mlp_intermediate_dim, embed_dim)
            self.mlp_act = ACT2FN[config.hidden_act]

        self.final_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        if self.__class__.freeze_final_norm:
            for p in self.final_norm.parameters():
                p.requires_grad = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def InitWeights(self):
        pass

    def MayAddWeight(self, k, v, weights, prefix):
        if k.startswith(prefix):
            weights[k[len(prefix) :]] = v

    def LoadWeights(self, weights):
        final_norm_weights = {}
        for k, v in weights.items():
            self.MayAddWeight(k, v, final_norm_weights, "text_model.final_layer_norm.")
        self.final_norm.load_state_dict(final_norm_weights)

    def ApplyMLP(self, states):
        states = self.mlp_norm(states)
        states = self.mlp_fc1(states)
        states = self.mlp_act(states)
        states = self.mlp_fc2(states)
        return states

    def ApplyLSTM(self, state_i, h_c):
        residual = state_i

        hidden, context = self.lstm(state_i, h_c)
        if self.config.use_c_norm:
            context = self.c_norm(context)
        if self.config.use_h_norm:
            hidden = self.h_norm(hidden)

        if self.use_proj:
            next_state = hidden.unsqueeze(1)
            if self.config.lstm_hidden_with_residual:
                hidden = hidden + self.lstm_in_linear(residual)
            return next_state, (hidden, context)

        state_i = hidden + residual
        if self.config.lstm_hidden_with_residual:
            h_c = (state_i, context)
        else:
            h_c = (hidden, context)

        return state_i.unsqueeze(1), h_c

    def InitializeHandC(self, embeds):
        device, dtype = embeds.device, embeds.dtype
        bsz, embed_dim = embeds.shape
        if self.use_proj:
            hidden_dim = self.config.lstm_hidden_size
        else:
            hidden_dim = embed_dim

        context = torch.zeros([bsz, hidden_dim], dtype=dtype, device=device)

        if self.use_proj:
            return self.lstm_in_linear(embeds), context
        return embeds, context

    def forward(self, embeds, states, first_state, last_state, h_c):
        if h_c is None:
            h_c = self.InitializeHandC(embeds)

        if last_state is None:
            # for whole states
            states = torch.cat([first_state, states], dim=1)
            if self.config.use_global_state_norm:
                states = self.global_state_norm(states)
            residual = states
            next_states = []
            for i in range(states.shape[1]):
                state_i, h_c = self.ApplyLSTM(states[:, i, :], h_c)
                next_states.append(state_i)
            states = torch.cat(next_states, dim=1)
        else:
            # only for last_state
            states = last_state
            if self.config.use_global_state_norm:
                states = self.global_state_norm(states)
            residual = states
            states, h_c = self.ApplyLSTM(states[:, 0, :], h_c)

        if not self.use_proj:
            residual = states

        embeds = self.embed_linear(embeds) + embeds
        embeds = embeds.unsqueeze(1).repeat([1, states.shape[1], 1])
        states = self.lstm_linear(states) + residual
        states = states + embeds

        if self.config.mlp_intermediate_dim > 0:
            states = self.ApplyMLP(states) + states

        return self.final_norm(states), h_c


class CLIPTextDeprojectorLSTMModel(CLIPPreTrainedModel):
    config_class = CLIPTextDeprojectorLSTMConfig
    freeze_attn = False
    freeze_mlp = False
    freeze_final_norm = False

    def __init__(self, config: CLIPTextDeprojectorLSTMConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        embed_dim = self.embed_dim
        hidden_dim = config.lstm_hidden_size
        max_seq_len = config.max_position_embeddings

        if embed_dim == hidden_dim:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim, proj_size=embed_dim, batch_first=True
            )
            if config.use_context_0:
                self.context_proj = nn.Linear(embed_dim, hidden_dim)
        if config.use_out_scale:
            self.out_scale = nn.Parameter(torch.tensor(1.0))
        elif config.use_out_extended_proj:
            self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        elif config.use_out_proj:
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        if config.use_out_norm:
            self.out_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        if config.use_extra:
            if config.use_extra_scale:
                self.extra_scale = nn.Parameter(torch.tensor(1.0))
            if config.use_extra_linear:
                self.embed_proj = nn.Linear(embed_dim, embed_dim)
            if config.use_extra_norm:
                self.embed_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        if config.use_attention:
            # Position_ids is not saved.
            self.position_ids = torch.arange(max_seq_len).expand((1, -1))
            self.position_embedding = nn.Embedding(max_seq_len, self.embed_dim)
            self.norm1 = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            self.attn = CLIPAttention(config)
            if self.__class__.freeze_attn:
                for p in self.attn.parameters():
                    p.requires_grad = False
        if config.use_mlp:
            self.norm2 = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            self.mlp = CLIPMLP(config)
            if self.__class__.freeze_mlp:
                for p in self.mlp.parameters():
                    p.requires_grad = False
        self.final_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        if self.__class__.freeze_final_norm:
            for p in self.final_norm.parameters():
                p.requires_grad = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.config.use_attention:
            self.position_ids = self.position_ids.to(*args, **kwargs)
        return self

    def InitWeights(self):
        pass

    def MayAddWeight(self, k, v, weights, prefix):
        if k.startswith(prefix):
            weights[k[len(prefix) :]] = v

    def LoadWeights(self, weights):
        norm1_weights = {}
        attn_weights = {}
        norm2_weights = {}
        mlp_weights = {}
        final_norm_weights = {}
        for k, v in weights.items():
            self.MayAddWeight(
                k, v, norm1_weights, "text_model.encoder.layers.11.layer_norm1."
            )
            self.MayAddWeight(
                k, v, attn_weights, "text_model.encoder.layers.11.self_attn."
            )
            self.MayAddWeight(
                k, v, norm2_weights, "text_model.encoder.layers.11.layer_norm2."
            )
            self.MayAddWeight(k, v, mlp_weights, "text_model.encoder.layers.11.mlp.")
            self.MayAddWeight(k, v, final_norm_weights, "text_model.final_layer_norm.")

        if self.config.use_attention:
            self.position_embedding.data = weights[
                "text_model.embeddings.position_embedding.weight"
            ]
            self.norm1.load_state_dict(norm1_weights)
            self.attn.load_state_dict(attn_weights)
        if self.config.use_mlp:
            self.norm2.load_state_dict(norm2_weights)
            self.mlp.load_state_dict(mlp_weights)
        self.final_norm.load_state_dict(final_norm_weights)

    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        self,
        bsz,
        tgt_len,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        # shape = (bsz, 1, tgt_len, tgt_len)
        # ex) shape = (2, 1, 4, 4)
        # [[[[0, min, min, min],
        #    [0,   0, min, min],
        #    [0,   0,   0, min],
        #    [0,   0,   0,   0]],
        #   [[0, min, min, min],
        #    [0,   0, min, min],
        #    [0,   0,   0, min],
        #    [0,   0,   0,   0]]]]
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            zero_pads = torch.zeros(
                tgt_len, past_key_values_length, dtype=dtype, device=device
            )
            mask = torch.cat([zero_pads, mask], dim=-1)
        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    def ApplyAttention(self, states):
        device, dtype = states.device, states.dtype
        bsz, seq_len, _ = states.shape

        mask = None
        causal_mask = self._make_causal_mask(bsz, seq_len, dtype, device)
        states = self.attn(states, mask, causal_mask)[0]
        return states

    def ApplyLSTM(self, embeds, states, h_c):
        residual = states
        states, h_c = self.lstm(states, h_c)
        if self.config.use_out_scale:
            states = self.out_scale * states
        elif self.config.use_out_extended_proj:
            states = self.out_proj(torch.cat([states, residual], dim=2))
        elif self.config.use_out_proj:
            states = self.out_proj(states)
        if self.config.use_out_norm:
            states = self.out_norm(states)
        states = states + residual

        if self.config.use_extra:
            embeds_output = embeds.unsqueeze(1)
            if self.config.use_extra_linear:
                embeds_output = embeds_output + self.embed_proj(embeds_output)
            if self.config.use_extra_norm:
                embeds_output = self.embed_norm(embeds_output)
            if self.config.use_extra_scale:
                states = self.extra_scale * states
            states = states + embeds_output

        return states, h_c

    def InitializeHandC(self, embeds):
        device, dtype = embeds.device, embeds.dtype
        bsz, embed_dim = embeds.shape
        hidden_dim = self.config.lstm_hidden_size

        if self.config.use_hidden_0:
            hidden = embeds.unsqueeze(0)
        else:
            hidden = torch.zeros([1, bsz, embed_dim], dtype=dtype, device=device)

        if self.config.use_context_0:
            context = embeds.unsqueeze(0)
            if embed_dim != hidden_dim:
                context = self.context_proj(context)
        else:
            context = torch.zeros([1, bsz, hidden_dim], dtype=dtype, device=device)

        return hidden, context

    def forward(self, embeds, states, first_state, last_state, h_c):
        states = torch.cat([first_state, states], dim=1)
        if h_c is None:
            h_c = self.InitializeHandC(embeds)

        if last_state is None:
            states, h_c = self.ApplyLSTM(embeds, states, h_c)
        else:
            last_state, h_c = self.ApplyLSTM(embeds, last_state, h_c)
            states = last_state
            # TODO: only works when use_attention is false.
            if self.config.use_attention:
                raise NotImplementedError()

        if self.config.use_attention:
            seq_len = states.shape[1]

            residual = states
            position_embedding = self.position_embedding(self.position_ids[:, :seq_len])
            states = states + position_embedding
            states = self.norm1(states)
            states = self.ApplyAttention(states)
            states = states + residual

        if self.config.use_mlp:
            residual = states
            states = self.norm2(states)
            states = self.mlp(states)
            states = states + residual

        states = self.final_norm(states)
        return states, h_c


class CLIPTextDeprojectorLSTM(CLIPTextDeprojectorEnsemble):
    config_class = CLIPTextDeprojectorLSTMConfig

    incremental = True
    discard_states = False

    def __init__(self, config: CLIPTextDeprojectorLSTMConfig):
        super().__init__(config)
        embed_dim = self.embed_dim

        if config.use_model3:
            model_class = CLIPTextDeprojectorLSTMModel3
        elif config.use_model2:
            model_class = CLIPTextDeprojectorLSTMModel2
        else:
            model_class = CLIPTextDeprojectorLSTMModel
        self.models = nn.ModuleList(
            model_class(config) for _ in range(config.ensemble_size)
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for model in self.models:
            model.to(*args, **kwargs)
        return self

    def _init_weights(self, module):
        super()._init_weights(module)
        if (isinstance(module, CLIPTextDeprojectorLSTMModel)) or (
            isinstance(module, CLIPTextDeprojectorLSTMModel2)
        ):
            module.InitWeights()

    def LoadWeights(self, weights):
        for model in self.models:
            model.LoadWeights(weights)

    def forward(self, embeds, states):
        first_state = self.GetFirstState(embeds)
        return [
            self.ApplyModel(model, embeds, s, first_state)
            for model, s in zip(self.models, states)
        ]

    def ApplyModel(self, model, embeds, states, first_state):
        if not self.__class__.discard_states:
            return model(embeds, states, first_state, None, None)[0]

        bsz, seq_len, embed_dim = states.shape
        output = torch.empty([bsz, 0, embed_dim], device=states.device)
        last_state = first_state
        context = None
        for _ in range(seq_len + 1):
            last_state, context = model(
                embeds, output, first_state, last_state, context
            )
            # last_state = last_state.clone()
            output = torch.cat([output, last_state], dim=1)
        return output

    def InferOnlyLast(self, embeds, states, context):
        first_state = self.GetFirstState(embeds)

        output_states = []
        output_context = []
        for i, model in enumerate(self.models):
            if states[i].shape[1] == 0:
                last_state = first_state
            else:
                last_state = states[i][:, -1:, :]
            s, c = model(
                embeds,
                states[i],
                first_state,
                last_state,
                None if context is None else context[i],
            )
            output_states.append(s)
            output_context.append(c)

        return output_states, output_context

    def GetFirstState(self, embeds):
        device, dtype = embeds.device, embeds.dtype
        bsz, embed_dim = embeds.shape

        if self.config.use_state_0:
            return embeds.unsqueeze(1)
        if self.config.use_sos:
            return self.GetSOSEmb(bsz, device)
        return torch.zeros(bsz, 1, embed_dim, dtype=dtype, device=device)

    def InferenceStep(self, embeds, states, context):
        if self.__class__.incremental:
            last_states, context = self.InferOnlyLast(embeds, states, context)
            return [ls.clone() for ls in last_states], context

        output = self(embeds, states)
        # clone() is needed here to discard intermediate computation results.
        last_states = [s[:, -1:, :].clone() for s in output]
        self.OffloadTensor(*output)
        return last_states, context


#
# -- Multi-layer Vicinity-Transformer model --
#
class CLIPTextDeprojectorMvtConfig(CLIPTextConfig):
    model_type = "clip_text_deprojector_mvt_model"

    default_ensemble_size = 1
    default_vicinity = 2
    default_only_first_vicinity = False
    default_vicinity_before_attention = False
    default_vicinity_res_plus_embeds = False
    default_all_in_vicinity = False
    default_use_vicinity_mlp = False
    default_vicinity_mlp_dim = 768 * 1
    default_vicinity_act = ""
    default_use_vicinity_gate = False
    default_num_vt_layers = 1
    default_mlp_layer_dim = 768 * 4
    default_use_old_norm = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ensemble_size = kwargs.get(
            "ensemble_size", self.__class__.default_ensemble_size
        )
        self.vicinity = kwargs.get("vicinity", self.__class__.default_vicinity)
        self.only_first_vicinity = kwargs.get(
            "only_first_vicinity", self.__class__.default_only_first_vicinity
        )
        self.vicinity_before_attention = kwargs.get(
            "vicinity_before_attention",
            self.__class__.default_vicinity_before_attention,
        )
        self.vicinity_res_plus_embeds = kwargs.get(
            "vicinity_res_plus_embeds", self.__class__.default_vicinity_res_plus_embeds
        )
        self.all_in_vicinity = kwargs.get(
            "all_in_vicinity", self.__class__.default_all_in_vicinity
        )
        self.use_vicinity_mlp = kwargs.get(
            "use_vicinity_mlp", self.__class__.default_use_vicinity_mlp
        )
        self.vicinity_mlp_dim = kwargs.get(
            "vicinity_mlp_dim", self.__class__.default_vicinity_mlp_dim
        )
        self.vicinity_act = kwargs.get(
            "vicinity_act", self.__class__.default_vicinity_act
        )
        self.use_vicinity_gate = kwargs.get(
            "use_vicinity_gate", self.__class__.default_use_vicinity_gate
        )
        self.num_vt_layers = kwargs.get(
            "num_vt_layers", self.__class__.default_num_vt_layers
        )
        self.mlp_layer_dim = kwargs.get(
            "mlp_layer_dim", self.__class__.default_mlp_layer_dim
        )
        self.use_old_norm = kwargs.get(
            "use_old_norm", self.__class__.default_use_old_norm
        )


class CLIPTextDeprojectorMvtBase(CLIPPreTrainedModel):
    config_class = CLIPTextDeprojectorMvtConfig

    def __init__(self, config: CLIPTextDeprojectorMvtConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size

    def Assign(self, p, d):
        if p.shape != d.shape:
            raise ValueError(f"Parameter shape doesn't match: {p.shape} v.s. {d.shape}")
        p.data = d

    def ForceAssign(self, p, d):
        psize = p.shape
        dsize = d.shape
        if psize == dsize:
            p.data = d
            return

        msize = [min(ps, ds) for ps, ds in zip(psize, dsize)]
        if len(psize) == 1:
            p.data[: msize[0]] = d[: msize[0]]
        elif len(psize) == 2:
            p.data[: msize[0], : msize[1]] = d[: msize[0], : msize[1]]
        elif len(psize) == 3:
            p.data[: msize[0], : msize[1], : msize[2]] = d[
                : msize[0], : msize[1], : msize[2]
            ]
        else:
            raise ValueError(f"ForceAssign supports up to 3D parameters. Got {psize}")


class CLIPTextDeprojectorMvtMLP(CLIPTextDeprojectorMvtBase):
    def __init__(self, config: CLIPTextDeprojectorMvtConfig):
        super().__init__(config)
        embed_dim = self.embed_dim

        self.activation_fn = ACT2FN[config.hidden_act]
        self.norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(embed_dim, config.mlp_layer_dim)
        self.linear2 = nn.Linear(config.mlp_layer_dim, embed_dim)

    def InitWeights(self):
        # Same as CLIPMLP
        factor = self.config.initializer_factor
        in_proj_std = (
            (self.embed_dim**-0.5)
            * ((2 * self.config.num_vt_layers) ** -0.5)
            * factor
        )
        fc_std = (2 * self.embed_dim) ** -0.5 * factor
        nn.init.normal_(self.linear1.weight, std=fc_std)
        nn.init.normal_(self.linear2.weight, std=in_proj_std)

    def LoadWeights(self, weights, prefix):
        self.Assign(self.norm.weight, weights[prefix + "layer_norm2.weight"])
        self.Assign(self.norm.bias, weights[prefix + "layer_norm2.bias"])
        self.ForceAssign(self.linear1.weight, weights[prefix + "mlp.fc1.weight"])
        self.ForceAssign(self.linear1.bias, weights[prefix + "mlp.fc1.bias"])
        self.ForceAssign(self.linear2.weight, weights[prefix + "mlp.fc2.weight"])
        self.ForceAssign(self.linear2.bias, weights[prefix + "mlp.fc2.bias"])

    def forward(self, states):
        residual = states
        states = self.norm(states)
        states = self.linear1(states)
        states = self.activation_fn(states)
        states = self.linear2(states)
        return states + residual


class CLIPTextDeprojectorMvtVicinity(CLIPTextDeprojectorMvtBase):
    def __init__(self, config: CLIPTextDeprojectorMvtConfig):
        super().__init__(config)

        linear_input_size = 1 + self.config.vicinity
        if config.all_in_vicinity:
            linear_input_size += 1

        if config.use_vicinity_mlp:
            self.InitMLP(config, linear_input_size)
        else:
            self.InitLinear(config, linear_input_size)

    def InitGate(self, config, output_dim):
        embed_dim = self.embed_dim
        self.gate_linear = nn.Linear(embed_dim * (2 + self.config.vicinity), output_dim)
        self.gate_activation_fn = ACT2FN["sigmoid"]

    def InitLinear(self, config, linear_input_size):
        embed_dim = self.embed_dim
        bias = False

        if config.use_vicinity_gate:
            self.InitGate(config, embed_dim)
            bias = True
        if config.vicinity_act:
            self.activation_fn = ACT2FN[config.vicinity_act]
            bias = True
        self.linear = nn.Linear(embed_dim * linear_input_size, embed_dim, bias=bias)

    def InitMLP(self, config, linear_input_size):
        embed_dim = self.embed_dim
        intermediate_dim = config.vicinity_mlp_dim
        if config.use_vicinity_gate:
            self.InitGate(config, intermediate_dim)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.linear1 = nn.Linear(embed_dim * linear_input_size, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, embed_dim)

    def InitWeights(self):
        # Same as CLIPAttention.out_proj
        factor = self.config.initializer_factor
        fc_std = (self.embed_dim**-0.5) * factor
        if self.config.use_vicinity_gate:
            nn.init.normal_(self.gate_linear.weight, std=fc_std)
        if self.config.use_vicinity_mlp:
            nn.init.normal_(self.linear1.weight, std=fc_std)
            nn.init.normal_(self.linear2.weight, std=fc_std)
        else:
            nn.init.normal_(self.linear.weight, std=fc_std)

    def LoadWeights(self, weights, prefix):
        # Do nothing
        pass

    def forward(self, states, vicinity):
        residual = states

        if self.config.vicinity_res_plus_embeds:
            residual = residual + vicinity[:, :, : self.embed_dim]

        combined_vicinity = None
        if self.config.all_in_vicinity or self.config.use_vicinity_gate:
            combined_vicinity = torch.cat([states, vicinity], dim=2)
        if self.config.all_in_vicinity:
            vicinity = combined_vicinity

        gate = None
        if self.config.use_vicinity_gate:
            gate = self.gate_linear(combined_vicinity)
            gate = self.gate_activation_fn(gate)

        if self.config.use_vicinity_mlp:
            vicinity = self.RunMLP(vicinity, gate)
        else:
            vicinity = self.RunLinear(vicinity, gate)

        return vicinity + residual

    def RunLinear(self, vicinity, gate):
        vicinity = self.linear(vicinity)
        if self.config.vicinity_act:
            vicinity = self.activation_fn(vicinity)
        if gate is not None:
            vicinity = gate * vicinity
        return vicinity

    def RunMLP(self, vicinity, gate):
        vicinity = self.linear1(vicinity)
        vicinity = self.activation_fn(vicinity)
        if gate is not None:
            vicinity = gate * vicinity
        vicinity = self.linear2(vicinity)
        return vicinity

    @classmethod
    def MakeInput(cls, config, embeds, states, zeros, seq_len):
        inputs = [embeds.repeat([1, seq_len, 1])]
        for shift in range(config.vicinity):
            zero_len = shift + 2
            if zero_len > seq_len:
                inputs.append(zeros.repeat([1, seq_len, 1]))
            elif shift == 0:
                inputs.append(torch.cat([zeros.repeat([1, 2, 1]), states], dim=1))
            else:
                inputs.append(
                    torch.cat(
                        [zeros.repeat([1, zero_len, 1]), states[:, :-shift, :]], dim=1
                    )
                )
        return torch.cat(inputs, dim=2)


class CLIPTextDeprojectorMvtLayer(CLIPTextDeprojectorMvtBase):
    freeze_q_proj = False
    freeze_k_proj = False
    freeze_v_proj = False
    freeze_out_proj = False

    freeze_attn = False
    freeze_mlp = False

    freeze_only_last = False

    def __init__(
        self, config: CLIPTextDeprojectorMvtConfig, reverse_idx: int, is_first: bool
    ):
        super().__init__(config)
        self.reverse_idx = reverse_idx
        self.apply_norm = config.use_old_norm or (not is_first)
        self.apply_vicinity = (not config.only_first_vicinity) or is_first
        embed_dim = self.embed_dim

        if self.apply_norm:
            self.norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.attn = CLIPAttention(config)
        if self.apply_vicinity:
            self.vicinity = CLIPTextDeprojectorMvtVicinity(config)
        self.mlp = CLIPTextDeprojectorMvtMLP(config)

        if (self.__class__.freeze_only_last) and (reverse_idx > 0):
            return

        if self.__class__.freeze_q_proj:
            for p in self.attn.q_proj.parameters():
                p.requires_grad = False
        if self.__class__.freeze_k_proj:
            for p in self.attn.k_proj.parameters():
                p.requires_grad = False
        if self.__class__.freeze_v_proj:
            for p in self.attn.v_proj.parameters():
                p.requires_grad = False
        if self.__class__.freeze_out_proj:
            for p in self.attn.out_proj.parameters():
                p.requires_grad = False

        if self.__class__.freeze_attn:
            for p in self.attn.parameters():
                p.requires_grad = False
        if self.__class__.freeze_mlp:
            for p in self.mlp.parameters():
                p.requires_grad = False

    def LoadWeights(self, weights):
        orig_layer_idx = self.config.num_hidden_layers - self.reverse_idx - 1
        prefix = f"text_model.encoder.layers.{orig_layer_idx}."

        if self.apply_vicinity:
            self.vicinity.LoadWeights(weights, prefix)
        self.mlp.LoadWeights(weights, prefix)

        # attn
        self.Assign(
            self.attn.k_proj.weight, weights[prefix + "self_attn.k_proj.weight"]
        )
        self.Assign(self.attn.k_proj.bias, weights[prefix + "self_attn.k_proj.bias"])
        self.Assign(
            self.attn.v_proj.weight, weights[prefix + "self_attn.v_proj.weight"]
        )
        self.Assign(self.attn.v_proj.bias, weights[prefix + "self_attn.v_proj.bias"])
        self.Assign(
            self.attn.q_proj.weight, weights[prefix + "self_attn.q_proj.weight"]
        )
        self.Assign(self.attn.q_proj.bias, weights[prefix + "self_attn.q_proj.bias"])
        self.Assign(
            self.attn.out_proj.weight, weights[prefix + "self_attn.out_proj.weight"]
        )
        self.Assign(
            self.attn.out_proj.bias, weights[prefix + "self_attn.out_proj.bias"]
        )

        # norm
        if self.apply_norm:
            if self.config.use_old_norm:
                if self.reverse_idx == 0:
                    norm_prefix = "text_model.final_layer_norm."
                else:
                    norm_prefix = (
                        f"text_model.encoder.layers.{orig_layer_idx + 1}.layer_norm1."
                    )
            else:
                norm_prefix = f"{prefix}layer_norm1."
            self.Assign(self.norm.weight, weights[norm_prefix + "weight"])
            self.Assign(self.norm.bias, weights[norm_prefix + "bias"])

    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        self,
        bsz,
        tgt_len,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        # shape = (bsz, 1, tgt_len, tgt_len)
        # ex) shape = (2, 1, 4, 4)
        # [[[[0, min, min, min],
        #    [0,   0, min, min],
        #    [0,   0,   0, min],
        #    [0,   0,   0,   0]],
        #   [[0, min, min, min],
        #    [0,   0, min, min],
        #    [0,   0,   0, min],
        #    [0,   0,   0,   0]]]]
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            zero_pads = torch.zeros(
                tgt_len, past_key_values_length, dtype=dtype, device=device
            )
            mask = torch.cat([zero_pads, mask], dim=-1)
        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    def ApplyAttn(self, states):
        device, dtype = states.device, states.dtype
        bsz, seq_len, _ = states.shape

        residual = states
        if (not self.config.use_old_norm) and self.apply_norm:
            states = self.norm(states)
        mask = None
        causal_mask = self._make_causal_mask(bsz, seq_len, dtype, device)
        states = self.attn(states, mask, causal_mask)[0]
        return states + residual

    def forward(self, states, vicinity):
        if self.config.vicinity_before_attention and self.apply_vicinity:
            states = self.vicinity(states, vicinity).clone()
        states = self.ApplyAttn(states)
        if not self.config.vicinity_before_attention and self.apply_vicinity:
            states = self.vicinity(states, vicinity).clone()
        states = self.mlp(states)
        if self.config.use_old_norm and self.apply_norm:
            states = self.norm(states)
        return states.clone()


class CLIPTextDeprojectorMvtModel(CLIPTextDeprojectorMvtBase):
    def __init__(self, config: CLIPTextDeprojectorMvtConfig):
        super().__init__(config)
        max_seq_len = config.max_position_embeddings

        # Position_ids is not saved.
        self.position_ids = torch.arange(max_seq_len).expand((1, -1))

        self.position_embedding = nn.Embedding(max_seq_len, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            CLIPTextDeprojectorMvtLayer(
                config, config.num_vt_layers - idx - 1, idx == 0
            )
            for idx in range(config.num_vt_layers)
        )
        if not config.use_old_norm:
            self.final_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.position_ids = self.position_ids.to(*args, **kwargs)
        return self

    def LoadWeights(self, weights):
        self.position_embedding.data = weights[
            "text_model.embeddings.position_embedding.weight"
        ]
        for layer in self.layers:
            layer.LoadWeights(weights)

        # norm
        orig_layer_idx = self.config.num_hidden_layers - self.config.num_vt_layers
        prefix = f"text_model.encoder.layers.{orig_layer_idx}."
        self.Assign(self.norm.weight, weights[prefix + "layer_norm1.weight"])
        self.Assign(self.norm.bias, weights[prefix + "layer_norm1.bias"])
        if not self.config.use_old_norm:
            prefix = "text_model.final_layer_norm."
            self.Assign(self.final_norm.weight, weights[prefix + "weight"])
            self.Assign(self.final_norm.bias, weights[prefix + "bias"])

    def Initialize(self, embeds, states):
        embeds = embeds.unsqueeze(1)
        all_states = torch.cat([embeds, states], dim=1)
        all_states = self.norm(all_states)
        embeds = all_states[:, :1, :]
        states = all_states[:, 1:, :]
        zeros = torch.zeros_like(embeds)

        input_states = torch.cat([embeds, zeros, states], dim=1)
        position_embedding = self.position_embedding(
            self.position_ids[:, : input_states.shape[1]]
        )
        input_states = input_states + position_embedding

        seq_len = input_states.shape[1]
        vicinity = CLIPTextDeprojectorMvtVicinity.MakeInput(
            self.config, embeds, states, zeros, seq_len
        )

        return input_states, vicinity

    def forward(self, embeds, states):
        states, vicinity = self.Initialize(embeds, states)
        for layer in self.layers:
            states = layer(states, vicinity)
        states = states[:, 1:, :]
        if not self.config.use_old_norm:
            states = self.final_norm(states)
        return states


class CLIPTextDeprojectorMvt(CLIPTextDeprojectorEnsemble):
    config_class = CLIPTextDeprojectorMvtConfig

    def __init__(self, config: CLIPTextDeprojectorMvtConfig):
        super().__init__(config)
        embed_dim = self.embed_dim

        self.models = nn.ModuleList(
            CLIPTextDeprojectorMvtModel(config) for _ in range(config.ensemble_size)
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for model in self.models:
            model.to(*args, **kwargs)
        return self

    def _init_weights(self, module):
        # CLIPPreTrainedModel has _init_weights() for CLIP modules.
        # This is called by model.from_pretrained().
        # You can also call model.init_weights() to invoke this manually.

        # CLIPAttention, LayerNorm and Linear.bias are initialized in the base class.
        super()._init_weights(module)

        if isinstance(module, CLIPTextDeprojectorMvtMLP):
            module.InitWeights()
        elif isinstance(module, CLIPTextDeprojectorMvtVicinity):
            module.InitWeights()

    def LoadWeights(self, weights):
        for model in self.models:
            model.LoadWeights(weights)

    def forward(self, embeds, states):
        return [model(embeds, s) for model, s in zip(self.models, states)]


#
# -- Vicinity-Transformer model --
#
class CLIPTextDeprojectorVTConfig(CLIPTextConfig):
    model_type = "clip_text_deprojector_vt_model"

    default_squashed = False
    default_ensemble_size = 1
    default_relative_to_null = True
    default_apply_norm_first = False
    default_apply_norm_to_all = False
    default_vicinity = 2
    default_use_mlp_first_layer = False
    default_use_same_mlp_first_layer = False
    default_use_individual_first_layer = False
    default_no_first_embed_layer = False
    default_first_layer_activation = False
    default_first_layer_residual_connection = False
    default_first_layer_dim = 768 * 1
    default_first_embed_pre_layer_dim = 0
    default_first_embed_layer_dim = 768 * 1
    default_first_embed_layer_dropout = 0.0
    default_require_grad_in_attention = False
    default_use_full_attention = False
    default_use_regular_attention_residual_connection = False
    default_embeds_in_attention = False
    default_num_mlp_layers = 1
    default_mlp_layer_dim = 768 * 4
    default_mlp_dropout = 0.0
    default_use_residual_connection = True
    default_use_embeds_for_residual_connection = False
    default_use_final_vicinity = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.squashed = kwargs.get("squashed", self.__class__.default_squashed)
        self.ensemble_size = kwargs.get(
            "ensemble_size", self.__class__.default_ensemble_size
        )
        self.relative_to_null = kwargs.get(
            "relative_to_null", self.__class__.default_relative_to_null
        )
        self.apply_norm_first = kwargs.get(
            "apply_norm_first", self.__class__.default_apply_norm_first
        )
        self.apply_norm_to_all = kwargs.get(
            "apply_norm_to_all", self.__class__.default_apply_norm_to_all
        )
        self.vicinity = kwargs.get("vicinity", self.__class__.default_vicinity)
        self.use_mlp_first_layer = kwargs.get(
            "use_mlp_first_layer", self.__class__.default_use_mlp_first_layer
        )
        self.use_same_mlp_first_layer = kwargs.get(
            "use_same_mlp_first_layer", self.__class__.default_use_same_mlp_first_layer
        )
        self.use_individual_first_layer = kwargs.get(
            "use_individual_first_layer",
            self.__class__.default_use_individual_first_layer,
        )
        self.no_first_embed_layer = kwargs.get(
            "no_first_embed_layer", self.__class__.default_no_first_embed_layer
        )
        self.first_layer_activation = kwargs.get(
            "first_layer_activation", self.__class__.default_first_layer_activation
        )
        self.first_layer_residual_connection = kwargs.get(
            "first_layer_residual_connection",
            self.__class__.default_first_layer_residual_connection,
        )
        self.first_layer_dim = kwargs.get(
            "first_layer_dim", self.__class__.default_first_layer_dim
        )
        self.first_embed_pre_layer_dim = kwargs.get(
            "first_embed_pre_layer_dim",
            self.__class__.default_first_embed_pre_layer_dim,
        )
        self.first_embed_layer_dim = kwargs.get(
            "first_embed_layer_dim", self.__class__.default_first_embed_layer_dim
        )
        self.first_embed_layer_dropout = kwargs.get(
            "first_embed_layer_dropout",
            self.__class__.default_first_embed_layer_dropout,
        )
        self.require_grad_in_attention = kwargs.get(
            "require_grad_in_attention",
            self.__class__.default_require_grad_in_attention,
        )
        self.use_full_attention = kwargs.get(
            "use_full_attention", self.__class__.default_use_full_attention
        )
        self.use_regular_attention_residual_connection = kwargs.get(
            "use_regular_attention_residual_connection",
            self.__class__.default_use_regular_attention_residual_connection,
        )
        self.embeds_in_attention = kwargs.get(
            "embeds_in_attention", self.__class__.default_embeds_in_attention
        )
        self.num_mlp_layers = kwargs.get(
            "num_mlp_layers", self.__class__.default_num_mlp_layers
        )
        self.mlp_layer_dim = kwargs.get(
            "mlp_layer_dim", self.__class__.default_mlp_layer_dim
        )
        self.mlp_dropout = kwargs.get("mlp_dropout", self.__class__.default_mlp_dropout)
        self.use_residual_connection = kwargs.get(
            "use_residual_connection", self.__class__.default_use_residual_connection
        )
        self.use_embeds_for_residual_connection = kwargs.get(
            "use_embeds_for_residual_connection",
            self.__class__.default_use_embeds_for_residual_connection,
        )
        self.use_final_vicinity = kwargs.get(
            "use_final_vicinity", self.__class__.default_use_final_vicinity
        )

        if self.squashed and self.use_full_attention:
            raise ValueError(
                "squashed and use_full_attention can't be set True simultaneously."
            )


class CLIPTextDeprojectorVTBase(CLIPPreTrainedModel):
    config_class = CLIPTextDeprojectorVTConfig

    def __init__(self, config: CLIPTextDeprojectorVTConfig):
        super().__init__(config)
        self.config = config

    def _init_weights(self, module):
        super()._init_weights(module)

    def CheckAndInit(self, p, d):
        if p.data.shape != d.shape:
            raise ValueError(
                f"Parameter shape doesn't match: {p.data.shape} v.s. {d.shape}"
            )
        p.data = d

    def ResizeAndInit(self, p, d):
        psize = p.shape
        dsize = d.shape
        if psize == dsize:
            p.data = d
            return

        msize = [min(ps, ds) for ps, ds in zip(psize, dsize)]
        if len(psize) == 1:
            p.data[: msize[0]] = d[: msize[0]]
        elif len(psize) == 2:
            p.data[: msize[0], : msize[1]] = d[: msize[0], : msize[1]]
        elif len(psize) == 3:
            p.data[: msize[0], : msize[1], : msize[2]] = d[
                : msize[0], : msize[1], : msize[2]
            ]
        else:
            raise ValueError(
                f"CheckAndInitWithResize supports only 2-D parameters. Got {psize}"
            )
        return


class CLIPTextDeprojectorVTMLPLayer(CLIPTextDeprojectorVTBase):
    def __init__(self, config: CLIPTextDeprojectorVTConfig):
        super().__init__(config)
        embed_dim = config.hidden_size

        self.activation_fn = ACT2FN[config.hidden_act]
        self.norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(embed_dim, config.mlp_layer_dim)
        self.linear2 = nn.Linear(config.mlp_layer_dim, embed_dim)
        if config.mlp_dropout:
            self.dropout = nn.Dropout(p=config.mlp_dropout)

    def forward(self, states):
        states = self.norm(states)
        states = self.linear1(states)
        if self.config.mlp_dropout:
            states = self.dropout(states)
        states = self.activation_fn(states)
        states = self.linear2(states)
        return states

    def InitFromWeights(self, weights):
        self.CheckAndInit(self.norm.weight, weights["encoder_layer.layer_norm2.weight"])
        self.CheckAndInit(self.norm.bias, weights["encoder_layer.layer_norm2.bias"])
        self.ResizeAndInit(self.linear1.weight, weights["encoder_layer.mlp.fc1.weight"])
        self.ResizeAndInit(self.linear1.bias, weights["encoder_layer.mlp.fc1.bias"])
        self.ResizeAndInit(self.linear2.weight, weights["encoder_layer.mlp.fc2.weight"])
        self.ResizeAndInit(self.linear2.bias, weights["encoder_layer.mlp.fc2.bias"])


class CLIPTextDeprojectorVT(CLIPTextDeprojectorVTBase):
    default_fuse = 0.0
    learn_only_uninitialized = False
    dont_learn_attention = False

    def CreateMLPFirstLayer(self, config, embed_dim, ensemble_size):
        first_layer_dim = config.first_layer_dim
        first_embed_layer_dim = config.first_embed_layer_dim
        use_same_mlp_first_layer = config.use_same_mlp_first_layer
        use_first_embed_layer = not config.no_first_embed_layer

        if (
            use_first_embed_layer
            and first_embed_layer_dim
            and (not use_same_mlp_first_layer)
        ):
            self.linear1_embeds1 = nn.ModuleList(
                nn.Linear(embed_dim, first_embed_layer_dim)
                for _ in range(ensemble_size)
            )
            self.linear1_embeds2 = nn.ModuleList(
                nn.Linear(first_embed_layer_dim, embed_dim)
                for _ in range(ensemble_size)
            )

        if first_layer_dim:
            self.linear1_states1 = nn.ModuleList(
                nn.Linear(embed_dim, first_layer_dim) for _ in range(ensemble_size)
            )
            self.linear1_states2 = nn.ModuleList(
                nn.Linear(first_layer_dim, embed_dim) for _ in range(ensemble_size)
            )

        if use_first_embed_layer:
            return embed_dim * (1 + config.vicinity)
        else:
            return embed_dim * config.vicinity

    def CreateIndividualFirstLayer(self, config, embed_dim, ensemble_size):
        first_layer_dim = config.first_layer_dim
        first_embed_pre_layer_dim = config.first_embed_pre_layer_dim
        first_embed_layer_dim = config.first_embed_layer_dim
        if first_embed_pre_layer_dim:
            self.linear_pre_embeds = nn.ModuleList(
                nn.Linear(embed_dim, first_embed_pre_layer_dim)
                for _ in range(ensemble_size)
            )
            self.linear1_embeds = nn.ModuleList(
                nn.Linear(first_embed_pre_layer_dim, first_embed_layer_dim)
                for _ in range(ensemble_size)
            )
        else:
            self.linear1_embeds = nn.ModuleList(
                nn.Linear(embed_dim, first_embed_layer_dim)
                for _ in range(ensemble_size)
            )
        if config.first_embed_layer_dropout:
            self.linear1_embeds_dropout = nn.Dropout(p=config.first_embed_layer_dropout)
        self.linear1 = nn.ModuleList(
            nn.Linear(embed_dim, first_layer_dim) for _ in range(ensemble_size)
        )
        first_intermediate_dim = (
            first_embed_layer_dim + first_layer_dim * config.vicinity
        )
        return first_intermediate_dim

    def CreateCombinedFirstLayer(self, config, embed_dim, ensemble_size):
        first_layer_dim = config.first_layer_dim
        input_dim = embed_dim * (1 + config.vicinity)
        self.linear1 = nn.ModuleList(
            nn.Linear(input_dim, first_layer_dim) for _ in range(ensemble_size)
        )
        first_intermediate_dim = first_layer_dim
        return first_intermediate_dim

    def __init__(self, config: CLIPTextDeprojectorVTConfig):
        super().__init__(config)
        embed_dim = config.hidden_size
        ensemble_size = config.ensemble_size
        max_seq_len = config.max_position_embeddings - 1
        if config.embeds_in_attention:
            max_seq_len += 1

        # Position_ids is not saved.
        self.position_ids = torch.arange(max_seq_len).expand((1, -1))

        self.register_buffer(
            "deprojection", torch.empty([embed_dim, config.projection_dim])
        )
        self.register_buffer("sos_embed", torch.empty([embed_dim]))
        self.register_buffer("null_embed", torch.empty([embed_dim]))

        # Attention layer
        self.position_embedding = nn.ModuleList(
            nn.Embedding(max_seq_len, embed_dim) for _ in range(ensemble_size)
        )
        self.norm = nn.ModuleList(
            nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            for _ in range(ensemble_size)
        )
        # This is independent of seq_len (aka max_position_embeddings)
        self.self_attn = nn.ModuleList(
            CLIPAttention(config) for _ in range(ensemble_size)
        )
        if (
            self.__class__.learn_only_uninitialized
            or self.__class__.dont_learn_attention
            or (not config.require_grad_in_attention)
        ):
            # Don't learn all parameters
            for p in self.self_attn.parameters():
                p.requires_grad = False
            if self.config.squashed:
                for attn in self.self_attn:
                    attn.out_proj.bias.requires_grad = True
        elif (not config.use_full_attention) and (not self.config.squashed):
            # Don't learn out_proj parameters
            for attn in self.self_attn:
                for p in attn.out_proj.parameters():
                    p.requires_grad = False

        # Vicinity layer (= First layer)
        self.activation_fn = ACT2FN[config.hidden_act]
        if config.use_mlp_first_layer:
            first_intermediate_dim = self.CreateMLPFirstLayer(
                config, embed_dim, ensemble_size
            )
        elif config.use_individual_first_layer:
            first_intermediate_dim = self.CreateIndividualFirstLayer(
                config, embed_dim, ensemble_size
            )
        else:
            first_intermediate_dim = self.CreateCombinedFirstLayer(
                config, embed_dim, ensemble_size
            )

        # Second layer
        if self.config.squashed:
            self.linear2 = nn.ModuleList(
                nn.Linear(first_intermediate_dim, embed_dim, bias=False)
                for _ in range(ensemble_size)
            )
        else:
            self.linear2 = nn.ModuleList(
                nn.Linear(first_intermediate_dim + embed_dim, embed_dim)
                for _ in range(ensemble_size)
            )

        # MLP layers
        self.mlp_layers = nn.ModuleList(
            nn.ModuleList(
                CLIPTextDeprojectorVTMLPLayer(config)
                for _ in range(config.num_mlp_layers)
            )
            for _ in range(ensemble_size)
        )
        if self.__class__.learn_only_uninitialized:
            # Don't learn all parameters
            for p in self.mlp_layers.parameters():
                p.requires_grad = False

        # Final vicinity layer
        if self.config.use_final_vicinity:
            self.final_linear = nn.ModuleList(
                nn.Linear(first_intermediate_dim, embed_dim, bias=False)
                for _ in range(ensemble_size)
            )

        # Final normalizer
        self.final_norm = nn.ModuleList(
            nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
            for _ in range(ensemble_size)
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.position_ids = self.position_ids.to(*args, **kwargs)
        return self

    def _init_weights(self, module):
        # CLIPPreTrainedModel has _init_weights() for CLIP modules.
        # This is called by model.from_pretrained().
        # You can also call model.init_weights() to invoke this manually.

        # LayerNorm and Linear.bias are initialized in the base class.
        super()._init_weights(module)

        # Overwrite out_proj of CLIPAttention
        if isinstance(module, CLIPAttention):
            if not module.config.use_full_attention:
                embed_dim = module.config.hidden_size
                self.CheckAndInit(module.out_proj.weight, torch.eye(embed_dim))
                module.out_proj.bias.data.zero_()

        # Similar to CLIPMLP
        if isinstance(module, CLIPTextDeprojectorVTMLPLayer):
            factor = module.config.initializer_factor
            embed_dim = module.config.hidden_size
            num_mlp_layers = module.config.num_mlp_layers
            # in_proj_std == fc_std if num_mlp_layers == 1
            # in_proj_std = 0.036 * 0.707 * 1.0 = 0.025
            in_proj_std = (embed_dim**-0.5) * ((2 * num_mlp_layers) ** -0.5) * factor
            # fc_std = 0.026 * 1.0 = 0.026
            fc_std = (2 * embed_dim) ** -0.5 * factor

            nn.init.normal_(module.linear1.weight, std=fc_std)
            nn.init.normal_(module.linear2.weight, std=in_proj_std)

        if isinstance(module, CLIPTextDeprojectorVT):
            factor = module.config.initializer_factor
            # first_layer_dim = module.config.first_layer_dim
            embed_dim = module.config.hidden_size
            fc_std = embed_dim**-0.5 * factor
            if module.config.use_mlp_first_layer:
                if module.config.first_embed_layer_dim and (
                    not module.config.use_same_mlp_first_layer
                ):
                    for m in module.linear1_embeds1:
                        nn.init.normal_(m.weight, std=fc_std)
                    for m in module.linear1_embeds2:
                        nn.init.normal_(m.weight, std=fc_std)
                if module.config.first_layer_dim:
                    for m in module.linear1_states1:
                        nn.init.normal_(m.weight, std=fc_std)
                    for m in module.linear1_states2:
                        nn.init.normal_(m.weight, std=fc_std)
            else:
                if module.config.use_individual_first_layer:
                    if module.config.first_embed_pre_layer_dim:
                        for m in module.linear_pre_embeds:
                            torch.nn.init.normal_(m.weight, std=fc_std)
                    for m in module.linear1_embeds:
                        torch.nn.init.normal_(m.weight, std=fc_std)
                for m in module.linear1:
                    torch.nn.init.normal_(m.weight, std=fc_std)
            for m in module.linear2:
                torch.nn.init.normal_(m.weight, std=fc_std)
            if module.config.use_final_vicinity:
                for m in module.final_linear:
                    torch.nn.init.normal_(m.weight, std=fc_std)

    def InitFromWeights(self, weights):
        if self.config.ensemble_size > 1:
            raise ValueError(
                f"ensemble_size must be 1 for training: {self.config.ensemble_size}"
            )

        # Second layer (= linear2)
        if (not self.config.use_full_attention) and (not self.config.squashed):
            self.ResizeAndInit(
                self.linear2[0].weight,
                weights["encoder_layer.self_attn.out_proj.weight"],
            )
            self.CheckAndInit(
                self.linear2[0].bias,
                weights["encoder_layer.self_attn.out_proj.bias"],
            )

        # CLIPAttention
        self.CheckAndInit(
            self.norm[0].weight, weights["encoder_layer.layer_norm1.weight"]
        )
        self.CheckAndInit(self.norm[0].bias, weights["encoder_layer.layer_norm1.bias"])
        self.CheckAndInit(
            self.self_attn[0].q_proj.weight,
            weights["encoder_layer.self_attn.q_proj.weight"],
        )
        self.CheckAndInit(
            self.self_attn[0].q_proj.bias,
            weights["encoder_layer.self_attn.q_proj.bias"],
        )
        self.CheckAndInit(
            self.self_attn[0].k_proj.weight,
            weights["encoder_layer.self_attn.k_proj.weight"],
        )
        self.CheckAndInit(
            self.self_attn[0].k_proj.bias,
            weights["encoder_layer.self_attn.k_proj.bias"],
        )
        self.CheckAndInit(
            self.self_attn[0].v_proj.weight,
            weights["encoder_layer.self_attn.v_proj.weight"],
        )
        self.CheckAndInit(
            self.self_attn[0].v_proj.bias,
            weights["encoder_layer.self_attn.v_proj.bias"],
        )
        if self.config.use_full_attention or self.config.squashed:
            self.CheckAndInit(
                self.self_attn[0].v_proj.weight,
                weights["encoder_layer.self_attn.out_proj.weight"],
            )
            self.CheckAndInit(
                self.self_attn[0].v_proj.bias,
                weights["encoder_layer.self_attn.out_proj.bias"],
            )

        # MLP layers
        for l in self.mlp_layers[0]:
            l.InitFromWeights(weights)

        # Final normalizer
        self.CheckAndInit(self.final_norm[0].weight, weights["final_layer_norm.weight"])
        self.CheckAndInit(self.final_norm[0].bias, weights["final_layer_norm.bias"])

    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        self,
        bsz,
        tgt_len,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        # shape = (bsz, 1, tgt_len, tgt_len)
        # ex) shape = (2, 1, 4, 4)
        # [[[[0, min, min, min],
        #    [0,   0, min, min],
        #    [0,   0,   0, min],
        #    [0,   0,   0,   0]],
        #   [[0, min, min, min],
        #    [0,   0, min, min],
        #    [0,   0,   0, min],
        #    [0,   0,   0,   0]]]]
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            zero_pads = torch.zeros(
                tgt_len, past_key_values_length, dtype=dtype, device=device
            )
            mask = torch.cat([zero_pads, mask], dim=-1)
        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    def MakeVicinityInput(self, embeds, states, seq_len):
        if embeds is None:
            inputs = []
        else:
            inputs = [embeds.repeat([1, seq_len, 1])]
        inputs.append(states)

        zeros = states[:, :1, :]
        for i in range(1, self.config.vicinity):
            if i > seq_len:
                inputs.append(zeros.repeat([1, seq_len, 1]))
            else:
                inputs.append(
                    torch.cat([zeros.repeat([1, i, 1]), states[:, :-i, :]], dim=1)
                )
        return torch.cat(inputs, dim=2)

    def ApplyMLPFirstLayer(self, idx, embeds, states, bsz, seq_len, embed_dim, device):
        first_layer_dim = self.config.first_layer_dim
        first_embed_layer_dim = self.config.first_embed_layer_dim
        use_same_mlp_first_layer = self.config.use_same_mlp_first_layer
        use_first_embed_layer = not self.config.no_first_embed_layer

        if use_same_mlp_first_layer:
            if first_layer_dim:
                all_states = torch.cat([embeds, states], dim=1)
                all_states_output = all_states
                all_states_output = self.linear1_states1[idx](all_states_output)
                all_states_output = self.activation_fn(all_states_output)
                all_states_output = self.linear1_states2[idx](all_states_output)
                all_states_output = all_states_output + all_states
                embeds_output = all_states_output[:, :1, :]
                states_output = all_states_output[:, 1:, :]
            else:
                embeds_output = embeds
                states_output = states
        else:
            embeds_output = embeds
            if first_embed_layer_dim:
                embeds_output = self.linear1_embeds1[idx](embeds_output)
                embeds_output = self.activation_fn(embeds_output)
                embeds_output = self.linear1_embeds2[idx](embeds_output)
                embeds_output = embeds_output + embeds
            states_output = states
            if first_layer_dim:
                states_output = self.linear1_states1[idx](states_output)
                states_output = self.activation_fn(states_output)
                states_output = self.linear1_states2[idx](states_output)
                states_output = states_output + states

        if use_first_embed_layer:
            return self.MakeVicinityInput(embeds_output, states_output, seq_len)
        else:
            return self.MakeVicinityInput(None, states_output, seq_len)

    def ApplyIndividualFirstLayer(
        self, idx, embeds, states, bsz, seq_len, embed_dim, device
    ):
        first_layer_dim = self.config.first_layer_dim

        if self.config.first_embed_pre_layer_dim:
            embeds_output = self.linear_pre_embeds[idx](embeds)
            embeds_output = self.activation_fn(embeds_output)
            embeds_output = self.linear1_embeds[idx](embeds_output)
        else:
            embeds_output = self.linear1_embeds[idx](embeds)

        if self.config.first_embed_layer_dropout:
            embeds_output = self.linear1_embeds_dropout(embeds_output)

        states_output = self.linear1[idx](states)

        # states[0] is a zero tensor.
        zero_states = states[:, :1, :]
        zero_output = states_output[:, :1, :]

        # TODO: make sure first_embed_layer_dim is an integer multiple of embed_dim.
        embeds_dim_mult = self.config.first_embed_layer_dim // embed_dim
        first_input = [
            embeds.repeat([1, seq_len, embeds_dim_mult]),
            states,
        ]
        first_output = [embeds_output.repeat([1, seq_len, 1]), states_output]
        for i in range(1, self.config.vicinity):
            if i > seq_len:
                first_input.append(zero_states.repeat([1, seq_len, 1]))
                first_output.append(zero_output.repeat([1, seq_len, 1]))
            else:
                first_input.append(
                    torch.cat([zero_states.repeat([1, i, 1]), states[:, :-i, :]], dim=1)
                )
                first_output.append(
                    torch.cat(
                        [zero_output.repeat([1, i, 1]), states_output[:, :-i, :]], dim=1
                    )
                )
        first_output = torch.cat(first_output, dim=2)

        if self.config.first_layer_activation:
            first_output = self.activation_fn(first_output)
        if self.config.first_layer_residual_connection:
            first_input = torch.cat(first_input, dim=2)  # = first layer residual
            first_output = first_output + first_input

        return first_output.clone()

    def ApplyCombinedFirstLayer(
        self, idx, embeds, states, bsz, seq_len, embed_dim, device
    ):
        first_layer_dim = self.config.first_layer_dim
        if first_layer_dim > 0:
            first_input = [embeds.repeat([1, seq_len, 1]), states]
            for i in range(1, self.config.vicinity):
                zero_states = torch.zeros([bsz, i, embed_dim], device=device)
                first_input.append(torch.cat([zero_states, states[:, :-i, :]], dim=1))
            first_input = torch.cat(first_input, dim=2)
            first_output = self.linear1[idx](first_input)
            first_output = self.activation_fn(first_output)
        else:
            first_output = torch.empty([bsz, seq_len, 0], device=device)
        return first_output

    def forward(self, idx, embeds, states, **kwargs):
        if self.config.relative_to_null:
            embeds = embeds - self.null_embed
            states = states - self.null_embed
        embeds = embeds.unsqueeze(1)
        device, dtype = embeds.device, embeds.dtype
        bsz, seq_len, embed_dim = states.shape

        # Input Normalization
        if self.config.apply_norm_first:
            if self.config.apply_norm_to_all:
                all_states = torch.cat([embeds, states], dim=1)
                all_states = self.norm[idx](all_states)
                embeds = all_states[:, :1, :]
                states = all_states[:, 1:, :]
            else:
                states = self.norm[idx](states)

        # Expand the hidden states with zeros to the left
        zero_states = torch.zeros([bsz, 1, embed_dim], device=device)
        states = torch.cat([zero_states, states], dim=1)
        seq_len += 1

        # Residual Connection
        if self.config.use_residual_connection:
            if self.config.use_embeds_for_residual_connection:
                residual = embeds.repeat([1, seq_len, 1])
            else:
                residual = states

        # Self Attention
        attn_len = seq_len
        attention_input = states
        if self.config.use_full_attention:
            if self.config.use_regular_attention_residual_connection:
                attention_residual = residual
            else:
                attention_residual = attention_input
        if self.config.embeds_in_attention:
            attention_input = torch.cat([embeds, attention_input], dim=1)
            attn_len += 1
        position_embedding = self.position_embedding[idx](
            self.position_ids[:, :attn_len]
        )
        attention_input = attention_input + position_embedding
        if not self.config.apply_norm_first:
            # Not normalized yet
            attention_input = self.norm[idx](attention_input)
        attention_mask = None
        causal_attention_mask = self._make_causal_mask(bsz, attn_len, dtype, device)
        attention_output = self.self_attn[idx](
            attention_input, attention_mask, causal_attention_mask
        )[0]
        if self.config.embeds_in_attention:
            attention_output = attention_output[:, 1:, :]
        if self.config.use_full_attention:
            attention_output = attention_output + attention_residual
            if self.config.use_regular_attention_residual_connection:
                residual = attention_output

        # Vicinity (= First Layer, linear1)
        if self.config.use_mlp_first_layer:
            vicinity_output = self.ApplyMLPFirstLayer(
                idx, embeds, states, bsz, seq_len, embed_dim, device
            )
        elif self.config.use_individual_first_layer:
            vicinity_output = self.ApplyIndividualFirstLayer(
                idx, embeds, states, bsz, seq_len, embed_dim, device
            )
        else:
            vicinity_output = self.ApplyCombinedFirstLayer(
                idx, embeds, states, bsz, seq_len, embed_dim, device
            )

        # Second Layer (= linear2)
        if self.config.squashed:
            states = attention_output + self.linear2[idx](vicinity_output)
        else:
            states = self.linear2[idx](
                torch.cat([attention_output, vicinity_output], dim=2)
            )
        if self.config.use_residual_connection:
            states = (states + residual).clone()

        # MLP Layers
        for layer in self.mlp_layers[idx]:
            if self.config.use_residual_connection:
                residual = states
            states = layer(states)
            if self.config.use_residual_connection:
                states = states + residual

        # Final vicinity layer
        if self.config.use_final_vicinity:
            states = states + self.final_linear[idx](
                self.MakeVicinityInput(embeds, states, seq_len)
            )

        # Final Norm
        return self.final_norm[idx](states)

    def OffloadTensor(self, *ts):
        for t in ts:
            t.to("cpu")
        torch.cuda.empty_cache()

    def Deproject(self, embeds):
        return nn.functional.linear(embeds, self.deprojection, None)

    def RunForTraining(self, embeds, final_states, from_projected=False, **kwargs):
        if self.config.ensemble_size > 1:
            raise ValueError(
                f"ensemble_size must be 1 for training: {self.config.ensemble_size}"
            )
        bsz, embed_dim = embeds.shape
        if embed_dim != self.config.hidden_size:
            raise ValueError(
                f"Dimension of `embeds` must match the model's dimension."
                f" embeds.shape = {embeds.shape}"
            )
        if from_projected:
            embeds = self.Deproject(embeds)
        states = self(0, embeds, final_states[:, 1:-1, :])

        sos_embeds = self.GetSOSEmb(bsz)
        result = torch.cat([sos_embeds, states], dim=1).clone()
        self.OffloadTensor(sos_embeds, states)
        return result

    def Inference(self, embeds, from_projected=False, fuse=None, **kwargs):
        if fuse is None:
            fuse = self.__class__.default_fuse

        self.eval()
        device = embeds.device
        bsz, embed_dim = embeds.shape
        ensemble_size = self.config.ensemble_size
        max_seq_len = self.config.max_position_embeddings - 1
        if embed_dim != self.config.hidden_size:
            raise ValueError(
                f"Dimension of `embeds` must match the model's dimension."
                f" embeds.shape = {embeds.shape}"
            )
        if from_projected:
            embeds = nn.functional.linear(embeds, self.deprojection, None)

        states = [
            torch.empty([bsz, 0, embed_dim], device=device)
            for _ in range(ensemble_size)
        ]
        final_states = []
        for i in range(max_seq_len):
            output = [self(idx, embeds, states[idx]) for idx in range(ensemble_size)]
            # clone() is needed here to discard intermediate computation results.
            last_states = [s[:, -1:, :].clone() for s in output]
            self.OffloadTensor(*output)

            # Take average for final states
            average_last_state = self.average(last_states, ensemble_size)
            final_states.append(average_last_state)

            # Prepare for next token
            if fuse != 0:
                last_states = [
                    (1.0 - fuse) * last_i + fuse * average_last_state
                    for last_i in last_states
                ]
            next_states = [
                torch.cat([s, ls], dim=1) for s, ls in zip(states, last_states)
            ]
            self.OffloadTensor(*states)
            self.OffloadTensor(*last_states)

            states = next_states

        sos_embeds = self.GetSOSEmb(bsz)
        # clone() is needed here to discard intermediate computation results.
        result = torch.cat([sos_embeds] + final_states, dim=1).clone()
        self.OffloadTensor(sos_embeds)
        self.OffloadTensor(*states)
        self.OffloadTensor(*final_states)
        return result

    def GetSOSEmb(self, bsz):
        return self.sos_embed.view([1, 1, -1]).repeat([bsz, 1, 1])

    def average(self, ls, size):
        if size == 1:
            return ls[0]
        else:
            return sum(ls) / size

    def Merge(self, vicinity=False, attention=False, mlp=False):
        def avg(ls):
            state_dict = ls[0].state_dict()
            for key in state_dict:
                state_sum = sum(m.state_dict()[key] for m in ls)
                state_dict[key] = state_sum / len(ls)
            for m in ls:
                m.load_state_dict(state_dict)

        if vicinity:
            if self.config.use_mlp_first_layer:
                if not self.config.use_same_mlp_first_layer:
                    avg(self.linear1_embeds1)
                    avg(self.linear1_embeds2)
                avg(self.linear1_states1)
                avg(self.linear1_states2)
            else:
                if self.config.use_individual_first_layer:
                    if self.config.first_embed_pre_layer_dim:
                        avg(self.linear_pre_embeds)
                    avg(self.linear1_embeds)
                avg(self.linear1)
            avg(self.linear2)
        if attention:
            avg(self.position_embedding)
            avg(self.norm)
            avg(self.self_attn)
        if mlp:
            avg(self.mlp_layers)
            avg(self.final_linear)
            avg(self.final_norm)
        return self

    @classmethod
    def from_units(cls, models):
        if not models:
            raise ValueError(f"At least one model must be provided.")
        if any(model.config.ensemble_size > 1 for model in models):
            raise ValueError(f"The length of all `models` must be 1.")
        new_config = copy.deepcopy(models[0].config)
        new_config.ensemble_size = len(models)

        new_model = cls(new_config)
        new_model.deprojection = models[0].deprojection
        new_model.sos_embed = models[0].sos_embed
        new_model.null_embed = models[0].null_embed

        def copy_params(cls, a, b):
            for pa, pb in zip(a.parameters(), b.parameters()):
                pa.data = pb.data

        for i, model in enumerate(models):
            if new_config.use_mlp_first_layer:
                if not new_config.use_same_mlp_first_layer:
                    copy_params(new_model1.linear1_embeds1[i], model.linear1_embeds1[0])
                    copy_params(new_model1.linear1_embeds2[i], model.linear1_embeds2[0])
                copy_params(new_model1.linear1_states1[i], model.linear1_states1[0])
                copy_params(new_model1.linear1_states2[i], model.linear1_states2[0])
            else:
                if new_config.use_individual_first_layer:
                    if new_config.first_embed_pre_layer_dim:
                        copy_params(
                            new_model.linear_pre_embeds[i], model.linear_pre_embeds[0]
                        )
                    copy_params(new_model.linear1_embeds[i], model.linear1_embeds[0])
                copy_params(new_model.linear1[i], model.linear1[0])
            copy_params(new_model.linear2[i], model.linear2[0])
            copy_params(new_model.position_embedding[i], model.position_embedding[0])
            copy_params(new_model.norm[i], model.norm[0])
            copy_params(new_model.self_attn[i], model.self_attn[0])
            for j in range(new_config.num_mlp_layers):
                copy_params(new_model.mlp_layers[i][j], model.mlp_layers[0][j])
            if new_config.use_final_vicinity:
                copy_params(new_model.final_linear[i], model.final_linear[0])
            copy_params(new_model.final_norm[i], model.final_norm[0])

        return new_model

    @classmethod
    def merge_units(cls, models):
        if not models:
            raise ValueError(f"At least one model must be provided.")
        if any(model.config.ensemble_size > 1 for model in models):
            raise ValueError(f"The length of all `models` must be 1.")
        model = cls(models[0].config)
        state_dict = model.state_dict()
        for key in state_dict:
            state_sum = sum(m.state_dict()[key] for m in models)
            state_dict[key] = state_sum / len(models)
        model.load_state_dict(state_dict)
        return model


#
# -- Vicinity model --
#
class CLIPTextDeprojectorVConfig(PretrainedConfig):
    model_type = "clip_text_deprojector_v_model"

    default_activation = "quick_gelu"
    default_projection_dim = 768
    default_seq_len = 77
    default_embed_dim = 768
    default_use_individual_first_layer = False
    default_sum_after_first_activation = False
    default_first_embed_layer_dim = 768
    default_first_layer_dim = 768 * 1
    default_num_mlp_layers = 1
    default_mlp_layer_dim = 768 * 4
    default_use_residual_connection = False
    default_layer_norm_eps = 1e-05
    default_use_side_input = False
    default_vicinity = 3
    default_relative_to_null = True
    default_ensemble_size = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = kwargs.get("activation", self.__class__.default_activation)
        self.projection_dim = kwargs.get(
            "projection_dim", self.__class__.default_projection_dim
        )
        self.seq_len = kwargs.get("seq_len", self.__class__.default_seq_len)
        self.embed_dim = kwargs.get("embed_dim", self.__class__.default_embed_dim)
        self.use_individual_first_layer = kwargs.get(
            "use_individual_first_layer",
            self.__class__.default_use_individual_first_layer,
        )
        self.sum_after_first_activation = kwargs.get(
            "sum_after_first_activation",
            self.__class__.default_sum_after_first_activation,
        )
        self.first_embed_layer_dim = kwargs.get(
            "first_embed_layer_dim", self.__class__.default_first_embed_layer_dim
        )
        self.first_layer_dim = kwargs.get(
            "first_layer_dim", self.__class__.default_first_layer_dim
        )
        self.num_mlp_layers = kwargs.get(
            "num_mlp_layers", self.__class__.default_num_mlp_layers
        )
        self.mlp_layer_dim = kwargs.get(
            "mlp_layer_dim", self.__class__.default_mlp_layer_dim
        )
        self.use_residual_connection = kwargs.get(
            "use_residual_connection", self.__class__.default_use_residual_connection
        )
        self.layer_norm_eps = kwargs.get(
            "layer_norm_eps", self.__class__.default_layer_norm_eps
        )
        self.use_side_input = kwargs.get(
            "use_side_input", self.__class__.default_use_side_input
        )
        self.vicinity = kwargs.get("vicinity", self.__class__.default_vicinity)
        self.relative_to_null = kwargs.get(
            "relative_to_null", self.__class__.default_relative_to_null
        )
        self.ensemble_size = kwargs.get(
            "ensemble_size", self.__class__.default_ensemble_size
        )


class CLIPTextDeprojectorMLPLayer(PreTrainedModel):
    config_class = CLIPTextDeprojectorVConfig

    def __init__(self, config: CLIPTextDeprojectorVConfig):
        super().__init__(config)
        self.config = config

        self.activation_fn = ACT2FN[config.activation]
        self.norm = nn.BatchNorm1d(config.embed_dim, eps=config.layer_norm_eps)
        if config.use_side_input:
            input_dim = config.embed_dim * 2
        else:
            input_dim = config.embed_dim
        self.linear1 = nn.Linear(input_dim, config.mlp_layer_dim)
        self.linear2 = nn.Linear(config.mlp_layer_dim, config.embed_dim)

    def forward(self, states, embeds):
        if self.config.use_residual_connection:
            residual = states
        states = self.norm(states)
        if self.config.use_side_input:
            states = torch.cat([embeds, states], dim=1)
        states = self.linear1(states)
        states = self.activation_fn(states)
        states = self.linear2(states)
        if self.config.use_residual_connection:
            states = states + residual
        return states


class CLIPTextDeprojectorV(PreTrainedModel):
    config_class = CLIPTextDeprojectorVConfig

    def __init__(self, config: CLIPTextDeprojectorVConfig):
        super().__init__(config)
        self.config = config

        self.register_buffer(
            "deprojection", torch.empty([config.embed_dim, config.projection_dim])
        )
        self.register_buffer("sos_embed", torch.empty([config.embed_dim]))
        self.register_buffer("null_embed", torch.empty([config.embed_dim]))

        embed_dim = config.embed_dim
        ensemble_size = config.ensemble_size

        self.activation_fn = ACT2FN[config.activation]
        if config.use_individual_first_layer:
            first_embed_layer_dim = config.first_embed_layer_dim
            first_states_layer_dim = config.first_layer_dim
            self.linear1_embeds = nn.ModuleList(
                nn.Linear(embed_dim, first_embed_layer_dim)
                for _ in range(ensemble_size)
            )
            self.linear1_states = nn.ModuleList(
                nn.Linear(embed_dim, first_states_layer_dim)
                for _ in range(ensemble_size)
            )
            if config.sum_after_first_activation:
                first_layer_dim = first_embed_layer_dim + first_states_layer_dim
            else:
                first_layer_dim = (
                    first_embed_layer_dim + first_states_layer_dim * config.vicinity
                )
            self.linear2 = nn.ModuleList(
                nn.Linear(first_layer_dim, embed_dim) for _ in range(ensemble_size)
            )
        else:
            input_dim = embed_dim * (1 + config.vicinity)
            first_layer_dim = config.first_layer_dim
            self.linear1 = nn.ModuleList(
                nn.Linear(input_dim, first_layer_dim) for _ in range(ensemble_size)
            )
            self.linear2 = nn.ModuleList(
                nn.Linear(first_layer_dim, embed_dim) for _ in range(ensemble_size)
            )
        self.mlp_layers = nn.ModuleList(
            nn.ModuleList(
                CLIPTextDeprojectorMLPLayer(config)
                for _ in range(config.num_mlp_layers)
            )
            for _ in range(ensemble_size)
        )

        # Final normalizer
        self.final_norm = nn.ModuleList(
            nn.BatchNorm1d(embed_dim, eps=config.layer_norm_eps)
            for _ in range(ensemble_size)
        )

    def InitializeForTraining(self, mlp_weights=None):
        embed_dim = self.config.embed_dim
        first_layer_dim = self.config.first_layer_dim
        mlp_layer_dim = self.config.mlp_layer_dim

        if self.config.ensemble_size > 1:
            raise ValueError(
                f"ensemble_size must be 1 for training: {self.config.ensemble_size}"
            )
        for p in self.parameters():
            Debug(1, f"initialize parameter: {p.shape}")
            torch.nn.init.normal_(p, mean=0, std=0.01)

        if mlp_weights:
            input_mult = 1 + self.config.vicinity
            v_proj = mlp_weights["encoder_layer.self_attn.v_proj.weight"]
            v_proj = torch.cat([v_proj] * input_mult, dim=1) * (1.0 / input_mult)
            self.CheckAndInit(self.linear1[0].weight, v_proj)
            self.CheckAndInit(
                self.linear1[0].bias,
                mlp_weights["encoder_layer.self_attn.v_proj.bias"],
            )
            self.CheckAndInit(
                self.linear2[0].weight,
                mlp_weights["encoder_layer.self_attn.out_proj.weight"],
            )
            self.CheckAndInit(
                self.linear2[0].bias,
                mlp_weights["encoder_layer.self_attn.out_proj.bias"],
            )
            for layer in self.mlp_layers[0]:
                self.CheckAndInit(
                    layer.linear1.weight, mlp_weights["encoder_layer.mlp.fc1.weight"]
                )
                self.CheckAndInit(
                    layer.linear1.bias, mlp_weights["encoder_layer.mlp.fc1.bias"]
                )
                self.CheckAndInit(
                    layer.linear2.weight, mlp_weights["encoder_layer.mlp.fc2.weight"]
                )
                self.CheckAndInit(
                    layer.linear2.bias, mlp_weights["encoder_layer.mlp.fc2.bias"]
                )

    def CheckAndInit(self, p, d):
        if p.data.shape != d.shape:
            raise ValueError(
                f"Parameter shape doesn't match: {p.data.shape} v.s. {d.shape}"
            )
        p.data = d
        Debug(1, f"initialize parameter by pretrained: {p.shape}")

    def forward(self, idx, embeds, states):
        bsz, embed_dim = embeds.shape
        if self.config.use_residual_connection:
            residual = states[0]

        if self.config.use_individual_first_layer:
            states0 = self.linear1_embeds[idx](embeds)
            states = [self.linear1_states[idx](s) for s in states]
            states0 = self.activation_fn(states0)
            states = [self.activation_fn(s) for s in states]
            if self.config.sum_after_first_activation:
                states = [sum(states)]
            states = torch.cat([states0] + states, dim=1)
        else:
            states = torch.cat([embeds] + states, dim=1)
            states = self.linear1[idx](states)
            states = self.activation_fn(states)

        states = self.linear2[idx](states)
        if self.config.use_residual_connection:
            states = states + residual

        for layer in self.mlp_layers[idx]:
            states = layer(states, embeds)

        states = self.final_norm[idx](states)
        return states

    def Deproject(self, embeds):
        return nn.functional.linear(embeds, self.deprojection, None)

    def RunForTraining(self, embeds, final_states, from_projected=False, **kwargs):
        if self.config.ensemble_size > 1:
            raise ValueError(
                f"ensemble_size must be 1 for training: {self.config.ensemble_size}"
            )
        device = embeds.device
        bsz, embed_dim = embeds.shape
        if embed_dim != self.config.embed_dim:
            raise ValueError(
                f"Dimension of `embeds` must match the model's dimension."
                f" embeds.shape = {embeds.shape}"
            )
        if from_projected:
            embeds = self.Deproject(embeds)
        if self.config.relative_to_null:
            embeds = embeds - self.null_embed
        zero_state = torch.zeros(embeds.shape).to(device)
        state_window = deque([zero_state] * self.config.vicinity, self.config.vicinity)
        inputs = [[embeds] + list(state_window)]
        for seq in range(1, self.config.seq_len - 1):
            state = final_states[:, seq : seq + 1, :].squeeze(1)
            if self.config.relative_to_null:
                state = state - self.null_embed
            state_window.appendleft(state)
            inputs.append([embeds] + list(state_window))
        inputs = [torch.cat(ts, dim=0) for ts in zip(*inputs)]
        outputs = self(0, inputs[0], inputs[1:])
        # print(f"outputs shape = {outputs.shape}")
        return torch.cat(
            [
                self.sos_embed.reshape([1, 1, embed_dim]).repeat([1, bsz, 1]),
                outputs.reshape([self.config.seq_len - 1, bsz, embed_dim]),
            ],
            dim=0,
        ).permute([1, 0, 2])

    def Inference(self, embeds, from_projected=False, **kwargs):
        self.eval()
        device = embeds.device
        bsz, embed_dim = embeds.shape
        if embed_dim != self.config.embed_dim:
            raise ValueError(
                f"Dimension of `embeds` must match the model's dimension."
                f" embeds.shape = {embeds.shape}"
            )
        if from_projected:
            embeds = nn.functional.linear(embeds, self.deprojection, None)
        if self.config.relative_to_null:
            embeds = embeds - self.null_embed
        zero_state = torch.zeros(embeds.shape).to(device)
        state_windows = [
            deque([zero_state] * self.config.vicinity, self.config.vicinity)
            for _ in range(self.config.ensemble_size)
        ]
        outputs = [[]] * self.config.ensemble_size
        final_output = [self.sos_embed.reshape([1, 1, embed_dim]).repeat([bsz, 1, 1])]
        for _ in range(1, self.config.seq_len):
            outputs = [
                self(idx, embeds, list(state_windows[idx]))
                for idx in range(self.config.ensemble_size)
            ]

            # Take average for final output
            if self.config.ensemble_size == 1:
                average_output = outputs[0]
            else:
                average_output = sum(outputs) / self.config.ensemble_size
            final_output.append(average_output.unsqueeze(1))

            # Prepare for next token
            for idx in range(self.config.ensemble_size):
                output = outputs[idx]
                if self.config.relative_to_null:
                    output = output - self.null_embed
                state_windows[idx].appendleft(output)

        return torch.cat(final_output, dim=1)


#
# -- Transformer model --
#
class CLIPTextDeprojectorConfig(CLIPTextConfig):
    model_type = "clip_text_deprojector_model"
    default_ensemble_size = 1
    default_relative_to_null = False
    default_relative_to_prev = False
    default_state_projection = False
    default_state_weited_sum = False
    default_residual_wrt_prev = False
    default_residual_wrt_hidden = False
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
        self.state_projection = kwargs.get(
            "state_projection", self.__class__.default_state_projection
        )
        self.state_weited_sum = kwargs.get(
            "state_weited_sum", self.__class__.default_state_weited_sum
        )
        self.residual_wrt_prev = kwargs.get(
            "residual_wrt_prev", self.__class__.default_residual_wrt_prev
        )
        self.residual_wrt_hidden = kwargs.get(
            "residual_wrt_hidden", self.__class__.default_residual_wrt_hidden
        )
        self.apply_mlp_to_input = kwargs.get(
            "apply_mlp_to_input", self.__class__.default_apply_mlp_to_input
        )


class CLIPTextDeprojector(CLIPPreTrainedModel):
    config_class = CLIPTextDeprojectorConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    default_fuse = 0.0

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

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
            if self.config.state_projection:
                self.state_proj = nn.Linear(embed_dim * 3, embed_dim)
            elif self.config.state_weited_sum:
                self.state_weight = torch.nn.Parameter(
                    torch.Tensor([0.0, 1.0, 0.0]).unsqueeze(1)
                )
            if self.config.apply_mlp_to_input:
                self.mlp_to_input = CLIPMLP(config)
            self.position_embedding = nn.Embedding(
                config.max_position_embeddings, embed_dim
            )
            self.encoder_layer = CLIPEncoderLayer(config)
            self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            if self.config.state_projection:
                self.state_proj = nn.ModuleList(
                    [
                        nn.Linear(embed_dim * 3, embed_dim)
                        for _ in range(config.ensemble_size)
                    ]
                )
            elif self.config.state_weited_sum:
                self.state_weight = nn.ModuleList(
                    [
                        torch.nn.Parameter(torch.Tensor([0.0, 1.0, 0.0]).unsqueeze(1))
                        for _ in range(config.ensemble_size)
                    ]
                )
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

    def InitStateProjectionForTraining(self):
        if not self.config.state_projection:
            return
        torch.nn.init.normal_(self.state_proj.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.state_proj.bias, mean=0, std=0.01)

        embed_dim = self.config.hidden_size
        self.state_proj.weight.data = self.state_proj.weight.data + torch.cat(
            [
                0.5 * torch.eye(embed_dim),
                0.5 * torch.eye(embed_dim),
                torch.zeros([embed_dim, embed_dim]),
            ],
            dim=1,
        )

    def ConstructInput(
        self,
        embeds,
        prev_output,
        fn_to_input,
        fn_to_state,
        from_projected=False,
    ):
        seq_len = self.config.max_position_embeddings
        bsz, embed_dim = embeds.size()
        device = embeds.device
        if prev_output is None:
            prev_len = 0
        else:
            prev_len = prev_output.size()[1]

        if from_projected:
            embeds = self.projection(embeds)
        embeds = embeds.unsqueeze(1)
        orig_embeds = embeds
        if self.config.relative_to_null:
            embeds = embeds - self.null_embed

        null_embeds = self.null_embed.reshape([1, 1, embed_dim]).repeat([bsz, 1, 1])

        if fn_to_input:
            result = [fn_to_input(embeds), embeds]
        else:
            result = [null_embeds, embeds]

        result_len = 2
        if prev_len > 1:
            prev_output = prev_output[:, 1:, :]
            if self.config.state_weited_sum:
                if not fn_to_state:
                    raise ValueError("fn_to_state must not be None.")
                # prev of prev
                p1 = torch.cat([null_embeds, prev_output[:, :-1, :]], dim=1)
                # prev
                p2 = prev_output
                if self.config.relative_to_null:
                    p1 = p1 - self.null_embed
                    p2 = p2 - self.null_embed
                # embeds (already relative_to_null)
                p3 = embeds.repeat([1, prev_len - 1, 1])
                prev_output = fn_to_state(p1, p2, p3)
            elif self.config.state_projection:
                if not fn_to_state:
                    raise ValueError("fn_to_state must not be None.")
                # relative to prev
                p1 = prev_output - torch.cat(
                    [
                        null_embeds,
                        prev_output[:, :-1, :],
                    ],
                    dim=1,
                )
                # relative to null
                p2 = prev_output - null_embeds.repeat([1, prev_len - 1, 1])
                # relative to embeds
                p3 = orig_embeds.repeat([1, prev_len - 1, 1]) - prev_output
                # state projection
                prev_output = fn_to_state(p1, p2, p3)
            elif self.config.relative_to_prev:
                prev_output = prev_output - torch.cat(
                    [
                        null_embeds,
                        prev_output[:, :-1, :],
                    ],
                    dim=1,
                )
            elif self.config.relative_to_null:
                prev_output = prev_output - self.null_embed
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

    def StateProjFn(self, proj):
        def fn(p1, p2, p3):
            return proj(torch.cat([p1, p2, p3], dim=2))

        return fn

    def StateWeightFn(self, weight):
        def fn(p1, p2, p3):
            p = torch.cat([p1.unsqueeze(2), p2.unsqueeze(2), p3.unsqueeze(2)], dim=2)
            return torch.sum(weight * p, dim=2)

        return fn

    def forward(self, embeds, prev_outputs, **kwargs):
        if isinstance(self.encoder_layer, nn.ModuleList):
            if self.config.state_projection:
                state_fn = [self.StateProjFn(p) for p in self.state_proj]
            elif self.config.state_weited_sum:
                state_fn = [self.StateWeightFn(w) for w in self.state_weight]
            else:
                state_fn = [None] * self.config.ensemble_size
            if self.config.apply_mlp_to_input:
                mlp_to_input_fn = self.mlp_to_input
            else:
                mlp_to_input_fn = [None] * self.config.ensemble_size
            position_embedding_fn = self.position_embedding
            encoder_layer_fn = self.encoder_layer
            final_layer_norm_fn = self.final_layer_norm
        else:
            if self.config.state_projection:
                state_fn = [self.StateProjFn(self.state_proj)]
            elif self.config.state_weited_sum:
                state_fn = [self.StateWeightFn(self.state_weight)]
            else:
                state_fn = [None]
            if self.config.apply_mlp_to_input:
                mlp_to_input_fn = [self.mlp_to_input]
            else:
                mlp_to_input_fn = [None]
            position_embedding_fn = [self.position_embedding]
            encoder_layer_fn = [self.encoder_layer]
            final_layer_norm_fn = [self.final_layer_norm]

        # input construction
        position_embeddings = [fn(self.position_ids) for fn in position_embedding_fn]
        hidden_states = [
            self.ConstructInput(embeds, p, mlp, st, **kwargs) + pe
            for p, pe, mlp, st in zip(
                prev_outputs, position_embeddings, mlp_to_input_fn, state_fn
            )
        ]

        bsz, seq_len, embed_dim = hidden_states[0].size()
        device = embeds.device

        # transformer
        attention_mask = None
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states[0].dtype
        ).to(hidden_states[0].device)
        layer_outputs = [
            fn(hs, attention_mask, causal_attention_mask)[0]
            for fn, hs in zip(encoder_layer_fn, hidden_states)
        ]

        # final formulation
        if self.config.residual_wrt_prev:
            layer_outputs = [
                self.AddPrevOutput(lo, po)
                for lo, po in zip(layer_outputs, prev_outputs)
            ]
        elif self.config.residual_wrt_hidden:
            if self.config.relative_to_null:
                layer_outputs = [
                    lo
                    + torch.cat(
                        [torch.zeros([bsz, 2, embed_dim]).to(device), hs[:, 2:, :]],
                        dim=1,
                    )
                    + self.null_embed
                    for lo, hs in zip(layer_outputs, hidden_states)
                ]
            else:
                raise NotImplementedError(
                    "residual_wrt_hidden isn't implemented without relative_to_null."
                )
        output = [fn(lo) for fn, lo in zip(final_layer_norm_fn, layer_outputs)]

        # replacing SOS
        sos_embeds = self.sos_embed.reshape([1, 1, embed_dim]).repeat([bsz, 1, 1])
        return [torch.cat([sos_embeds, o[:, 1:]], dim=1) for o in output]

    def RunForTraining(self, embeds, final_states, **kwargs):
        return self(embeds, [final_states], **kwargs)[0]

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
            if self.config.state_projection:
                copy_params(new_model.state_proj, models[0].state_proj)
            elif self.config.state_weited_sum:
                copy_params(new_model.state_weight, models[0].state_weight)
            if new_config.apply_mlp_to_input:
                copy_params(new_model.mlp_to_input, models[0].mlp_to_input)
            copy_params(new_model.position_embedding, models[0].position_embedding)
            copy_params(new_model.encoder_layer, models[0].encoder_layer)
            copy_params(new_model.final_layer_norm, models[0].final_layer_norm)
        else:
            for i, model in enumerate(models):
                if self.config.state_projection:
                    copy_params(new_model.state_proj[i], models[i].state_proj)
                elif self.config.state_weited_sum:
                    copy_params(new_model.state_weight[i], models[i].state_weight)
                if new_config.apply_mlp_to_input:
                    copy_params(new_model.mlp_to_input[i], models[i].mlp_to_input)
                copy_params(new_model.position_embedding[i], model.position_embedding)
                copy_params(new_model.encoder_layer[i], model.encoder_layer)
                copy_params(new_model.final_layer_norm[i], model.final_layer_norm)
        return new_model
