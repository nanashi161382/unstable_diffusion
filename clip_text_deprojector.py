import copy
from collections import deque
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

        zero_output = states_output[:, :1, :]
        if use_first_embed_layer:
            first_output = [embeds_output.repeat([1, seq_len, 1]), states_output]
        else:
            first_output = [states_output]
        for i in range(1, self.config.vicinity):
            zero_states = torch.zeros([bsz, i, embed_dim], device=device)
            if i > seq_len:
                first_output.append(zero_output.repeat([1, seq_len, 1]))
            else:
                first_output.append(
                    torch.cat(
                        [zero_output.repeat([1, i, 1]), states_output[:, :-i, :]], dim=1
                    )
                )
        first_output = torch.cat(first_output, dim=2)
        return first_output.clone()

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


#
# -- Deprecated --
#
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
