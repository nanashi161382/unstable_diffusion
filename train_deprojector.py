import csv
import math
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPPreTrainedModel,
    CLIPTextConfig,
)
from transformers.activations import ACT2FN
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

device = "cuda"


class MLP3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        intermediate_size = hidden_size * 4
        self.activation_fn = ACT2FN["quick_gelu"]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(0.02)
        self.fc2 = nn.Linear(intermediate_size, intermediate_size)
        self.fc3 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hideen_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc3(hidden_states)
        return hidden_states


class CLIPTextDeprojector(CLIPPreTrainedModel):
    config_class = CLIPTextConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.projection = nn.Linear(config.projection_dim, embed_dim, bias=False)
        for param in self.projection.parameters():
            param.requires_grad = False  # Fix the parameter of the projection layer.

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )
        self.encoder_layer = CLIPEncoderLayer(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.register_buffer("sos_embed", torch.zeros([embed_dim]))
        self.register_buffer("null_embed", torch.zeros([embed_dim]))

        # self.expand = nn.Linear(embed_dim, embed_dim)  # * 2)
        # self.expand_act = ACT2FN["quick_gelu"]

    def forward(self, embeds, prev_output, **kwargs):
        hidden_state = self.ConstructInput(embeds, prev_output, **kwargs)
        bsz, seq_len, embed_dim = hidden_state.size()

        attention_mask = None
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_state.dtype
        ).to(hidden_state.device)

        position_embeddings = self.position_embedding(self.position_ids)
        hidden_state = hidden_state + position_embeddings

        layer_outputs = self.encoder_layer(
            hidden_state,
            attention_mask,
            causal_attention_mask,
        )
        output = self.final_layer_norm(layer_outputs[0])

        sos_embeds = self.sos_embed.reshape([1, 1, embed_dim]).repeat([bsz, 1, 1])
        return torch.cat([sos_embeds, output[:, 1:]], dim=1)

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

    def Inference(self, embeds, **kwargs):
        self.eval()
        seq_len = self.config.max_position_embeddings
        result_len = 2
        result = self(embeds, None, **kwargs)[:, :result_len, :]
        while True:
            result_len += 1
            result = self(embeds, result, **kwargs)[:, :result_len, :]
            if result_len == seq_len:
                return result

    def ConstructInput(
        self,
        embeds,
        prev_output,
        proj=False,
        diff=True,
        embed_second=True,
        expand=False,
    ):
        seq_len = self.config.max_position_embeddings
        bsz, embed_dim = embeds.size()
        device = embeds.device
        if prev_output is None:
            prev_len = 0
        else:
            prev_len = prev_output.size()[1]

        if proj:
            embeds = self.projection(embeds)
        embeds = embeds.unsqueeze(1)

        if diff or embed_second or expand:
            null_embeds = self.null_embed.reshape([1, 1, embed_dim]).repeat([bsz, 1, 1])

        if expand:
            if diff:
                embeds = embeds - null_embeds
            result = [self.expand_act(self.expand(embeds)), embeds]
            # result = [self.expand_act(self.expand(embeds)).reshape([bsz, 2, embed_dim])]
        elif embed_second:
            if diff:
                embeds = embeds - null_embeds
            result = [null_embeds, embeds]
        else:
            sos_embeds = self.sos_embed.reshape([1, 1, embed_dim]).repeat([bsz, 1, 1])
            if diff:
                sos_embeds = sos_embeds - null_embeds
                embeds = embeds - null_embeds
            result = [embeds, sos_embeds]

        result_len = 2
        if prev_len > 1:
            if diff:
                prev_output = prev_output - null_embeds.repeat([1, prev_len, 1])
            result.append(prev_output[:, 1:, :])
            result_len += prev_len - 1

        if result_len < seq_len:
            result.append(
                torch.zeros([bsz, seq_len - result_len, embed_dim]).to(device)
            )
        elif result_len > seq_len:
            result[-1] = result[-1][:, : (seq_len - result_len), :]

        result = torch.cat(result, dim=1)
        return result


def Offload(*ls):
    for x in ls:
        if x is not None:
            x.to("cpu")


def ReadTexts(base_dir, start, end):
    is_first_row = True
    texts = []
    with open(f"{base_dir}/texts.csv", newline="") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if len(row) == 0:
                print(f"skipping empty row {i:08n}")
                continue
            if i == 0:
                print(f"{i:08n} => {row}")
                continue
            if i < start:
                continue
            if is_first_row:
                print(f"First row: {i:08n} => {row}")
                is_first_row = False
            texts.append((i, row[0]))
            if i >= end:
                break
    return texts


class TextModel:
    def __init__(self, old_model=None, path="openai/clip-vit-large-patch14"):
        if old_model:
            self.tokenizer = old_model.tokenizer
            self.encoder = old_model.encoder
            return
        self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=False)
        self.encoder = CLIPTextModelWithProjection.from_pretrained(
            path, local_files_only=False
        ).to(device)

    def tokenize(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="pt",  # PyTorch
        )

    def Encode(self, text, img_embeds=None):
        tokenized = self.tokenize(text)
        tokens = tokenized.input_ids.to(device)
        encoded = self.encoder.text_model(tokens)
        embeds = self.encoder.text_projection(encoded[1])
        truncated = tokenized.num_truncated_tokens > 0

        if img_embeds is not None:  # Add noise with image embeds
            ratio = np.random.exponential(2) / 100
            embeds = (1.0 - ratio) * embeds + ratio * img_embeds.to(device)

        return (
            embeds,  # projected
            encoded[1],  # pooled_output
            encoded.last_hidden_state,
            tokenized.attention_mask,
            truncated,
        )


class DataGenerator:
    def __init__(self, text_model):
        self.text_model = text_model
        p, e, s, m, _ = self.text_model.Encode("", None)
        Offload(p, s, m)
        self.base_embeds = e

    def GetImageEmbeds(self, img_embed_data, i):
        if img_embed_data is not None:
            return (
                torch.from_numpy(img_embed_data[i - 1].astype(np.float32))
                .clone()
                .unsqueeze(0)
            )
        return None

    def SplitList(self, ls):
        if len(ls) < 2:
            return None, None
        i = random.randint(1, len(ls) - 1)
        return ls[:i], ls[i:]

    def AverageEmbeddings(self, *es):
        embeds = [e - self.base_embeds for e in es]
        embeds = sum(embeds) + self.base_embeds
        return embeds

        # embeds = sum(es)
        # coeff = sum(torch.norm(e) for e in es) / len(es) / torch.norm(embeds)
        # embeds *= coeff
        # embeds = (embeds * coeff).clone()
        # Offload(coeff)
        # return embeds

    def SplitEmbeds(self, data, text, img_embeds, embeds, last_state, mask):
        fst, snd = self.SplitList(text.split(" "))
        if not fst:
            return False
        p1, e1, s1, m1, _ = self.text_model.Encode(" ".join(fst), img_embeds)
        p2, e2, s2, m2, _ = self.text_model.Encode(" ".join(snd), img_embeds)
        data.Add(self.AverageEmbeddings(e1, e2), embeds, last_state, mask)
        Offload(p1, e1, s1, m1, p2, e2, s2, m2)
        return True

    def PickIndex(self, length, used_idx):
        i = random.randint(0, length - 1 - len(used_idx))
        for k, idx in enumerate(used_idx):
            if i == idx:
                i = length - 1 - k
                break
        return i

    def CombineEmbeds(self, data, text, i2, texts, img_embed_data, embeds):
        text2 = texts[i2][1]
        img_embeds2 = self.GetImageEmbeds(img_embed_data, i2)
        p2, e2, s2, m2, truncated = self.text_model.Encode(text2, img_embeds2)
        Offload(p2, e2, s2, m2, img_embeds2)
        if truncated:
            return False
        text3 = text + " " + text2
        p3, e3, s3, mask, truncated = self.text_model.Encode(text3)
        Offload(p3)
        if truncated:
            Offload(e3, s3)
            return False
        data.Add(self.AverageEmbeddings(embeds, e2), e3, s3, mask)
        return True

    def GenerateData(
        self, data, texts, percent, img_embed_data, only_normal, add_empty=True
    ):
        skip = False
        num_normal = 0
        num_split = 0
        num_combine = 0

        # Add empty text
        if add_empty:
            projected, embeds, last_state, mask, truncated = self.text_model.Encode(
                "", None
            )
            Offload(projected)
            data.Add(embeds, embeds, last_state, mask)
            num_normal += 1
            print(f"Add embedding for the empty text.")

        for i, (orig_i, text) in enumerate(texts):
            if i % 500 == 499:
                print(f"{i+1:05n}: {orig_i:08n} => {text}")
                torch.cuda.empty_cache()
            if skip:
                skip = False
                continue

            img_embeds = self.GetImageEmbeds(img_embed_data, orig_i)
            try:
                # normal
                projected, embeds, last_state, mask, truncated = self.text_model.Encode(
                    text, img_embeds
                )
                Offload(projected)
                if only_normal:
                    data.Add(embeds, embeds, last_state, mask)
                    num_normal += 1
                    Offload(img_embeds)
                    continue

                if False:  # v9
                    data.Add(embeds, embeds, last_state, mask)
                    num_normal += 1

                    rn = random.randint(0, 100)
                    i2 = self.PickIndex(len(texts), i)
                    if rn == 0:  #  1 / 101
                        if self.SplitEmbeds(
                            data, text, img_embeds, embeds, last_state, mask
                        ):
                            num_split += 1
                        if self.CombineEmbeds(
                            data, text, i2, texts, img_embed_data, embeds
                        ):
                            num_combine += 1
                    elif rn <= 50:  # 50 / 101
                        if self.SplitEmbeds(
                            data, text, img_embeds, embeds, last_state, mask
                        ):
                            num_split += 1
                        elif self.CombineEmbeds(
                            data, text, i2, texts, img_embed_data, embeds
                        ):
                            num_combine += 1
                    else:  # 50 / 101
                        if self.CombineEmbeds(
                            data, text, i2, texts, img_embed_data, embeds
                        ):
                            num_combine += 1
                        elif self.SplitEmbeds(
                            data, text, img_embeds, embeds, last_state, mask
                        ):
                            num_split += 1
                    Offload(img_embeds)
                    continue

                rn = random.randint(0, 199)
                if (rn % 100 < percent) or truncated:
                    # normal
                    data.Add(embeds, embeds, last_state, mask)
                    num_normal += 1
                elif rn < 100:
                    # split
                    if self.SplitEmbeds(
                        data, text, img_embeds, embeds, last_state, mask
                    ):
                        num_split += 1
                        Offload(embeds)
                    else:
                        # fall back to normal
                        data.Add(embeds, embeds, last_state, mask)
                        num_normal += 1
                else:
                    # combine
                    i2 = i + 1
                    ok = False
                    if i2 < len(texts):
                        ok = self.CombineEmbeds(
                            data, text, i2, texts, img_embed_data, embeds
                        )
                        if ok:
                            num_combine += 1
                            Offload(last_state, mask)
                            skip = True
                    if not ok:
                        # fall back to split
                        if self.SplitEmbeds(
                            data, text, img_embeds, embeds, last_state, mask
                        ):
                            num_split += 1
                            Offload(embeds)
                        else:
                            # fall back to normal
                            data.Add(embeds, embeds, last_state, mask)
                            num_normal += 1

            except Exception:
                print(f"error at: {orig_i:08n} => {text}")
                print(f"data size = {len(data)}")
                raise
            Offload(img_embeds)

        torch.cuda.empty_cache()
        return num_normal, num_split, num_combine


class Data:
    def __init__(self, data=None):
        if data:  # copy constructor
            self.embed_ls = data.embed_ls
            self.target_embed_ls = data.target_embed_ls
            self.last_state_ls = data.last_state_ls
            self.mask_ls = data.mask_ls
            return
        self.embed_ls = []
        self.target_embed_ls = []
        self.last_state_ls = []
        self.mask_ls = []

    def Add(self, embeds, target_embeds, last_states, mask):
        # embeds.shape = [1, 768]
        # target_embeds.shape = [1, 768]
        # last_states = [1, 77, 768]
        # mask = [1, 77]
        self.embed_ls.append(embeds)
        self.target_embed_ls.append(target_embeds)
        self.last_state_ls.append(last_states)
        self.mask_ls.append(mask)

    def Finalize(self):
        # embeds_ls.shape = [data_len, 768]
        # target_embed_ls.shape = [data_len, 768]
        # last_states_ls = [data_len, 77, 768]
        # mask_ls = [data_len, 77]
        self.embed_ls = self.CatAndRelease(self.embed_ls)
        self.target_embed_ls = self.CatAndRelease(self.target_embed_ls)
        self.last_state_ls = self.CatAndRelease(self.last_state_ls)
        self.mask_ls = self.CatAndRelease(self.mask_ls)
        torch.cuda.empty_cache()

    def CatAndRelease(self, ls):
        t = torch.cat(ls, dim=0).clone()
        for x in ls:
            x.to("cpu")
        return t

    def __len__(self):
        return len(self.embed_ls)

    def __getitem__(self, index):
        # [0].shape = [768]
        # [1].shape = [768]
        # [2].shape = [77, 768]
        # [3].shape = [77]
        return (
            self.embed_ls[index],
            self.target_embed_ls[index],
            self.last_state_ls[index],
            self.mask_ls[index],
        )

    def to(self, device):
        if not isinstance(self.embed_ls, torch.Tensor):
            print(f"Data isn't finalized yet.")
            return self
        self.embed_ls.to(device)
        self.target_embed_ls.to(device)
        self.last_state_ls.to(device)
        self.mask_ls.to(device)
        return self

    def CleanUp(self):
        self.to("cpu")
        del self.embed_ls
        del self.target_embed_ls
        del self.last_state_ls
        del self.mask_ls
        torch.cuda.empty_cache()

    def Save(self, filename, with_target_embeds=False):
        if not isinstance(self.embed_ls, torch.Tensor):
            raise ValueError(f"Data isn't finalized yet.")
        target_embeds = 0.0
        if with_target_embeds:
            target_embeds = self.target_embed_ls
        torch.save(
            {
                "embeds": self.embed_ls,
                "target_embeds": target_embeds,
                "last_states": self.last_state_ls,
                "masks": self.mask_ls,
            },
            filename,
        )

    def Load(self, filename, with_target_embeds=False):
        dct = torch.load(filename)
        self.embed_ls = dct["embeds"]
        if with_target_embeds:
            self.target_embed_ls = dct["target_embeds"]
        else:
            self.target_embed_ls = torch.zeros(self.embed_ls.shape)
        self.last_state_ls = dct["last_states"]
        self.mask_ls = dct["masks"]


class DataSet:
    def __init__(self, data, for_training, weight_mult=25, noise_std=0):
        self.data = data.to(device)
        self.for_training = for_training
        self.weight_mult = weight_mult
        self.noise_std = noise_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        embeds, _, last_state, mask = self.data[index]
        embeds = self.AddNoise(embeds, self.noise_std)
        return (
            last_state,
            embeds,  # torch.cat([embeds.unsqueeze(0), last_state], dim=0)[:-1, :],
            self.ComputeWeightsForTraining(mask)
            if self.for_training
            else self.ComputeWeightsForTest(mask),
            mask.to(device),
        )

    def AddNoise(self, t, std):
        return t if std == 0 else t + torch.normal(0, std, t.shape).to(t.device)

    def ComputeWeightsForTraining(self, weights):
        mult = math.sqrt(self.weight_mult)
        weights = torch.ones(76) + (mult - 1) * weights[1:]
        weights = torch.cat([torch.zeros(1), weights], dim=0)
        weights = weights * (len(weights) / sum(weights))
        return weights.to(device)

    def ComputeWeightsForTest(self, weights):
        mult = math.sqrt(self.weight_mult)
        weights = torch.ones(76) + (mult - 1) * weights[1:]
        weights = torch.cat([torch.zeros(1), weights], dim=0)
        weights = weights * (len(weights) / sum(weights))
        return weights.to(device)
        # weights = torch.cat([torch.zeros(1), weights[1:]], dim=0)
        # weights = weights * (len(weights) / sum(weights))
        # return weights.to(device)


class Trainer:
    def __init__(self, model, dataset, batch_size, test_dataset, reg_weight=[0.0, 0.0]):
        self.model = model
        self.ds = dataset
        self.bsz = batch_size
        self.test_ds = test_dataset
        self.reg_weight = reg_weight

        # learning_rate = 0.001
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters())

    def regularize(self, pred, masks, weights):
        return self.regularize_1(pred, masks), self.regularize_2(pred, masks, weights)

    def regularize_1(self, pred, masks):
        masks = masks[:, :-2]
        # masks.shape = [bsz, 75] => [bsz, 75, 768]
        repeated_masks = (
            masks.reshape(masks.shape + (1,)).repeat(1, 1, pred.shape[-1]).to(device)
        )

        pred = pred[:, 1:-1, :] * repeated_masks
        sum_pred = torch.sum(pred, dim=1)
        sum_mask = (
            torch.sum(masks, dim=1)
            .reshape([sum_pred.shape[0], 1])
            .repeat(1, pred.shape[2])
            .to(device)
        )

        idx = sum_mask != 0.0
        avg_pred = torch.zeros(sum_pred.shape).to(pred.device)
        avg_pred[idx] = sum_pred[idx] / sum_mask[idx]

        # avg_pred.shape = [bsz, 768] => [bsz, 75, 768]
        avg_pred = (
            avg_pred.reshape((avg_pred.shape[0], 1, avg_pred.shape[1]))
            .repeat(1, pred.shape[1], 1)
            .to(device)
        )
        return 0.0 - self.criterion(pred, avg_pred * repeated_masks)  # nagate MSE

    def regularize_2(self, pred, masks, weights):
        pred = pred[:, 1:, :]
        masks = masks[:, 1:].to(device)
        weights = weights[:, 1:, :]
        masks = torch.ones(masks.shape).to(device) - masks
        masks = (
            masks.reshape(masks.shape + (1,)).repeat(1, 1, pred.shape[-1]).to(device)
        )
        return self.criterion(
            torch.zeros(pred.shape).to(device), weights * pred * masks
        )

    def ReshapeWeights(self, weights, input_shape):
        # weights.shape = [bsz, 77] => [bsz, 77, 768]
        return weights.reshape(weights.shape + (1,)).repeat(1, 1, input_shape[-1])

    def training(self, **kwargs):
        self.model.train()
        # sampler = RandomSampler(self.ds, replacement=False, num_samples=self.ds_size)
        dl = DataLoader(
            self.ds, batch_size=self.bsz, shuffle=True, drop_last=True
        )  # sampler=sampler)
        for i, (target, embeds, weights, masks) in enumerate(dl):
            weights = self.ReshapeWeights(weights, target.shape)
            self.optimizer.zero_grad()
            pred = self.model(embeds, target, **kwargs)
            w_loss = self.criterion(weights * pred, weights * target)
            reg_loss = self.regularize(pred, masks, weights)
            total_loss = w_loss + sum(w * l for w, l in zip(self.reg_weight, reg_loss))
            total_loss.backward()
            self.optimizer.step()
            if (i + 1) % 10 == 0:
                reg_loss = ", ".join(f"{l.item():.4f}" for l in reg_loss)
                print(
                    f"Batch {i+1}, total {total_loss.item():.4f}, weighted {w_loss.item():.4f},"
                    f" reg [{reg_loss}]"
                )
            Offload(weights, masks, pred, target, embeds)
            torch.cuda.empty_cache()
        if not isinstance(reg_loss, str):
            reg_loss = ", ".join(f"{l.item():.4f}" for l in reg_loss)
        print(
            f"Batch {i+1}, total {total_loss.item():.4f}, weighted {w_loss.item():.4f},"
            f" reg [{reg_loss}]"
        )

    def test(self, **kwargs):
        with torch.no_grad():
            self.model.eval()
            dl = DataLoader(self.test_ds, batch_size=len(self.test_ds))
            for target, embeds, weights, masks in dl:
                weights = self.ReshapeWeights(weights, target.shape)
                pred = self.model(embeds, target, **kwargs)
                loss = self.criterion(pred, target).item()
                w_loss = self.criterion(weights * pred, weights * target).item()
                reg_loss = self.regularize(pred, masks, weights)
                total_loss = (
                    w_loss
                    + (sum(w * l for w, l in zip(self.reg_weight, reg_loss))).item()
                )
                reg_str = ", ".join(f"{l.item():.4f}" for l in reg_loss)
                print(
                    f"Test, unweighted {loss:.4f}, total {total_loss:.4f},"
                    f" weighted {w_loss:.4f}, reg [{reg_str}]"
                )
                cs = self.cosine_similarity(pred, target)
                print(f"Test, cosine similarity:", cs)
                Offload(weights, masks, pred, target, embeds)
                return loss, w_loss, cs, [l.item() for l in reg_loss]

    def test_inference(self, **kwargs):
        with torch.no_grad():
            self.model.eval()
            dl = DataLoader(self.test_ds, batch_size=len(self.test_ds))
            for target, embeds, weights, masks in dl:
                weights = self.ReshapeWeights(weights, target.shape)
                pred = self.model.Inference(embeds, **kwargs)
                loss = self.criterion(pred, target).item()
                w_loss = self.criterion(weights * pred, weights * target).item()
                reg_loss = self.regularize(pred, masks, weights)
                total_loss = (
                    w_loss
                    + (sum(w * l for w, l in zip(self.reg_weight, reg_loss))).item()
                )
                reg_str = ", ".join(f"{l.item():.4f}" for l in reg_loss)
                print(
                    f"Inference test, unweighted {loss:.4f}, total {total_loss:.4f},"
                    f" weighted {w_loss:.4f}, reg [{reg_str}]"
                )
                cs = self.cosine_similarity(pred, target)
                print(f"Inference test, cosine similarity:", cs)
                Offload(weights, masks, pred, target, embeds)
                return loss, w_loss, cs, [l.item() for l in reg_loss]

    def cosine_similarity(self, t0, t1):
        return torch.nn.CosineSimilarity(dim=-1)(t0, t1).mean(dim=0).tolist()
