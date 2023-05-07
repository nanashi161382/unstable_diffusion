from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import random
import math
import csv
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModelWithProjection


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
        encoded = self.encoder(tokens)
        truncated = tokenized.num_truncated_tokens > 0

        embeds = encoded.text_embeds
        if img_embeds is not None:  # Add noise with image embeds
            ratio = np.random.exponential(2) / 100
            embeds = (1.0 - ratio) * embeds + ratio * img_embeds.to(device)

        return embeds, encoded.last_hidden_state, tokenized.attention_mask, truncated


class DataGenerator:
    def __init__(self, text_model):
        self.text_model = text_model

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
        embeds = sum(es)
        coeff = sum(torch.norm(e) for e in es) / len(es) / torch.norm(embeds)
        embeds *= coeff
        embeds = (embeds * coeff).clone()
        Offload(coeff)
        return embeds

    def SplitEmbeds(self, data, text, img_embeds, last_state, mask):
        fst, snd = self.SplitList(text.split(" "))
        if not fst:
            return False
        e1, s1, m1, _ = self.text_model.Encode(" ".join(fst), img_embeds)
        e2, s2, m2, _ = self.text_model.Encode(" ".join(snd), img_embeds)
        data.Add(self.AverageEmbeddings(e1, e2), last_state, mask)
        Offload(e1, s1, m1, e2, s2, m2)
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
        e2, s2, m2, truncated = self.text_model.Encode(text2, img_embeds2)
        Offload(e2, s2, m2, img_embeds2)
        if truncated:
            return False
        text3 = text + " " + text2
        e3, s3, mask, truncated = self.text_model.Encode(text3)
        Offload(e3)
        if truncated:
            Offload(s3)
            return False
        data.Add(self.AverageEmbeddings(embeds, e2), s3, mask)
        return True

    def GenerateData(self, data, texts, img_embed_data, only_normal):
        skip = False
        num_normal = 0
        num_split = 0
        num_combine = 0
        for i, (orig_i, text) in enumerate(texts):
            if i % 500 == 499:
                print(f"{i+1:04n}: {orig_i:08n} => {text}")
                torch.cuda.empty_cache()
            if skip:
                skip = False
                continue

            img_embeds = self.GetImageEmbeds(img_embed_data, orig_i)
            try:
                # normal
                embeds, last_state, mask, truncated = self.text_model.Encode(
                    text, img_embeds
                )
                if only_normal:
                    data.Add(embeds, last_state, mask)
                    num_normal += 1
                    Offload(img_embeds)
                    continue

                if False:  # v9
                    data.Add(embeds, last_state, mask)
                    num_normal += 1

                    rn = random.randint(0, 100)
                    i2 = self.PickIndex(len(texts), i)
                    if rn == 0:  #  1 / 101
                        if self.SplitEmbeds(data, text, img_embeds, last_state, mask):
                            num_split += 1
                        if self.CombineEmbeds(
                            data, text, i2, texts, img_embed_data, embeds
                        ):
                            num_combine += 1
                    elif rn <= 50:  # 50 / 101
                        if self.SplitEmbeds(data, text, img_embeds, last_state, mask):
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
                        elif self.SplitEmbeds(data, text, img_embeds, last_state, mask):
                            num_combine += 1
                    Offload(img_embeds)
                    continue

                # v10 or later
                rn = random.randint(0, 2)
                if (rn == 0) or truncated:
                    # normal
                    data.Add(embeds, last_state, mask)
                    num_normal += 1
                elif rn == 1:
                    # split
                    if self.SplitEmbeds(data, text, img_embeds, last_state, mask):
                        num_split += 1
                        Offload(embeds)
                    else:
                        # fall back to normal
                        data.Add(embeds, last_state, mask)
                        num_normal += 1
                elif rn == 2:
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
                        if self.SplitEmbeds(data, text, img_embeds, last_state, mask):
                            num_split += 1
                            Offload(embeds)
                        else:
                            # fall back to normal
                            data.Add(embeds, last_state, mask)
                            num_normal += 1
                else:
                    raise ValueError(f"unexpected random number: {rn}")

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
            self.last_state_ls = data.last_state_ls
            self.mask_ls = data.mask_ls
            return
        self.embed_ls = []
        self.last_state_ls = []
        self.mask_ls = []

    def Add(self, embeds, last_states, mask):
        # embeds.shape = [1, 768]
        # last_states = [1, 77, 768]
        # mask = [1, 77]
        self.embed_ls.append(embeds)
        self.last_state_ls.append(last_states)
        self.mask_ls.append(mask)

    def Finalize(self):
        # embeds_ls.shape = [data_len, 768]
        # last_states_ls = [data_len, 77, 768]
        # mask_ls = [data_len, 77]
        self.embed_ls = self.CatAndRelease(self.embed_ls)
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
        # [1].shape = [77, 768]
        # [2].shape = [77]
        return (
            self.embed_ls[index],
            self.last_state_ls[index],
            self.mask_ls[index],
        )

    def to(self, device):
        if not isinstance(self.embed_ls, torch.Tensor):
            print(f"Data isn't finalized yet.")
            return self
        self.embed_ls.to(device)
        self.last_state_ls.to(device)
        self.mask_ls.to(device)
        return self

    def CleanUp(self):
        self.to("cpu")
        del self.embed_ls
        del self.last_state_ls
        del self.mask_ls
        torch.cuda.empty_cache()

    def Save(self, filename):
        if not isinstance(self.embed_ls, torch.Tensor):
            raise ValueError(f"Data isn't finalized yet.")
        torch.save(
            {
                "embeds": self.embed_ls,
                "last_state": self.last_state_ls,
                "mask": self.mask_ls,
            },
            filename,
        )

    def Load(self, filename):
        raise NotImplementedError()


class DataSet:
    def __init__(self, data, noise_std=0, start=0, size=None):
        self.data = data.to(device)
        self.noise_std = noise_std
        self.start = start
        if size:
            self.size = size
        else:
            self.size = len(self.data) - start

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        embeds, last_state, mask = self.data[index]
        embeds = self.AddNoise(embeds, self.noise_std)
        token_len = self.GetTokenLen(mask)
        return (
            last_state,
            torch.cat([embeds.unsqueeze(0), last_state], dim=0)[:-1, :],
            self.GetWeights(token_len),
            self.GetCoreMask(mask),
        )

    def AddNoise(self, t, std):
        return t if std == 0 else t + torch.normal(0, std, t.shape).to(t.device)

    def GetTokenLen(self, mask):
        # mask.shape = [77]
        mask_len = torch.sum((mask > 0).long())
        return mask_len

    def GetWeights(self, token_len):
        mask_len = token_len + 1  # 1 more after EOS
        if mask_len > 77:
            mask_len = 77
        mask = torch.cat([torch.ones(mask_len - 1), torch.zeros(77 - mask_len)], dim=0)

        mult = math.sqrt(25)
        weights = torch.ones(76) + (mult - 1) * mask
        weights = torch.cat([torch.zeros(1), weights], dim=0)
        weights *= len(weights) / sum(weights)
        return weights.to(device)

    def GetCoreMask(self, mask):
        # Fill zeros for SOS and EOS.
        # mask.shape = [77]
        mask = torch.cat([torch.zeros(1), mask[2:], torch.zeros(1)], dim=0)
        return mask.to(device)


class Trainer:
    def __init__(self, dataset, batch_size, test_dataset, reg_weight=0.0):
        self.ds = dataset
        self.bsz = batch_size
        self.test_ds = test_dataset
        self.reg_weight = reg_weight

        # learning_rate = 0.001
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters())

    def regularize(self, pred, masks):
        pred = pred * masks
        sum_pred = torch.sum(pred, dim=1)
        sum_mask = torch.sum(masks, dim=1)
        avg_pred = sum_pred / sum_mask
        # avg_pred.shape = [bsz, 768] => [bsz, 77, 768]
        avg_pred = avg_pred.reshape((avg_pred.shape[0], 1, avg_pred.shape[1])).repeat(
            1, pred.shape[1], 1
        )
        return 0.0 - self.criterion(pred, avg_pred)  # nagate MSE

    def ReshapeWeights(self, weights):
        # weights.shape = [bsz, 77] => [bsz, 77, 768]
        return weights.reshape(weights.shape + (1,)).repeat(1, 1, input.shape[-1])

    def ReshapeMasks(self, masks):
        # masks.shape = [bsz, 77] => [bsz, 77, 768]
        return masks.reshape(masks.shape + (1,)).repeat(1, 1, input.shape[-1])

    def training(self):
        global epoch
        epoch += 1
        print(f"Epoch: {epoch}")

        # sampler = RandomSampler(self.ds, replacement=False, num_samples=self.ds_size)
        dl = DataLoader(
            self.ds, batch_size=self.bsz, shuffle=True, drop_last=True
        )  # sampler=sampler)
        for i, (target, input, weights, masks) in enumerate(dl):
            weights = self.ReshapeWeights(weights)
            masks = self.ReshapeMasks(masks)
            self.optimizer.zero_grad()
            pred = model(input)
            loss = self.criterion(weights * pred, weights * target)
            if self.reg_weight:
                loss += self.reg_weight * self.regularize(pred, masks)
            loss.backward()
            self.optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    f"Batch {i+1}, loss {loss.item():.4f}, target: {target.shape},"
                    f" input: {input.shape}"
                )
            Offload(weights, masks, pred, target, input)
            torch.cuda.empty_cache()

    def test(self):
        with torch.no_grad():
            dl = DataLoader(self.test_ds, batch_size=len(self.test_ds))
            for target, input, weights, masks in dl:
                weights = self.ReshapeWeights(weights)
                masks = self.ReshapeMasks(masks)
                pred = model(input)
                loss = self.criterion(pred, target).item()
                print(
                    f"Test, loss {loss:.4f}, target: {target.shape},"
                    f" input: {input.shape}"
                )
                w_loss = self.criterion(weights * pred, weights * target)
                if self.reg_weight:
                    w_loss += self.reg_weight * self.regularize(pred, masks)
                w_loss = w_loss.item()
                print(
                    f"Test, weighted loss {w_loss:.4f}, target: {target.shape},"
                    f" input: {input.shape}"
                )
                cs = self.cosine_similarity(pred, target)
                print(f"Test, cosine similarit:", cs)
                Offload(weights, masks, pred, target, input)
                return loss, w_loss, cs

    def test_inference(self):
        with torch.no_grad():
            dl = DataLoader(self.test_ds, batch_size=len(self.test_ds))
            for target, input, weights, masks in dl:
                weights = self.ReshapeWeights(weights)
                masks = self.ReshapeMasks(masks)
                pred = model.Inference(input[:, :1, :].squeeze(1))
                loss = self.criterion(pred, target).item()
                print(
                    f"Inference test, loss {loss:.4f}, target: {target.shape},"
                    f" input: {input.shape}"
                )
                w_loss = self.criterion(weights * pred, weights * target)
                if self.reg_weight:
                    w_loss += self.reg_weight * self.regularize(pred, masks)
                w_loss = w_loss.item()
                print(
                    f"Inference test, weighted loss {w_loss:.4f}, target: {target.shape},"
                    f" input: {input.shape}"
                )
                cs = self.cosine_similarity(pred, target)
                print(f"Inference test, cosine similarit:", cs)
                Offload(weights, masks, pred, target, input)
                return loss, w_loss, cs

    def cosine_similarity(self, t0, t1):
        return torch.nn.CosineSimilarity(dim=-1)(t0, t1).mean(dim=0).tolist()
