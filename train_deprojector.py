from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import random
import math
import csv
import torch
import numpy as np


def Offload(*ls):
    for x in ls:
        if x is not None:
            x.to("cpu")


def ReadTexts(start, end):
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


def GetImageEmbeds(img_embed_data, i):
    if img_embed_data is not None:
        return (
            torch.from_numpy(img_embed_data[i - 1].astype(np.float32))
            .clone()
            .unsqueeze(0)
        )
    return None


def Encode(text, img_embeds=None):
    tokenized = tokenize(text)
    tokens = tokenized.input_ids.to(device)
    encoded = encoder(tokens)
    truncated = tokenized.num_truncated_tokens > 0

    embeds = encoded.text_embeds
    if img_embeds is not None:  # Add noise with image embeds
        ratio = np.random.exponential(2) / 100
        embeds = (1.0 - ratio) * embeds + ratio * img_embeds.to(device)
    return embeds, encoded.last_hidden_state, tokenized.attention_mask, truncated


def SplitList(ls):
    if len(ls) < 2:
        return None, None
    i = random.randint(1, len(ls) - 1)
    return ls[:i], ls[i:]


def AverageEmbeddings(*es):
    embeds = sum(es)
    coeff = sum(torch.norm(e) for e in es) / len(es) / torch.norm(embeds)
    embeds *= coeff
    embeds = (embeds * coeff).clone()
    Offload(coeff)
    return embeds


def SplitEmbeds(data, text, img_embeds, last_state, mask):
    fst, snd = SplitList(text.split(" "))
    if not fst:
        return False
    e1, s1, m1, _ = Encode(" ".join(fst), img_embeds)
    e2, s2, m2, _ = Encode(" ".join(snd), img_embeds)
    data.Add(AverageEmbeddings(e1, e2), last_state, mask)
    Offload(e1, s1, m1, e2, s2, m2)
    return True


def PickIndex(length, used_idx):
    i = random.randint(0, length - 1 - len(used_idx))
    for k, idx in enumerate(used_idx):
        if i == idx:
            i = length - 1 - k
            break
    return i


def CombineEmbeds(data, text, i2, texts, img_embed_data, embeds):
    text2 = texts[i2][1]
    img_embeds2 = GetImageEmbeds(img_embed_data, i2)
    e2, s2, m2, truncated = Encode(text2, img_embeds2)
    Offload(e2, s2, m2, img_embeds2)
    if truncated:
        return False
    text3 = text + " " + text2
    e3, s3, mask, truncated = Encode(text3)
    Offload(e3)
    if truncated:
        Offload(s3)
        return False
    data.Add(AverageEmbeddings(embeds, e2), s3, mask)
    return True


def GenerateData(data, texts, img_embed_data, only_normal):
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

        img_embeds = GetImageEmbeds(img_embed_data, orig_i)
        try:
            # normal
            embeds, last_state, mask, truncated = Encode(text, img_embeds)
            if only_normal:
                data.Add(embeds, last_state, mask)
                num_normal += 1
                Offload(img_embeds)
                continue

            if False:  # v9
                data.Add(embeds, last_state, mask)
                num_normal += 1

                rn = random.randint(0, 100)
                i2 = PickIndex(len(texts), i)
                if rn == 0:  #  1 / 101
                    if SplitEmbeds(data, text, img_embeds, last_state, mask):
                        num_split += 1
                    if CombineEmbeds(data, text, i2, texts, img_embed_data, embeds):
                        num_combine += 1
                elif rn <= 50:  # 50 / 101
                    if SplitEmbeds(data, text, img_embeds, last_state, mask):
                        num_split += 1
                    elif CombineEmbeds(data, text, i2, texts, img_embed_data, embeds):
                        num_combine += 1
                else:  # 50 / 101
                    if CombineEmbeds(data, text, i2, texts, img_embed_data, embeds):
                        num_combine += 1
                    elif SplitEmbeds(data, text, img_embeds, last_state, mask):
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
                if SplitEmbeds(data, text, img_embeds, last_state, mask):
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
                    ok = CombineEmbeds(data, text, i2, texts, img_embed_data, embeds)
                    if ok:
                        num_combine += 1
                        Offload(last_state, mask)
                        skip = True
                if not ok:
                    # fall back to split
                    if SplitEmbeds(data, text, img_embeds, last_state, mask):
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


def AddNoise(t, std):
    return t if std == 0 else t + torch.normal(0, std, t.shape).to(t.device)


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
        self.embed_ls.append(embeds)
        self.last_state_ls.append(last_states)
        self.mask_ls.append(mask)

    def Finalize(self):
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
        embeds = AddNoise(embeds, self.noise_std)
        token_len = self.GetTokenLen(mask)
        return (
            last_state,
            torch.cat([embeds.unsqueeze(0), last_state], dim=0)[:-1, :],
            self.GetWeights(token_len),
            self.GetCoreMask(mask),
        )

    def GetTokenLen(self, mask):
        print(f"tolen_len: mask.shape {mask.shape}")
        # mask = mask[0]
        mask_len = torch.sum((mask > 0).long())
        print(f"token_len: mask_len {mask_len}")
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
        print(f"core_mask: mask.shape {mask.shape}")
        # mask = mask[0]
        mask = torch.cat([torch.zeros(1), mask[2:], torch.zeros(1)], dim=0)
        return mask.to(device)  # .unsqueeze(0)


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
        print(f"avg_pred.shape {avg_pred.shape}")
        avg_pred = avg_pred.reshape((avg_pred.shape[0], 1, avg_pred.shape[1])).repeat(
            1, pred.shape[1], 1
        )
        print(f"avg_pred.shape {avg_pred.shape}")
        return 0.0 - self.criterion(pred, avg_pred)  # nagate MSE

    def training(self):
        global epoch
        epoch += 1
        print(f"Epoch: {epoch}")

        # sampler = RandomSampler(self.ds, replacement=False, num_samples=self.ds_size)
        dl = DataLoader(
            self.ds, batch_size=self.bsz, shuffle=True, drop_last=True
        )  # sampler=sampler)
        for i, (target, input, weights, masks) in enumerate(dl):
            weights = weights.reshape(weights.shape + (1,)).repeat(
                1, 1, input.shape[-1]
            )
            self.optimizer.zero_grad()
            pred = model(input)
            loss = self.criterion(weights * pred, weights * target)
            if self.reg_weight:
                loss += self.reg_weight * self.regularize(pred, masks)
            loss.backward()
            self.optimizer.step()
            if (i + 1) % 10 == 0:
                print(
                    f"Batch {i+1}, loss {loss.item():.4f}, target: {target.shape}, input: {input.shape}"
                )
                # print(weights)
            Offload(weights, masks, pred, target, input)
            torch.cuda.empty_cache()

    def test(self):
        with torch.no_grad():
            dl = DataLoader(self.test_ds, batch_size=len(self.test_ds))
            for target, input, weights, masks in dl:
                print(f"input.shape {input.shape}")
                print(f"weights.shape {weights.shape}")
                weights = weights.reshape(weights.shape + (1,)).repeat(
                    1, 1, input.shape[-1]
                )
                masks = masks.reshape(masks.shape + (1,)).repeat(1, 1, input.shape[-1])
                print(f"weights.shape {weights.shape}")
                pred = model(input)
                loss = self.criterion(pred, target).item()
                print(
                    f"Test, loss {loss:.4f}, target: {target.shape}, input: {input.shape}"
                )
                w_loss = self.criterion(weights * pred, weights * target)
                if self.reg_weight:
                    w_loss += self.reg_weight * self.regularize(pred, masks)
                w_loss = w_loss.item()
                print(
                    f"Test, weighted loss {w_loss:.4f}, target: {target.shape}, input: {input.shape}"
                )
                cs = self.cosine_similarity(pred, target)
                print(f"Test, cosine similarit:", cs)
                return loss, w_loss, cs

    def test_inference(self):
        with torch.no_grad():
            dl = DataLoader(self.test_ds, batch_size=len(self.test_ds))
            for target, input, weights, masks in dl:
                weights = weights.reshape(weights.shape + (1,)).repeat(
                    1, 1, input.shape[-1]
                )
                pred = model.Inference(input[:, :1, :].squeeze(1))
                loss = self.criterion(pred, target).item()
                print(
                    f"Inference test, loss {loss:.4f}, target: {target.shape}, input: {input.shape}"
                )
                w_loss = self.criterion(weights * pred, weights * target)
                if self.reg_weight:
                    w_loss += self.reg_weight * self.regularize(pred, masks)
                w_loss = w_loss.item()
                print(
                    f"Inference test, weighted loss {w_loss:.4f}, target: {target.shape}, input: {input.shape}"
                )
                cs = self.cosine_similarity(pred, target)
                print(f"Inference test, cosine similarit:", cs)
                return loss, w_loss, cs

    def cosine_similarity(self, t0, t1):
        return torch.nn.CosineSimilarity(dim=-1)(t0, t1).mean(dim=0).tolist()
