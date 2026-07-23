import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class Dogs(Dataset):
    def __init__(self, df, img_dir, c2i, tfm):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.c2i = c2i
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.img_dir / f"{r.id}.jpg").convert("RGB")
        return self.tfm(img), self.c2i[r.breed]


p = argparse.ArgumentParser()
p.add_argument("--model", choices=["resnet50", "efficientnet_b2"], default="efficientnet_b2")
p.add_argument("--data-dir", type=Path, default=Path("dog-breed-identification"))
p.add_argument("--split", type=Path, default=Path("outputs/data_split.csv"))
p.add_argument("--out", type=Path, default=Path("outputs/optimized"))
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--batch-size", type=int, default=48)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--wd", type=float, default=2e-4)
p.add_argument("--mixup", type=float, default=0.2)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)
torch.backends.cudnn.benchmark = True

dev = torch.device("cuda")
df = pd.read_csv(a.split)
cls = sorted(df.breed.unique())
c2i = {c: i for i, c in enumerate(cls)}
size = 288 if a.model == "efficientnet_b2" else 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# RandAugment 和随机擦除增加图片变化，减轻小数据集过拟合
tfm_tr = transforms.Compose([
    transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
tfm_ev = transforms.Compose([
    transforms.Resize(int(size * 1.14)),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

dl = {}
for s in ["train", "validation", "test"]:
    part = df[df.split == s]
    ds = Dogs(part, a.data_dir / "train", c2i, tfm_tr if s == "train" else tfm_ev)
    dl[s] = DataLoader(
        ds,
        batch_size=a.batch_size,
        shuffle=s == "train",
        num_workers=a.workers,
        pin_memory=True,
        persistent_workers=True,
    )

ds = Dogs(df[df.split == "train"], a.data_dir / "train", c2i, tfm_ev)
dl["train_eval"] = DataLoader(
    ds,
    batch_size=a.batch_size,
    num_workers=a.workers,
    pin_memory=True,
    persistent_workers=True,
)

# 分类头加入 Dropout，主干保留 ImageNet 预训练参数
if a.model == "resnet50":
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    n = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(n, len(cls)))
    head = net.fc
else:
    net = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    n = net.classifier[1].in_features
    net.classifier[0] = nn.Dropout(0.35)
    net.classifier[1] = nn.Linear(n, len(cls))
    head = net.classifier

net = net.to(dev)
head_ids = {id(x) for x in head.parameters()}
backbone = [x for x in net.parameters() if id(x) not in head_ids]

# 分类头学习率较高，主干使用十分之一学习率避免破坏预训练特征
opt = torch.optim.AdamW([
    {"params": backbone, "lr": a.lr / 10},
    {"params": head.parameters(), "lr": a.lr},
], weight_decay=a.wd)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda")

save_dir = a.out / a.model
save_dir.mkdir(parents=True, exist_ok=True)
hist = []
best = -1
best_ep = 0
t0 = time.time()

for ep in range(1, a.epochs + 1):
    net.train()
    tr_loss = tr_ok = tr_n = 0

    for x, y in dl["train"]:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        # 一半批次使用 Mixup，减少模型记忆单张训练图片
        if random.random() < 0.5:
            lam = np.random.beta(a.mixup, a.mixup)
            idx = torch.randperm(len(x), device=dev)
            x = lam * x + (1 - lam) * x[idx]
            y2 = y[idx]
        else:
            lam, y2 = 1.0, y

        with torch.autocast("cuda", dtype=torch.float16):
            z = net(x)
            loss = lam * loss_fn(z, y) + (1 - lam) * loss_fn(z, y2)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tr_loss += loss.item() * len(y)
        pred = z.argmax(1)
        tr_ok += (lam * (pred == y).sum() + (1 - lam) * (pred == y2).sum()).item()
        tr_n += len(y)

    net.eval()
    va_loss = va_ok = va_n = 0
    with torch.inference_mode():
        for x, y in dl["validation"]:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                # 原图与水平翻转图取平均，降低单次裁剪带来的波动
                z = (net(x) + net(torch.flip(x, dims=[3]))) / 2
                loss = loss_fn(z, y)
            va_loss += loss.item() * len(y)
            va_ok += (z.argmax(1) == y).sum().item()
            va_n += len(y)

    tr_loss, tr_acc = tr_loss / tr_n, tr_ok / tr_n
    va_loss, va_acc = va_loss / va_n, va_ok / va_n
    sch.step()
    hist.append([ep, tr_loss, tr_acc, va_loss, va_acc])
    print(f"{a.model} [{ep:02d}/{a.epochs}] train={tr_acc:.4f} val={va_acc:.4f}", flush=True)

    if va_acc > best:
        best, best_ep = va_acc, ep
        torch.save({
            "model": a.model,
            "state": net.state_dict(),
            "classes": cls,
            "size": size,
        }, save_dir / "best.pt")

pd.DataFrame(
    hist,
    columns=["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"],
).to_csv(save_dir / "history.csv", index=False)

ckpt = torch.load(save_dir / "best.pt", weights_only=False)
net.load_state_dict(ckpt["state"])
res = {"model": a.model, "best_epoch": best_ep}

for s, key in [("train_eval", "train"), ("validation", "validation"), ("test", "test")]:
    net.eval()
    total_loss = ok = count = 0
    with torch.inference_mode():
        for x, y in dl[s]:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = (net(x) + net(torch.flip(x, dims=[3]))) / 2
                loss = loss_fn(z, y)
            total_loss += loss.item() * len(y)
            ok += (z.argmax(1) == y).sum().item()
            count += len(y)
    res[f"{key}_loss"] = total_loss / count
    res[f"{key}_accuracy"] = ok / count

res["minutes"] = (time.time() - t0) / 60
with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(res, flush=True)
