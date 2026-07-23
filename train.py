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
    # df 保存图片 id 和标签，真正读取图片放在 __getitem__ 中
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
p.add_argument("--model", choices=["resnet18", "resnet50", "vgg16_bn", "googlenet"], default="resnet18")
p.add_argument("--data-dir", type=Path, default=Path("dog-breed-identification"))
p.add_argument("--split", type=Path, default=Path("outputs/data_split.csv"))
p.add_argument("--out", type=Path, default=Path("outputs/models"))
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--freeze", action="store_true")
a = p.parse_args()

# 固定随机状态，保证不同模型使用相同的数据顺序和初始化条件
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)
torch.backends.cudnn.benchmark = True

dev = torch.device("cuda")
df = pd.read_csv(a.split)

# 类别按字母排序后映射到 0-119，预测时仍使用这份顺序
cls = sorted(df.breed.unique())
c2i = {c: i for i, c in enumerate(cls)}

# ImageNet 预训练权重要求使用相同的均值和标准差
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# 随机裁剪、翻转和颜色扰动只用于训练集，减少过拟合
tfm_tr = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# 验证集和测试集使用固定变换，保证每次评估结果一致
tfm_ev = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# 为三个集合分别建立 DataLoader，训练集打乱顺序
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

# 训练结束后用无随机增强的数据重新计算训练集准确率
ds = Dogs(df[df.split == "train"], a.data_dir / "train", c2i, tfm_ev)
dl["train_eval"] = DataLoader(
    ds,
    batch_size=a.batch_size,
    num_workers=a.workers,
    pin_memory=True,
    persistent_workers=True,
)

# 加载 ImageNet 预训练权重，只替换原模型的分类层
# 新分类层输出 120 个值，对应 120 个犬种
if a.model == "resnet18":
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, len(cls))
    head = net.fc
elif a.model == "resnet50":
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, len(cls))
    head = net.fc
elif a.model == "vgg16_bn":
    net = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    net.classifier[6] = nn.Linear(net.classifier[6].in_features, len(cls))
    head = net.classifier[6]
else:
    net = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, len(cls))
    net.aux_logits = False
    net.aux1 = None
    net.aux2 = None
    head = net.fc

# 冻结只训练最后的分类层
if a.freeze:
    for x in net.parameters():
        x.requires_grad = False
    for x in head.parameters():
        x.requires_grad = True

net = net.to(dev)

# 标签平滑降低模型对单一类别的过度自信
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW 同时完成参数更新和权重衰减
opt = torch.optim.AdamW(
    [x for x in net.parameters() if x.requires_grad], lr=a.lr, weight_decay=a.wd
)

# 余弦退火在训练后期逐渐减小学习率
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

# V100 使用 FP16 混合精度
scaler = torch.amp.GradScaler("cuda")

save_dir = a.out / a.model
save_dir.mkdir(parents=True, exist_ok=True)
hist = []
best = -1
best_ep = 0
t0 = time.time()

# 每个 epoch 先训练一次，再在验证集上评估一次
for ep in range(1, a.epochs + 1):
    net.train()
    tr_loss = tr_ok = tr_n = 0

    for x, y in dl["train"]:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        # 前向传播使用 FP16，损失缩放避免小梯度下溢
        with torch.autocast("cuda", dtype=torch.float16):
            z = net(x)
            z = z.logits if hasattr(z, "logits") else z
            loss = loss_fn(z, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tr_loss += loss.item() * len(y)
        tr_ok += (z.argmax(1) == y).sum().item()
        tr_n += len(y)

    # 验证阶段关闭梯度
    net.eval()
    va_loss = va_ok = va_n = 0
    with torch.inference_mode():
        for x, y in dl["validation"]:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = net(x)
                z = z.logits if hasattr(z, "logits") else z
                loss = loss_fn(z, y)
            va_loss += loss.item() * len(y)
            va_ok += (z.argmax(1) == y).sum().item()
            va_n += len(y)

    tr_loss, tr_acc = tr_loss / tr_n, tr_ok / tr_n
    va_loss, va_acc = va_loss / va_n, va_ok / va_n
    sch.step()
    hist.append([ep, tr_loss, tr_acc, va_loss, va_acc])
    print(f"{a.model} [{ep:02d}/{a.epochs}] train={tr_acc:.4f} val={va_acc:.4f}")

    # 测试集不参与模型选择，只保存验证集准确率最高的权重
    if va_acc > best:
        best, best_ep = va_acc, ep
        torch.save({
            "model": a.model,
            "state": net.state_dict(),
            "classes": cls,
        }, save_dir / "best.pt")

# history.csv 保留每轮数据
pd.DataFrame(
    hist,
    columns=["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"],
).to_csv(save_dir / "history.csv", index=False)

ckpt = torch.load(save_dir / "best.pt", weights_only=False)
net.load_state_dict(ckpt["state"])
res = {"model": a.model, "best_epoch": best_ep}

# 重新载入最佳权重，再统一计算训练、验证和内部测试结果
# 官方 test 没有标签，不参与这里的准确率计算
for s, key in [("train_eval", "train"), ("validation", "validation"), ("test", "test")]:
    net.eval()
    total_loss = ok = n = 0
    with torch.inference_mode():
        for x, y in dl[s]:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z = net(x)
                z = z.logits if hasattr(z, "logits") else z
                loss = loss_fn(z, y)
            total_loss += loss.item() * len(y)
            ok += (z.argmax(1) == y).sum().item()
            n += len(y)
    res[f"{key}_loss"] = total_loss / n
    res[f"{key}_accuracy"] = ok / n

res["minutes"] = (time.time() - t0) / 60

with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(res)
