import argparse
import json
from pathlib import Path

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
p.add_argument("--checkpoint", type=Path, default=Path("outputs/optimized/efficientnet_b2/best.pt"))
p.add_argument("--data-dir", type=Path, default=Path("dog-breed-identification"))
p.add_argument("--split", type=Path, default=Path("outputs/data_split.csv"))
p.add_argument("--output", type=Path, default=Path("outputs/optimized/efficientnet_b2/tta_comparison.json"))
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--workers", type=int, default=8)
a = p.parse_args()

dev = torch.device("cuda")
ckpt = torch.load(a.checkpoint, weights_only=False)
cls = ckpt["classes"]
c2i = {c: i for i, c in enumerate(cls)}
size = ckpt.get("size", 288)
df = pd.read_csv(a.split)

tfm = transforms.Compose([
    transforms.Resize(int(size * 1.14)),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

net = models.efficientnet_b2(weights=None)
net.classifier[0] = nn.Dropout(0.35)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(cls))
net.load_state_dict(ckpt["state"])
net = net.to(dev).eval()

res = {}
for s in ["validation", "test"]:
    ds = Dogs(df[df.split == s], a.data_dir / "train", c2i, tfm)
    dl = DataLoader(
        ds,
        batch_size=a.batch_size,
        num_workers=a.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    no_tta_ok = tta_ok = count = 0
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z1 = net(x)
                z2 = net(torch.flip(x, dims=[3]))
                z_tta = (z1 + z2) / 2
            no_tta_ok += (z1.argmax(1) == y).sum().item()
            tta_ok += (z_tta.argmax(1) == y).sum().item()
            count += len(y)

    no_tta_acc = no_tta_ok / count
    tta_acc = tta_ok / count
    res[s] = {
        "without_tta": no_tta_acc,
        "with_tta": tta_acc,
        "improvement": tta_acc - no_tta_acc,
    }
    print(s, res[s], flush=True)

a.output.parent.mkdir(parents=True, exist_ok=True)
with open(a.output, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
