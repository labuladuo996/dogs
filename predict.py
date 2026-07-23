import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class TestDogs(Dataset):
    # 官方测试集没有标签，只返回经过预处理的图片
    def __init__(self, ids, img_dir, tfm):
        self.ids = ids
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = Image.open(self.img_dir / f"{img_id}.jpg").convert("RGB")
        return self.tfm(img)


p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--data-dir", type=Path, default=Path("dog-breed-identification"))
p.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--workers", type=int, default=8)
a = p.parse_args()

dev = torch.device("cuda")
ckpt = torch.load(a.checkpoint, weights_only=False)
cls = ckpt["classes"]
sample = pd.read_csv(a.data_dir / "sample_submission.csv")

# 使用与验证集相同的固定预处理
tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
ds = TestDogs(sample.id.tolist(), a.data_dir / "test", tfm)
dl = DataLoader(
    ds,
    batch_size=a.batch_size,
    num_workers=a.workers,
    pin_memory=True,
    persistent_workers=True,
)

# 按 checkpoint 中的名称重建网络，再载入训练好的参数
if ckpt["model"] == "resnet18":
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(cls))
elif ckpt["model"] == "resnet50":
    net = models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(cls))
elif ckpt["model"] == "vgg16_bn":
    net = models.vgg16_bn(weights=None)
    net.classifier[6] = nn.Linear(net.classifier[6].in_features, len(cls))
else:
    net = models.googlenet(weights=None, init_weights=False)
    net.fc = nn.Linear(net.fc.in_features, len(cls))
    net.aux_logits = False
    net.aux1 = None
    net.aux2 = None

net.load_state_dict(ckpt["state"])
net = net.to(dev).eval()

# 分批预测并保存每个犬种的 softmax 概率
pred = []
with torch.inference_mode():
    for x in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            z = net(x.to(dev, non_blocking=True))
            z = z.logits if hasattr(z, "logits") else z
        pred.append(torch.softmax(z, 1).cpu())

# checkpoint 和 Kaggle 模板的类别顺序可能不同，需要重新排列
pred = torch.cat(pred).numpy()
c2i = {c: i for i, c in enumerate(cls)}
sub = pd.DataFrame(pred[:, [c2i[c] for c in sample.columns[1:]]], columns=sample.columns[1:])
sub.insert(0, "id", sample.id)
a.output.parent.mkdir(parents=True, exist_ok=True)

sub.to_csv(a.output, index=False)
