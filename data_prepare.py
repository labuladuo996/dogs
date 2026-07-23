import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


p = argparse.ArgumentParser()
p.add_argument("--data-dir", type=Path, default=Path("dog-breed-identification"))
p.add_argument("--output", type=Path, default=Path("outputs/data_split.csv"))
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

# 从有标签图片中另划内部测试集
df = pd.read_csv(a.data_dir / "labels.csv")

# 第一次划分保留 70% 训练数据，剩余 30% 再平分
# 保证 120 个犬种在三个集合中的比例一致
tr, rest = train_test_split(
    df, test_size=0.30, random_state=a.seed, stratify=df.breed
)
va, te = train_test_split(
    rest, test_size=0.50, random_state=a.seed, stratify=rest.breed
)

tr = tr.assign(split="train")
va = va.assign(split="validation")
te = te.assign(split="test")

# 保存图片 id、犬种和所属集合，后续四个模型读取同一份划分
df = pd.concat([tr, va, te]).sort_values("id")

a.output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.output, index=False)

print(df.split.value_counts())
print("classes:", df.breed.nunique())
