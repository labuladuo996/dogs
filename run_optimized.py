import subprocess
import sys


# 先补齐优化版 ResNet50，再训练更适合细粒度分类的 EfficientNet-B2
for m in ["resnet50", "efficientnet_b2"]:
    subprocess.run([sys.executable, "-u", "train_optimized.py", "--model", m])
