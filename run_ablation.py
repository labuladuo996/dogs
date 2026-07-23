import subprocess
import sys


# full 模式已经训练完成，只补跑基础版和强增强版
for mode in ["basic", "strong"]:
    subprocess.run([
        sys.executable,
        "-u",
        "train_optimized.py",
        "--model",
        "efficientnet_b2",
        "--mode",
        mode,
    ])
