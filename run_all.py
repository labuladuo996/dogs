import subprocess
import sys


# 逐个启动独立训练进程，前一个模型结束后再训练下一个
# 每个模型都读取同一份 data_split.csv，结果保存在各自目录
for m in ["resnet18", "resnet50", "vgg16_bn", "googlenet"]:
    subprocess.run([sys.executable, "train.py", "--model", m])
