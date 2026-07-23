# ImageNet Dogs 实验代码

在云服务器下载数据：

```bash
export KAGGLE_API_TOKEN="你的 Kaggle Token"
python download_data.py
```

数据放在 `dog-breed-identification/`：

```text
dog-breed-identification/
├── train/
├── test/
├── labels.csv
└── sample_submission.csv
```

先固定训练集、验证集和内部测试集：

```bash
python data_prepare.py
```

划分比例为 70%/15%/15%，四个模型共用 `outputs/data_split.csv`。

训练单个模型：

```bash
python train.py --model resnet18
python train.py --model resnet50
python train.py --model vgg16_bn
python train.py --model googlenet
```

V100 32GB 默认使用 `batch_size=64`、8 个数据加载进程和 FP16 混合精度。显存有余量时可以把 batch size 调到 96 或 128，VGG16-BN 建议先保持 64。

一次训练四个模型：

```bash
python run_all.py
```

每个模型保存三个文件：

```text
outputs/models/模型名/
├── best.pt
├── history.csv
└── metrics.json
```

`metrics.json` 记录最佳 epoch 以及训练集、验证集、内部测试集的 loss 和 accuracy。Kaggle 官方测试集没有标签，因此不能计算准确率。

生成 Kaggle 提交文件：

```bash
python predict.py --checkpoint outputs/models/resnet18/best.pt
```

当前代码只负责训练、评估和预测。混淆矩阵、训练曲线、模型对比图和实验结果分析暂未加入。
