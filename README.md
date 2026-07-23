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

服务器返回的基线训练记录保存在 `results/baseline/`。其中包含三个模型的逐轮历史、最终指标和完整训练日志，不包含体积较大的模型权重。

## 优化实验

基线模型完成后，运行优化版 ResNet50 和 EfficientNet-B2：

```bash
PYTHONUNBUFFERED=1 nohup python -u run_optimized.py > optimized.log 2>&1 &
tail -f optimized.log
```

优化实验使用更强数据增强、Mixup、Dropout、分层学习率和水平翻转 TTA。结果保存在：

```text
outputs/optimized/resnet50/
outputs/optimized/efficientnet_b2/
```

使用优化模型生成提交文件：

```bash
python predict.py \
  --checkpoint outputs/optimized/efficientnet_b2/best.pt \
  --output outputs/submission_efficientnet_b2.csv
```

补跑 EfficientNet-B2 消融实验：

```bash
PYTHONUNBUFFERED=1 nohup python -u run_ablation.py > ablation.log 2>&1 &
tail -f ablation.log
```

`basic` 使用普通增强和统一学习率；`strong` 加入 RandAugment、Random Erasing 和 Mixup；已有的 `full` 进一步加入 Dropout、分层学习率和 TTA。
