# LWST-HGN: A Learnable Wavelet-Enhanced Spatio-Temporal Hypergraph Network for Soft Sensing

## AirQuality 开源代码说明

该仓库是 **UCI Air Quality / C6H6(GT)** 任务的最佳结果开源版本。训练逻辑来自最终实验代码，主要进行了文件结构、命名、默认参数和说明文档整理，没有改动最佳实验所依赖的核心训练流程。

## 直接复现

1. 将 `AirQualityUCI.csv` 放到：

```text
data/AirQualityUCI.csv
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行：

```bash
python -u train_airquality_c6h6.py --data_path data/AirQualityUCI.csv --gpu 0
```

或者：

```bash
bash run_best.sh 0 data/AirQualityUCI.csv
```

## 最佳配置

- 目标：`C6H6_GT`
- 时间窗：3
- 连续时间划分：60% / 20% / 20%
- 仅用训练集 Pearson 相关性筛选，删除相关性最低 3 个变量
- 参考运行删除：`AH`, `RH`, `T`
- 静态超图：`k=3`
- MS-GTC：卷积核 `{3,5,7}`，gated fusion
- 可学习小波：db2 初始化
- 残差增强尺度：`alpha=0.055`
- 低频补偿比例：`rho=0.5`
- 能量约束：`1e-4`
- Huber loss
- 学习率：`1.5e-3`
- weight decay：`1e-4`
- batch size：128
- dropout：0.1
- seed：3407

## 参考结果

| Split | NMAE | NRMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Validation | 0.427638 | 0.926696 | 4.363315 | 0.995141 |
| Test | **0.768057** | **1.074252** | 7.656405 | **0.995024** |

参考运行最佳 epoch 为 147。不同 GPU、CUDA、cuDNN 或 PyTorch 版本可能产生轻微数值差异。
