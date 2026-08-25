# AE 异常检测项目 — 目录结构与处理形式方案

## Context

**问题**：当前要为 4096 长度复数 IQ 窗的 One-Class 异常检测项目搭代码骨架。主方案已定为 AE（靠重构误差判异常），验证三拆分必须遵守：train(正常) / val(unseen normal, 定阈值+验泛化) / test(unseen normal + 异常, 评分离)。

**为什么需要这份方案**：用户放入了一个开集识别(OSR)参考项目（位于 `D:\dabaidanzi\组内项目\项目\异常检测\算法结构\`），明确要求**不参考算法细节，只参考处理形式**——即"数据集 / 训练 / 测试 / 主代码 / 模型 / 评估 / 工具"如何分层与串联。参考项目已分析完毕，其形式特征：主入口脚本(argparse 配置内嵌) + 模块化文件分层(数据/模型/训练/评估/工具/可视化分离)，但分层不彻底（epoch 循环塞在 `main_worker`、阈值 `thr=0` 硬编码、known/unknown 二拆分）。

**预期产出**：一份在 `D:\dabaidanzi\组内项目\项目\异常检测\AE\` 落地的目录结构与文件分工方案，沿用参考项目的分层形式，修正其缺陷，并适配 One-Class AE 任务的特异需求（三拆分、阈值选择、unseen 泛化检验）。

## 参考项目的处理形式（OSR，仅参照形式）

参考目录：`D:\dabaidanzi\组内项目\项目\异常检测\算法结构\`

| 文件 | 职责 |
|------|------|
| `OSR.py` | 主入口：argparse 配置 + `build_model()` + `main_worker()` 编排全流程 |
| `HRRP_OSR.py` | 数据集定义：`Dataset` 子类 + 容器类读 CSV → 构建 train/test/out DataLoader |
| `train.py` | `train()` 单 epoch 循环函数（forward→loss→backward→step），不存模型 |
| `eval.py` | 评估函数 `test_1()`，算 acc/AUROC/precision/recall/f1/混淆矩阵 |
| `models.py` | 基线 `ConvNet` + `ResidualBlock` + `weights_init` |
| `models_NS_RFF.py` | 模型变体（按变体分文件） |
| `plot.py` | 可视化（t-SNE / feature 2D plot） |
| `preprocessing.py` | DSP 工具 |
| `utils.py` | `AverageMeter` / `Logger` / `save_networks` / `load_networks` |

**调用关系**：`OSR.py` (main, argparse) → import `HRRP_OSR` / `models` / `train` / `eval` / `utils` / `plot`；epoch 循环+模型保存在主入口 `main_worker`；`train.py` 只做单 epoch；`eval.py` 独立但被主循环每 epoch 调用。

## 本项目方案

### A. 目录树

```
D:\dabaidanzi\组内项目\项目\异常检测\AE\
├── main.py            # 主入口：argparse 配置 + build_model + 全流程编排
├── dataset.py         # IQ 窗数据集类 + 三拆分容器(train/val/test)
├── train.py           # 单 epoch 训练/验证循环(从主入口剥离)
├── threshold.py       # 基于验证集重构误差分布选阈值(percentile)
├── eval.py            # 测试评估: AUROC/F1/误差分布/unseen 泛化检验
├── loss.py            # 重构损失(MSE/L1/复数感知)
├── utils.py           # AverageMeter/Logger/save_load/set_seed
├── visualize.py       # 误差直方图/潜空间2D/ROC曲线
├── models/
│   ├── __init__.py    # 统一导出 AE，未来加 VAE
│   ├── base.py        # 共享组件(ResidualBlock/weights_init/复数卷积块)
│   └── ae.py          # Encoder + Decoder + AE
└── README.md          # 用法说明
```

### B. 文件职责详述

**main.py** — 主入口
- 暴露：`build_model(options) -> AE`、`main_worker(options)`
- 被 `__main__` 调用
- import：`dataset.IQData`、`models.ae.AE`、`train`、`threshold`、`eval`、`utils`、`visualize`、`loss`

**dataset.py** — 数据集与三拆分
- 暴露：`class IQWindowDataset(Dataset)`（读 .npy/.npz，输出 `(2, 4096)` 复数张量 + `is_anomaly` 标志）、`class IQData`（构造 train(正常) / val(unseen normal) / test(unseen normal + anomaly) 三个 DataLoader）
- 被 `main.py` import
- import：`torch` / `numpy` / `pathlib`

**models/ae.py** — AE 模型
- 暴露：`class Encoder(nn.Module)`、`class Decoder(nn.Module)`、`class AE(nn.Module)`（`forward(x) -> recon`、`encode(x) -> z`）
- 被 `build_model` import
- import：`models.base`
- 预留：未来 `models/vae.py` 复用 `base.py` 的块，不改 `ae.py`

**train.py** — 单 epoch 循环
- 暴露：`train_epoch(net, loader, optimizer, criterion, epoch, **opt) -> avg_loss`、`validate_epoch(net, loader, criterion, **opt) -> avg_loss`
- 被 `main_worker` import
- import：`utils.AverageMeter`

**threshold.py** — 阈值选择
- 暴露：`compute_errors(net, loader, **opt) -> np.ndarray`（逐样本重构误差）、`select_threshold(errors, method='percentile', percentile=95) -> float`
- 被 `main` import
- import：`torch` / `numpy`

**eval.py** — 测试评估
- 暴露：`evaluate(net, test_loader, threshold, **opt) -> dict`：返回 AUROC(异常为 positive)、F1@阈值、混淆矩阵、误差分布、`unseen_high_flag`(unseen normal 误差是否系统性偏高)
- 被 `main` import
- import：`sklearn.metrics`

**loss.py** — 重构损失
- 暴露：`class ReconLoss(nn.Module)`（MSE / L1 / 复数幅度感知）
- 被 `train` import
- import：`torch.nn`

**utils.py** — 工具
- 暴露：`AverageMeter`、`Logger`、`save_networks`、`load_networks`、`set_seed`
- 被多处 import

**visualize.py** — 可视化
- 暴露：`plot_error_histogram(train_err, val_err, test_err, thr)`、`plot_latent_2d(z, labels)`、`plot_roc(fpr, tpr)`
- 被 `main` import

### C. 主入口执行流

1. `argparse` 解析 → `options`；`set_seed`、GPU 设置
2. `IQData(...)` 构建 train / val / test 三 DataLoader
3. `build_model(options)` → AE；移至 GPU
4. 构造 `optimizer` / `scheduler` + `ReconLoss`
5. 训练循环：`for epoch: train_epoch → (每 N) validate_epoch → save_networks`
6. 训练结束 `load_networks` 取最佳/末次权重
7. `compute_errors(val_loader)` → `select_threshold` → `thr`
8. `evaluate(test_loader, thr)` → AUROC / F1 / 分布 / unseen 泛化检验
9. `visualize`：误差直方图 + 潜空间 + ROC
10. 保存 `results.csv` + 日志

### D. 配置项清单（argparse）

- **数据**：`--data-root`、`--train-npy`、`--val-npy`、`--test-normal-npy`、`--test-anomaly-npy`、`--seq-len`(4096)、`--input-form`(complex/real_imag)、`--batch-size`、`--num-workers`
- **模型**：`--model`(AE)、`--latent-dim`、`--base-channels`、`--depth`、`--bias`
- **训练**：`--lr`、`--weight-decay`、`--max-epoch`、`--scheduler`、`--stepsize`、`--loss`(mse/l1)、`--seed`、`--gpu`、`--use-cpu`
- **评估**：`--threshold-method`(percentile)、`--threshold-percentile`(95)、`--eval-freq`、`--eval-only`
- **路径**：`--out-dir`、`--save-dir`、`--resume`

### E. 与参考项目的差异说明

1. **epoch 循环移出主入口**：参考项目把 `for epoch` 塞在 `main_worker`，训练逻辑与编排耦合；本方案 `train.py` 只暴露 `train_epoch`，main 仅编排，便于单独测试训练步。
2. **阈值独立模块**：参考项目 `thr=0` 硬编码在 eval，无验证集依据；AE 异常判定完全依赖阈值，必须由 `threshold.py` 基于验证集(正常)误差分布(percentile)选，可复现、可调。
3. **三拆分替代二拆分**：参考项目 known/unknown 是类别二分；AE 需 train(正常) / val(unseen normal, 验泛化+定阈值) / test(unseen normal + anomaly, 评分离)，`IQData` 直接产出三个 loader，标签语义从类别改为 `is_anomaly`。
4. **评估面向 One-Class**：参考项目算多类 acc/AUROC；本方案以异常为 positive 算 AUROC、F1@选定阈值，并显式输出 `unseen_high_flag` 检验泛化(若 unseen normal 误差偏高则方案不成立)。
5. **models 改为包目录**：参考项目 `models.py` 单文件；本方案 `models/` 包内 `ae.py` 与未来 `vae.py` 并列、共享 `base.py`，加 VAE 不破坏结构与主入口。

## Critical Files for Implementation

落地时需创建的文件（按实现顺序）：

- `D:\dabaidanzi\组内项目\项目\异常检测\AE\utils.py` — 工具层，最先实现
- `D:\dabaidanzi\组内项目\项目\异常检测\AE\dataset.py` — 数据层，需用户提供数据组织方式后细化
- `D:\dabaidanzi\组内项目\项目\异常检测\AE\models\base.py` + `models\ae.py` + `models\__init__.py` — 模型层
- `D:\dabaidanzi\组内项目\项目\异常检测\AE\loss.py` — 损失层
- `D:\dabaidanzi\组内项目\项目\异常检测\AE\train.py` — 训练循环
- `D:\dabaidanzi\组内项目\项目\异常检测\AE\threshold.py` — 阈值选择
- `D:\dabaidanzi\组内项目\项目\异常检测\AE\eval.py` — 评估
- `D:\dabaidanzi\组内项目\项目\异常检测\AE\visualize.py` — 可视化
- `D:\dabaidanzi\组内项目\项目\异常检测\AE\main.py` — 主入口，最后组装

## 未决项（需要用户后续提供）

1. **数据组织方式**：.npy/.npz 是按"一个文件一窗"还是"一个文件多窗"？正常/异常/unseen normal 的文件物理如何区分（不同子目录 / 文件名前缀 / 标签 csv）？这决定 `IQWindowDataset` 的实现细节。
2. **IQ 数据维度**：原始数据是 `(2, 4096)` 实虚部双通道，还是 `complex64` 一维？是否需要做归一化（如 IQ 样本功率归一）？
3. **AE 架构偏好**：1D CNN（类似参考项目 ConvNet）还是 MLP？latent dim 倾向多少？

## Verification

实现完成后，用以下方式端到端验证：

1. **冒烟测试**：用小规模合成数据（10-20 个 4096 长度随机复数样本，标为 train/val/test-normal/test-anomaly）跑 `python main.py --max-epoch 2`，确认全流程无报错、能产出权重和 results.csv。
2. **三拆分正确性**：检查 `IQData` 输出的三个 loader 的样本标签语义——train/val 全为 `is_anomaly=0`，test 含 `is_anomaly=1`。
3. **阈值模块独立性**：单独 `python -c "from threshold import compute_errors, select_threshold; ..."` 跑一遍，确认阈值来自 val 集误差分布而非硬编码。
4. **泛化检验**：观察 `evaluate` 返回的 `unseen_high_flag`——若 val 集误差分布与 train 显著偏高（如均值 > 2× train 均值），方案不成立，需回看模型设计。
5. **可视化产出**：检查 `visualize` 产出的误差直方图——train/val/test-normal 应重叠，test-anomaly 应右移。
