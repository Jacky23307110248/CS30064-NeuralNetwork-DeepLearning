# PJ2（Neural Network and Deep Learning）

课程 Project 2：CIFAR-10 上的 ResNet 训练与消融（Task 1）、VGG-A 批归一化与优化 landscape 分析（Task 2）。

- 报告：`report.ipynb`（导出 `report.pdf` 提交 eLearning）
- 作业说明：`project_2_2026.md`

## 代码与数据下载

**代码仓库**：https://github.com/Jacky23307110248/CS30064-NeuralNetwork-DeepLearning/tree/main/PJ2

**数据集与模型权重（Google Drive）**：https://drive.google.com/drive/folders/13fio3sFs1qT7FBt6_mwabVRHcdk42U1d?usp=sharing

GitHub 仓库中**不上传** `data/cifar-10-batches-py/` 与整个 `outputs/`（体积过大）。`data/` 下课程提供的 `loaders.py`、`__init__.py` 以及可选的 `sample.png` 会随代码一并提交。本地运行前请从网盘下载 CIFAR-10 数据并解压到 `PJ2/data/cifar-10-batches-py/`，将训练产物解压到 `PJ2/outputs/`。

## 环境

- Python 3，PyTorch，`torchvision`，`tqdm`，`numpy`，`matplotlib`

## 目录结构

```
PJ2/
├── project_2_2026.md
├── report.ipynb
├── pic/                          # 报告插图（由 plot_report.py 生成）
├── README.md
├── data/
│   ├── __init__.py               # 包初始化
│   ├── loaders.py                # CIFAR-10 DataLoader；`python data/loaders.py` 可预览并生成 sample.png
│   ├── sample.png                # 样例图（由 loaders.py 生成，可选提交）
│   └── cifar-10-batches-py/      # CIFAR-10 数据（网盘下载，不上传 GitHub）
├── outputs/
│   ├── CIFAR10_NeuroNet/         # Task 1 训练输出（网盘 outputs）
│   └── VGG_BatchNorm/            # Task 2 训练输出（网盘 outputs）
└── codes/
    ├── plot_report.py            # 离线绘图（读 outputs → 写 pic/）
    ├── CIFAR10_NeuroNet/         # Task 1：CifarResNet 训练与消融
    └── VGG_BatchNorm/            # Task 2：VGG-A / VGG-A+BN
```

## 数据划分（两任务共用）

- 官方 50k train 按 `seed=2020` 划分为 **45000 train / 5000 val**
- 官方 10k test 不参与训练与选优，仅在 `best.pt` 上评估一次
- 训练集：随机裁剪 + 水平翻转；验证/测试：仅归一化

---

## Task 1：`codes/CIFAR10_NeuroNet`

CIFAR-10 上的 **CifarResNet**（残差 + BatchNorm）。默认 **200 epoch、无早停**，验证集最优权重存为 `best.pt`。

在 `codes/CIFAR10_NeuroNet` 下执行：

```bash
# 消融 / 对照（refine 计划）
python main.py refine --plan baseline   # 基线 1 组
python main.py refine --plan optim      # 优化器网格
python main.py refine --plan width      # 通道宽度
python main.py refine --plan loss       # 损失函数
python main.py refine --plan act        # 激活函数
python main.py refine --plan all        # 全部
python main.py refine --plan all --force #强制重跑全部

# 组合实验（CutMix / Mixup + SGD + 宽度）
python main.py combine --recipe all
python main.py combine --recipe cutmix_w96_sgd0.1
python main.py combine --recipe cutmix_w96_sgd0.05
python main.py combine --recipe cutmix_w64_sgd0.1
python main.py combine --recipe mixup_w64_sgd0.1

# 损失曲面（step-loss 包络，可选学习率）
python main.py analysis --task landscape_sgd
python main.py analysis --task landscape_sgd --lrs 0.1 0.2

# 对已有 run 重算 test
python main.py eval --run-dir ../../outputs/CIFAR10_NeuroNet/<run_folder>
```

**输出目录**：`outputs/CIFAR10_NeuroNet/{exp_type}-{hyper_tag}-{时间戳}/`

| 文件 | 说明 |
|------|------|
| `config.json` | 超参快照 |
| `curves.npz` | epoch 级 loss / acc |
| `best.pt` | 验证集最优权重 |
| `results.json` | 最优 val + 官方 test 指标 |
| `steps.npz` | 仅 `analysis --task landscape_sgd`：每 step 的 train loss 等（用于损失曲面包络图） |

汇总表：`outputs/CIFAR10_NeuroNet/experiment_summary.csv`

**combine 配方**：`cutmix_w96_sgd0.1`、`cutmix_w96_sgd0.05`、`cutmix_w64_sgd0.1`、`mixup_w64_sgd0.1`

**基线（refine）**：AdamW `lr=1e-3`，`weight_decay=5e-4`，通道 `64-128-256-512`，CosineAnnealingLR。

模型定义：`codes/CIFAR10_NeuroNet/models/model.py`（`channels`、`activation`、`dropout` 等）。

---

## Task 2：`codes/VGG_BatchNorm`

**VGG_A** 与 **VGG_A_BatchNorm** 对比；多学习率 loss 包络与梯度距离扫描。

在 `codes/VGG_BatchNorm` 下执行：

```bash
# 2 次 bn_compare（AdamW, 200 epoch）
python main.py compare

# 8 次 loss_landscape（4 个固定 lr, 100 epoch, steps.npz）
python main.py landscape

# 2 次 grad_probe（沿梯度方向多距离扫描, grad_sweep.npz）
python main.py grad
python main.py grad --lr 1e-3 --epochs 100

# compare + landscape（不含 grad）
python main.py all

python main.py eval --run-dir ../../outputs/VGG_BatchNorm/<run_folder>
```

**输出目录**：`outputs/VGG_BatchNorm/<model>_<exp_type>_<hyper_tag>_seed2020_<timestamp>/`

`exp_type` 三类：`bn_compare`（`compare`）、`loss_landscape`（`landscape`）、`grad_probe`（`grad`）。

| 文件 | 说明 | 出现的实验 |
|------|------|------------|
| `config.json` | 超参与实验类型 | 全部 |
| `split.npz` | train/val 索引 | 全部 |
| `train.log` | 每 epoch 一行文本日志 | 全部 |
| `curves.npz` | epoch 级 train/val loss/acc | 全部 |
| `steps.npz` | 每 step：train_loss、lr、grad_norm、update_norm | 全部 |
| `best.pt` | val 最优权重 | 全部 |
| `results.json` | 汇总 + test 指标 | 全部 |
| `grad_probe.npz` | 每 step 单距离梯度探测（旧版 landscape 辅助量） | 仅 `loss_landscape` |
| `grad_sweep.npz` | 每 step 多距离 \(\|g'-g\|\) 与 loss 扫描 | 仅 `grad_probe`（`python main.py grad`） |

汇总表：`outputs/VGG_BatchNorm/experiment_summary.csv`

- `VGG_A`：Conv → ReLU → Pool
- `VGG_A_BatchNorm`：Conv → BN → ReLU → Pool

---

## 报告插图：`codes/plot_report.py`

离线绘图脚本，**不重新训练**。默认从项目根目录下的 `outputs/CIFAR10_NeuroNet` 与 `outputs/VGG_BatchNorm` 读取数据——即网盘 **outputs** 解压到 `PJ2/outputs/` 后的内容（`curves.npz`、`best.pt`、`steps.npz`、`grad_sweep.npz` 等）。脚本会按各 figure 的配置自动匹配对应 run 目录（部分图使用 `plot_report.py` 内写死的默认 run 路径，与网盘包一致）；若本地目录名不同，可用 `--cifar-root`、`--vgg-root` 或 `--grad-vgg-a-run-dir` 等参数指定。

在项目根目录 `PJ2` 下执行，输出写入 `pic/`：

```bash
python codes/plot_report.py all
python codes/plot_report.py 1_3_2 1_9_2 2_3_2 2_4_2_loss_landscape
python codes/plot_report.py 2_3_3_feature_maps 2_3_3_activation_stats
```

常用 figure id：`1_3_2` … `1_9_2`、`1_10_2_kernels`、`1_10_2_confmat`、`2_3_2`、`2_4_2_loss_landscape`、`2_4_2_grad_predictability_landscape` 等（完整列表见 `plot_report.py` 文件头注释）。

VGG 损失曲面包络也可在 `codes/VGG_BatchNorm` 下运行 `python plot_loss_landscape.py`（等价读取 8 次 landscape 的 `steps.npz`）。

---

## 跳过已完成 run

两包均会扫描 `outputs/<包名>/`：若同前缀目录中 `config.json`、`best.pt`、`curves.npz`、`results.json` 齐全且配置与当前任务一致则跳过；否则可 `--force` 重训。
