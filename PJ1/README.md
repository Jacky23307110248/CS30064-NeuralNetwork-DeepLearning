# PJ1 (Neural Network and Deep Learning)

本目录对应 Project 1（MNIST: MLP vs CNN）的当前版本代码与报告。

## 1) 仓库提交说明

按课程要求，GitHub 仓库中**不上传**以下大文件目录：

- `codes/dataset/`（数据集）
- `codes/best_models/`（训练好的权重）

但本项目运行时**期望存在这两个目录**。因此请按下文的“期望目录结构”组织文件。

## 2) 期望目录结构（运行时）

```text
PJ1/
├─ project_1.md
├─ report.ipynb
├─ report.html
├─ report.pdf
├─ README.md
└─ codes/
   ├─ new_test_train.py
   ├─ test_train.py
   ├─ test_model.py
   ├─ RobustnessAnalysis.py
   ├─ RobustnessAnalysis.json
   ├─ ErrorAnalysis.py
   ├─ exp_artifacts.py
   ├─ dataset_explore.ipynb
   ├─ weight_visualization.py
   ├─ mynn/
   │  ├─ __init__.py
   │  ├─ op.py
   │  ├─ models.py
   │  ├─ optimizer.py
   │  ├─ lr_scheduler.py
   │  ├─ runner.py
   │  └─ metric.py
   ├─ draw_tools/
   │  ├─ draw.py
   │  └─ plot.py
   ├─ figs/
   │  └─ .../<某次实验>/
   │     └─ val_test_curves.png  # 该实验的 train/dev/test loss-accuracy 曲线图
   ├─ traininglogs/
   │  └─ .../<某次实验>/
   │     ├─ experiment.json      # 实验配置与元信息（模型、preset、超参数、路径等）
   │     ├─ training_log.txt     # 训练过程文本日志（关键设置与结果摘要）
   │     └─ curves.npz           # 曲线原始数组：train_loss, train_scores（启用评估的轮次还含 dev_loss, dev_scores，test_loss, test_scores）
   ├─ ErrorAnalysisDoc/     # 混淆矩阵与错分样例图
   ├─ dataset/              # 本地运行需要（不上传 GitHub）
   │  └─ MNIST/
   │     ├─ train-images-idx3-ubyte.gz
   │     ├─ train-labels-idx1-ubyte.gz
   │     ├─ t10k-images-idx3-ubyte.gz
   │     └─ t10k-labels-idx1-ubyte.gz
   └─ best_models/          # 本地运行需要（不上传 GitHub）
      ├─ mlp/
      └─ cnn/
```

报告文件说明：

- `report.ipynb`：报告源文件（可继续编辑）。
- `report.html`：Notebook 导出的网页版本，便于快速预览。
- `report.pdf`：最终提交到 eLearning 的报告版本。

## 3) best_models 下载位置与放置方式

`best_models` 已上传至 ModelScope：

- [https://modelscope.cn/models/JackYHon/CS30064-PJ1/tree/master/best_models](https://modelscope.cn/models/JackYHon/CS30064-PJ1/tree/master/best_models)

下载后请保持目录结构不变，并放到：

- `PJ1/codes/best_models/`

即保证存在类似路径：

- `codes/best_models/mlp/baseline/best_model.pickle`
- `codes/best_models/cnn/baseline/best_model.pickle`

## 4) 运行命令（完整配置）

以下命令均在 `codes/` 目录执行。

### 4.1 Baseline

```bash
python new_test_train.py --model MLP --preset baseline
python new_test_train.py --model CNN --preset baseline
```

### 4.2 Optimization（Momentum / StepLR / ExponentialLR）

```bash
# Momentum
python new_test_train.py --model MLP --preset momentum
python new_test_train.py --model CNN --preset momentum

# StepLR
python new_test_train.py --model MLP --preset lr_step
python new_test_train.py --model CNN --preset lr_step

# ExponentialLR
python new_test_train.py --model MLP --preset lr_exp
python new_test_train.py --model CNN --preset lr_exp
```

### 4.3 Regularization（L2 / Early Stop / L2+Early Stop）

```bash
# L2
python new_test_train.py --model MLP --preset reg_l2
python new_test_train.py --model CNN --preset reg_l2

# Early Stop
python new_test_train.py --model MLP --preset reg_early_stop
python new_test_train.py --model CNN --preset reg_early_stop

# L2 + Early Stop
python new_test_train.py --model MLP --preset reg_l2_early_stop
python new_test_train.py --model CNN --preset reg_l2_early_stop
```

### 4.4 可选：手动指定超参数（不走 preset）

```bash
# 常数学习率
python new_test_train.py --model CNN --optimizer_setting constant --lr 0.01

# 仅动量
python new_test_train.py --model CNN --optimizer_setting momentum --momentum 0.9 --lr 0.01

# 仅学习率调度（StepLR）
python new_test_train.py --model CNN --optimizer_setting lr_schedule --scheduler step --step_size 2000 --gamma 0.5 --lr 0.01

# 仅学习率调度（ExponentialLR）
python new_test_train.py --model CNN --optimizer_setting lr_schedule --scheduler exponential --gamma 0.9995 --lr 0.01

# 动量 + 学习率调度
python new_test_train.py --model CNN --optimizer_setting momentum_lr_schedule --scheduler step --momentum 0.9 --step_size 2000 --gamma 0.5 --lr 0.01
```

### 4.5 评估与分析脚本

```bash
# 使用 baseline 权重评估（默认路径）
python test_model.py --model MLP
python test_model.py --model CNN

# 指定任意实验权重评估
python test_model.py --model MLP --ckpt ./best_models/mlp/Optimization/momentum/best_model.pickle
python test_model.py --model CNN --ckpt ./best_models/cnn/Regularization/l2/best_model.pickle

# 如需自定义测试集路径
python test_model.py --model CNN --ckpt ./best_models/cnn/baseline/best_model.pickle --test_images ./dataset/MNIST/t10k-images-idx3-ubyte.gz --test_labels ./dataset/MNIST/t10k-labels-idx1-ubyte.gz

# 鲁棒性分析
python RobustnessAnalysis.py

# 错误分析与可视化
python ErrorAnalysis.py
```

`test_model.py` 默认会从 `./best_models/<mlp|cnn>/baseline/best_model.pickle` 加载权重；当评估其他实验配置时，请通过 `--ckpt` 显式指定权重路径。

## 5) 主要 Python 文件功能说明

- `codes/new_test_train.py`：统一训练入口；支持 MLP/CNN、不同优化器/学习率策略、L2、早停，并自动保存曲线、日志与权重到对应目录。
- `codes/test_train.py`：基础训练脚本（课程原始流程），用于快速跑通或对照基线。（已废除）
- `codes/test_model.py`：加载已保存模型并在测试集上计算 accuracy。
- `codes/RobustnessAnalysis.py`：对测试集施加高斯噪声并评估鲁棒性，输出统计结果（如均值/方差）到 `RobustnessAnalysis.json`。
- `codes/ErrorAnalysis.py`：生成混淆矩阵与典型错分样本可视化，输出到 `codes/ErrorAnalysisDoc/`。
- `codes/exp_artifacts.py`：实验产物管理工具；负责组织 `figs/`、`traininglogs/`、`best_models/` 的路径与落盘逻辑。
- `codes/weight_visualization.py`：可视化模型权重（可用于附加分析）。
- `codes/hyperparameter_search.py`：超参数搜索占位/扩展脚本（当前版本中使用较少）。

`codes/mynn/` 下核心模块：

- `codes/mynn/op.py`：基础算子实现（Linear、conv2D、激活、损失等）及其前向/反向传播。
- `codes/mynn/models.py`：模型结构定义（`Model_MLP`、`Model_CNN`）与模型保存/加载。
- `codes/mynn/optimizer.py`：优化器实现（如 SGD、MomentGD）。
- `codes/mynn/lr_scheduler.py`：学习率调度实现（如 StepLR、ExponentialLR）。
- `codes/mynn/runner.py`：训练循环与评估流程封装（日志、验证、测试、保存 best model）。
- `codes/mynn/metric.py`：评估指标实现（如 accuracy）。
