# CS30064-NeuralNetwork-DeepLearning

复旦大学 CS30064《Neural Network and Deep Learning》（神经网络与深度学习）课程相关代码与作业仓库。

本仓库用于集中存放各次实验与项目；当前已收录：

- **`PJ1/`** — Project 1：基于 NumPy 实现 MLP / CNN，完成 MNIST 分类、优化与正则、鲁棒性与误差分析等。运行说明、目录结构与命令见 [`PJ1/README.md`](PJ1/README.md)。
- **`PJ2/`** — Project 2：在 CIFAR-10 上训练 ResNet 并完成消融（Task 1），对比 VGG-A 与 VGG-A+BN 及损失曲面 / 梯度 landscape 分析（Task 2）。运行说明、目录结构与命令见 [`PJ2/README.md`](PJ2/README.md)。

## 快速导航

| 内容 | 路径 |
|------|------|
| PJ1 说明与运行方式 | [`PJ1/README.md`](PJ1/README.md) |
| PJ1 报告（源文件） | [`PJ1/report.ipynb`](PJ1/report.ipynb) |
| PJ1 报告（PDF / HTML） | [`PJ1/report.pdf`](PJ1/report.pdf) · [`PJ1/report.html`](PJ1/report.html) |
| PJ2 说明与运行方式 | [`PJ2/README.md`](PJ2/README.md) |
| PJ2 作业说明 | [`PJ2/project_2_2026.md`](PJ2/project_2_2026.md) |
| PJ2 报告（源文件） | [`PJ2/report.ipynb`](PJ2/report.ipynb) |

## 模型权重与数据（不上传 GitHub 的部分）

按课程要求，仓库中不上传完整数据集与大体积训练产物。各项目下载方式如下。

### PJ1

PJ1 训练得到的 `best_models` 已上传至 ModelScope：

- https://modelscope.cn/models/JackYHon/CS30064-PJ1/tree/master/best_models

下载与放置方式见 [`PJ1/README.md`](PJ1/README.md) 第三节。

### PJ2

PJ2 的 CIFAR-10 数据（`data/cifar-10-batches-py/`）与各实验输出（`outputs/`，含 `best.pt` 等）已上传至 Google Drive：

- https://drive.google.com/drive/folders/13fio3sFs1qT7FBt6_mwabVRHcdk42U1d?usp=sharing

解压后分别放到 `PJ2/data/`、`PJ2/outputs/`，详见 [`PJ2/README.md`](PJ2/README.md)「代码与数据下载」一节。仓库内保留 `data/loaders.py`、`data/__init__.py` 等脚本；`data/cifar-10-batches-py/` 与 `outputs/` 由 `PJ2/.gitignore` 排除。

## 说明

- 本仓库主要用于课程作业提交、复现与版本管理。
- 各子项目内的依赖、环境与命令以**各子目录下的 `README.md`** 为准。
