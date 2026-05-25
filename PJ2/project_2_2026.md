# 神经网络与深度学习 课程项目2
付彦伟
2026年5月8日

## 摘要
(1) 本文档为本课程的第二个项目说明。提交截止时间为**2026年6月14日23:59**，请通过在线学习平台上传报告。
(2) 报告的目标是记录你所做的实验与主要发现，务必对结果进行解释。提交一份PDF格式的报告，并在文件中附上代码的GitHub链接。报告中还需提供数据集与训练好的模型权重链接，可将数据集与模型上传至谷歌云端硬盘或其他网盘平台。同时在报告中注明姓名与学号。缺少代码链接或模型权重链接将被扣分。
(3) 关于截止时间与迟交处罚：原则上需按各项目截止时间提交报告；允许迟交，但**每延迟一周扣10%的分数**。
(4) 低年级学生建议从昇思（MindSpore）的CIFAR-10示例教程开始学习：
https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/dataset/mindspore.dataset.Cifar10Dataset.html

---

## 1 在CIFAR-10上训练网络（60分）
CIFAR-10[4]是视觉识别任务中广泛使用的数据集。CIFAR-10数据集（加拿大高级研究院）是常用于训练机器学习与计算机视觉算法的图像集合，也是机器学习研究中最常用的数据集之一。
CIFAR-10数据集包含**60000张32×32的彩色图像**，分为10个类别，分别对应飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车（如图1所示），每个类别有6000张图像。由于CIFAR-10中的图像分辨率较低（32×32），该数据集可用于快速验证模型是否有效。

在本项目中，你需要在CIFAR-10上训练神经网络模型以优化性能，报告在该数据集上能达到的**最佳测试误差**，以及实现该结果所构建的网络结构。

### 1.1 入门指南
1. 可从官方网站[2]下载数据集，或使用torchvision工具包。以下是PyTorch提供的示例代码[1]：
```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
```
代码第5行设置`download=True`时，会自动将数据集下载到指定根目录。关于神经网络模型的构建、训练与测试，可先阅读教程[1]。

2. 你的网络**必须包含**以下全部组件（16分）：
   (a) 全连接层；
   (b) 二维卷积层；
   (c) 二维池化层；
   (d) 激活函数。

3. 你的网络**至少包含**以下组件中的一种（8分）：
   (a) 批归一化层；
   (b) 随机失活（Dropout）；
   (c) 残差连接；
   (d) 其他组件。

4. 优化网络时**必须尝试**以下所有策略（8分）：
   (a) 尝试不同数量的神经元/卷积核；
   (b) 尝试不同的损失函数（搭配不同正则化）；
   (c) 尝试不同的激活函数。

5. 优化网络时**需选择**以下策略之一（8分）：
   (a) 使用`torch.optim`尝试不同优化器；
   (b) 手动实现包含上述2(a)-(d)组件网络的优化器（不可使用`torch.nn`、`torch.sigmoid`等，可使用`torch.matmul`等基础运算），再用`torch.optim`优化完整模型；
   (c) 手动实现完整模型的优化器。

6. 揭示网络的内在原理，例如卷积核可视化、损失曲面、网络可解释性分析等（8分）。

### 1.2 本任务评分标准
可自由使用PyTorch或其他深度学习框架的任意组件，本任务评分依据如下：
(1) （12分）网络的分类性能，综合考量网络总参数量、网络结构与训练速度。结果相近的项目，将进一步核查参数量、网络结构，或是否使用了新的优化算法训练网络。
(2) 学习到的模型与训练过程是否有有深度的结果、有趣的可视化等。
(3) 可报告多个网络的结果；但**直接使用未做任何修改的公开模型**，项目分数会酌情扣减。

---

## 2 批归一化（30分）
批归一化（Batch Normalization，BN）是被广泛采用的技术，能让深度神经网络（DNN）训练更快、更稳定。因其能提升精度、加速训练，BN已成为深度学习中最受欢迎的技术之一。
简单来说，BN通过稳定网络层输入的分布来改善神经网络训练，具体是通过引入额外网络层控制这些分布的一阶（均值）与二阶（方差）矩实现。

在本项目中，你将先测试BN在训练过程中的效果，再探究BN如何助力优化。本文档提供Python示例代码。

### 2.1 批归一化算法
本节主要针对卷积神经网络的BN层。BN层的输入与输出均为四维张量，分别记为\(I_{b, c, x, y}\)与\(O_{b, c, x, y}\)，各维度依次对应批次内样本\(b\)、通道\(c\)、两个空间维度\(x\)与\(y\)。输入图像的通道对应RGB通道。
BN对**同一通道**的所有激活值执行相同的归一化操作：
\[O_{b, c, x, y} \leftarrow \gamma_{c} \frac{I_{b, c, x, y}-\mu_{c}}{\sqrt{\sigma_{c}^{2}+\epsilon}}+\beta_{c} \quad \forall b, c, x, y\]

其中，BN从通道\(c\)的所有输入激活值中减去该通道的均值\(\mu_{c}\)，\(\mu_{c}\)的计算方式为：
\[\mu_{c}=\frac{1}{|B|} \sum_{b, x, y} I_{b, c, x, y}\]
\(B\)表示整个小批次中、所有空间位置\((x,y)\)上通道\(c\)的全部激活值。
随后，将中心化后的激活值除以标准差\(\sigma_{c}\)（加\(\epsilon\)保证数值稳定），\(\sigma_{c}\)的计算方式与均值类似。
**测试阶段**使用均值与方差的滑动平均。归一化后，通过通道级的仿射变换重构输出，\(\gamma_{c}\)与\(\beta_{c}\)为训练中学习的参数。

### 图2 VGG模型的卷积网络结构（按列展示）
| 卷积网络结构 | | | | | |
| --- | --- | --- | --- | --- | --- |
| A（11个权重层） | A-LRN | B（13个权重层） | C（16个权重层） | D（16个权重层） | E（19个权重层） |
| 输入（224×224 RGB图像） | | | | | |
| conv3-64 | conv3-64 LRN | conv3-64 conv3-64 | conv3-64 conv3-64 | conv3-64 conv3-64 | conv3-64 conv3-64 |
| 最大池化 | | | | | |
| conv3-128 | conv3-128 | conv3-128 | conv3-128 | conv3-128 | conv3-128 |
| | | conv3-128 | conv3-128 | conv3-128 | conv3-128 |
| 最大池化 | | | | | |
| conv3-256 conv3-256 | conv3-256 | conv3-256 | conv3-256 | conv3-256 | conv3-256 |
| conv3-256 | conv3-256 | conv3-256 | conv3-256 | conv3-256 | conv3-256 |
| | | conv1-256 | conv3-256 | conv3-256 | |
| | | | | conv3-256 | |
| 最大池化 | | | | | |
| conv3-512 conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 |
| conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 |
| | | conv1-512 | conv3-512 | conv3-512 | |
| 最大池化 | | | | | |
| conv3-512 conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 |
| conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 |
| | | | conv1-512 | conv3-512 | conv3-512 |
| | | | | | conv3-512 |
| 最大池化 | | | | | |
| FC-4096 | | | | | |
| FC-4096 | | | | | |
| FC-1000 | | | | | |
| Softmax | | | | | |

---

为研究批归一化，实验设置如下：基于CIFAR-10图像分类任务，使用与**VGG-A结构相同**的网络，但因输入为32×32×3（而非224×224×3），全连接层尺寸更小。所有示例代码均基于PyTorch实现。

可运行`loaders.py`下载CIFAR-10数据集，并输出部分样本，熟悉数据存储格式。注意：若使用远程服务器，需用`matplotlib.pyplot.savefig()`保存绘图结果，再下载到本地查看。

### 2.2 加BN与不加BN的VGG-A（15分）
本节需对比**加BN**与**不加BN**的VGG-A网络的性能与特征。建议扩展并修改提供的代码，让实验结果更清晰、更有说服力。注意需先理解代码，而非当作黑盒使用。

若想使用部分数据集加速训练，可设置`n_items`参数。基础VGG-A网络在`VGG.py`中实现，可先训练该网络，理解整体架构并查看训练结果。随后编写`VGG_BatchNorm`类，在原网络中添加BN层，最后将两者的训练结果可视化对比。
训练与可视化的示例代码已包含在`VGG_Loss_Landscape.py`中。

### 2.3 BN如何助力优化？（15分）
仅使用BN是不够的，需理解其在优化过程中发挥积极作用的原理，从而更全面地理解深度学习优化过程，并在后续工作中根据实际情况选择合适的网络结构。可参考相关论文[3]。

BN的核心原理是什么？为理解BN对训练性能的影响，合理的思路是分析BN对对应优化曲面的作用。
我们的训练基于梯度下降法，该方法属于一阶优化范式：用当前解附近损失的局部线性近似来确定最优更新步长。因此，这类算法的性能很大程度上取决于局部线性近似对邻近损失曲面的预测能力。

近期研究表明：**BN通过重新参数化底层优化问题，让损失曲面显著更平滑**。基于此，我们需要测量：
1. 损失曲面或损失值的变化；
2. 梯度预测性或损失梯度的变化；
3. 梯度随距离的最大差值。

#### 2.3.1 损失曲面
为测试BN对损失本身稳定性（即利普希茨连续性）的影响，在训练过程的每一步，需计算该步的损失梯度，并衡量沿该方向移动时损失的变化，即测量特定训练步的损失波动。简易实现步骤如下：
1. 选取一组学习率代表不同步长训练并保存模型，例如`[1e-3, 2e-3, 1e-4, 5e-4]`；
2. 保存所有模型每一步的训练损失；
3. 维护两个列表：`max_curve`与`min_curve`，选取所有模型在同一步的损失最大值加入`max_curve`，最小值加入`min_curve`；
4. 绘制两个列表的结果，并用`matplotlib.pyplot.fill_between`填充两条线之间的区域。

对加BN的VGG-A模型采用相同方法，最后将**加BN与不加BN的VGG-A结果可视化在同一张图**中，更直观对比。

为方便理解，我们提供损失曲面可视化的示例代码（`VGG_Loss_Landscape.py`）。需理解代码并训练不同模型复现图2结果，可自由修改与完善示例代码，并报告所选学习率。最重要的是，借助Matplotlib展示最终对比结果。

---

## 参考文献
[1] https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
[2] https://www.cs.toronto.edu/~kriz/cifar.html.
[3] How does batch normalization help optimization? In NeurIPS, 2018.
[4] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.