# MNIST MLP/CNN Checkpoints (Project 1)

本仓库用于提供 `Neural Network and Deep Learning` 课程 Project 1 的模型权重文件（NumPy 实现）。

模型任务：MNIST 手写数字分类（10 类）。

## 1. 对应代码仓库

本权重与课程代码仓库配套使用。请先获取代码（含 `mynn` 框架与数据读取脚本），再加载本仓库的权重。

- 代码根目录下核心路径：
  - `codes/mynn/`
  - `codes/new_test_train.py`
  - `codes/test_model.py`

## 2. 权重目录说明

本仓库权重目录建议保持如下结构（与训练产物一致）：

```text
best_models/
  mlp/
    baseline/best_model.pickle
    Optimization/momentum/best_model.pickle
    Optimization/LRscheduling/StepLR/best_model.pickle
    Optimization/LRscheduling/ExponentialLR/best_model.pickle
    Regularization/earlyStop/best_model.pickle
    Regularization/l2/best_model.pickle
    Regularization/l2earlyStop/best_model.pickle
  cnn/
    baseline/best_model.pickle
    Optimization/momentum/best_model.pickle
    Optimization/LRscheduling/StepLR/best_model.pickle
    Optimization/LRscheduling/ExponentialLR/best_model.pickle
    Regularization/earlyStop/best_model.pickle
    Regularization/l2/best_model.pickle
    Regularization/l2earlyStop/best_model.pickle
```

如仅需最小提交，可至少上传以下两个基线权重：

- `best_models/mlp/baseline/best_model.pickle`
- `best_models/cnn/baseline/best_model.pickle`

## 3. 环境要求

- Python 3.9+（推荐）
- `numpy`
- `matplotlib`（可选，用于画图）

安装示例：

```bash
pip install numpy matplotlib
```

## 4. 快速评测（推荐）

### 4.1 使用 MLP 基线权重

在项目 `codes/` 目录下运行：

```bash
python - <<'PY'
import gzip
from struct import unpack
import numpy as np
import mynn as nn

model = nn.models.Model_MLP()
model.load_model(r'./best_models/mlp/baseline/best_model.pickle')

with gzip.open(r'./dataset/MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
    _, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
with gzip.open(r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    _, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = (test_imgs / test_imgs.max()).astype(np.float64)
logits = model(test_imgs)
print('MLP test acc =', nn.metric.accuracy(logits, test_labs))
PY
```

### 4.2 使用 CNN 基线权重

```bash
python - <<'PY'
import gzip
from struct import unpack
import numpy as np
import mynn as nn

model = nn.models.Model_CNN()
model.load_model(r'./best_models/cnn/baseline/best_model.pickle')

with gzip.open(r'./dataset/MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
    _, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
with gzip.open(r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    _, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = (test_imgs / test_imgs.max()).astype(np.float64)
logits = model(test_imgs)
print('CNN test acc =', nn.metric.accuracy(logits, test_labs))
PY
```

## 5. 结果参考（基线）

- MLP baseline: `test acc ≈ 0.95`
- CNN baseline: `test acc ≈ 0.987`

说明：精度会因运行环境、随机种子与具体 checkpoint 略有波动。

## 6. 注意事项

- 本仓库只用于存放模型权重与必要说明，不建议上传数据集文件。
- 请配套课程代码使用；不同代码版本可能导致加载或精度不一致。
- `*.pickle` 为 Python pickle 格式文件，请仅从可信来源加载。

