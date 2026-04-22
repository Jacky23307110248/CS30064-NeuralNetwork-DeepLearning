# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
# --- BLAS thread caps (item 4: OpenBLAS / MKL-style backends) ---
# English: NumPy's matmul uses a BLAS library when available. These env vars must be set BEFORE NumPy is imported
# (directly or via mynn) so the native library picks up a thread count. Uses all logical CPUs by default.
import os

_nthread = str(max(1, (os.cpu_count() or 4)))
os.environ.setdefault("OMP_NUM_THREADS", _nthread)
os.environ.setdefault("MKL_NUM_THREADS", _nthread)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _nthread)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _nthread)

import gzip
import pickle
from pathlib import Path
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot

# Fixed seed when generating a new train/val index (ignored if idx.pickle already exists).
np.random.seed(309)

# --- Switch model here; all training hyper-parameters below are shared (fair MLP vs CNN comparison). ---
RUN_MODEL = "CNN"  # "MLP" or "CNN"

# Same settings you would pass to new_test_train.py for both runs (MLP and CNN).
INIT_LR = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 20
LOG_ITERS = 100
EVAL_EVERY = 50
DEV_BATCH_SIZE = 2048
IDX_PICKLE = Path("idx.pickle")  # Relative to cwd; load if present, else create with seed above.

train_images_path = r".\dataset\MNIST\train-images-idx3-ubyte.gz"
train_labels_path = r".\dataset\MNIST\train-labels-idx1-ubyte.gz"

with gzip.open(train_images_path, "rb") as f:
    magic, num, rows, cols = unpack(">4I", f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(train_labels_path, "rb") as f:
    magic, num = unpack(">2I", f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# Train/val split: reuse idx.pickle when present (same rule as new_test_train.py).
if IDX_PICKLE.is_file():
    with open(IDX_PICKLE, "rb") as f:
        idx = pickle.load(f)
    idx = np.asarray(idx).reshape(-1)
    if idx.shape[0] != num or np.unique(idx).size != num or int(idx.min()) != 0 or int(idx.max()) != num - 1:
        raise ValueError(f"Invalid idx in {IDX_PICKLE.resolve()}")
    print(f"Loaded train/val index from {IDX_PICKLE.resolve()}")
else:
    idx = np.random.permutation(np.arange(num))
    with open(IDX_PICKLE, "wb") as f:
        pickle.dump(idx, f)
    print(f"Saved new train/val index to {IDX_PICKLE.resolve()}")

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# Normalize from [0, 255] to [0, 1]; float64 aligns with NumPy default weights and improves stability.
train_imgs = (train_imgs / train_imgs.max()).astype(np.float64)
valid_imgs = (valid_imgs / valid_imgs.max()).astype(np.float64)

if RUN_MODEL.upper() == "MLP":
    model = nn.models.Model_MLP(
        [train_imgs.shape[-1], 260, 10],
        "ReLU",
        initialize_method=np.random.normal,
    )
    save_dir = r".\best_models"
elif RUN_MODEL.upper() == "CNN":
    model = nn.models.Model_CNN(
        [1, 32, 10],
        "ReLU",
        conv_kernel_size=3,
        conv_stride=1,
        conv_padding=1,
        input_height=rows,
        input_width=cols,
        initialize_method=np.random.normal,
    )
    save_dir = r".\best_models_cnn"
else:
    raise ValueError('RUN_MODEL must be "MLP" or "CNN"')

optimizer = nn.optimizer.SGD(init_lr=INIT_LR, model=model)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(train_labs.max()) + 1)
runner = nn.runner.RunnerM(
    model,
    optimizer,
    nn.metric.accuracy,
    loss_fn,
    batch_size=BATCH_SIZE,
)

runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=NUM_EPOCHS,
    log_iters=LOG_ITERS,
    save_dir=save_dir,
    eval_every=EVAL_EVERY,
    dev_batch_size=DEV_BATCH_SIZE,
)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
plt.tight_layout()
plot(runner, axes)

plt.show()
