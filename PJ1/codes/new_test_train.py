"""
Configurable MNIST training entrypoint (MLP / CNN, L2, early stopping, optimizer / LR schedule).

Run from the ``codes`` directory, e.g.:

  python new_test_train.py --model MLP --optimizer_setting constant --lr 0.01
  python new_test_train.py --model CNN --optimizer_setting lr_schedule --scheduler exponential --gamma 0.9995

Artifact layout under ``codes/`` when ``--preset`` or ``--experiment_slug`` is set, e.g. MLP + ``reg_l2_early_stop``:

  codes/figs/mlp/Regularization/l2earlyStop/val_test_curves.png
  codes/best_models/mlp/Regularization/l2earlyStop/best_model.pickle
  codes/traininglogs/mlp/Regularization/l2earlyStop/{experiment.json, curves.npz, training_log.txt}

  Baseline preset: ``codes/{figs,best_models,traininglogs}/mlp/baseline/`` (no ``研究方向/叶子`` nesting).

Train/val split index (``--idx_pickle``): if that file already exists under the **current working directory**
(when using a relative path), it is **loaded**; otherwise a permutation is drawn with ``--seed`` and saved there.
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
from pathlib import Path
from struct import unpack

# BLAS thread caps (must run before NumPy import via mynn)
_nthread = str(max(1, (os.cpu_count() or 4)))
os.environ.setdefault("OMP_NUM_THREADS", _nthread)
os.environ.setdefault("MKL_NUM_THREADS", _nthread)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _nthread)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _nthread)

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot
from exp_artifacts import (
    layout_paths,
    save_experiment_json,
    save_metrics_npz,
    save_val_test_curves,
    write_training_log_txt,
)
from mynn.lr_scheduler import ExponentialLR, StepLR


_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

# English: One-knob presets for the course report (each changes only the listed factor vs baseline, except
# ``reg_l2_early_stop`` which intentionally combines L2 + early stopping as a single retained experiment).
# Each dict may set any argparse field: preset runs after parse_args and overwrites matching keys for that run.
# Shared defaults for all presets: lr=0.01, batch_size=128, epochs=40 (see parse_args).
PRESETS: dict[str, dict] = {
    "baseline": {
        "optimizer_setting": "constant",
        "scheduler": "step",
        "l2_lambda": 0.0,
        "early_stop": False,
    },
    "lr_step": {
        "optimizer_setting": "lr_schedule",
        "scheduler": "step",
        "l2_lambda": 0.0,
        "early_stop": False,
        # English: ~5 epochs between decays at batch 128 on 50k samples (ceil(50000/128)≈391 steps/epoch).
        "step_size": 2000,
        "gamma": 0.5,
    },
    "lr_exp": {
        "optimizer_setting": "lr_schedule",
        "scheduler": "exponential",
        "gamma": 0.9995,
        "l2_lambda": 0.0,
        "early_stop": False,
    },
    "momentum": {
        "optimizer_setting": "momentum",
        "scheduler": "step",
        "l2_lambda": 0.0,
        "early_stop": False,
        "momentum": 0.9,
    },
    "reg_l2": {
        "optimizer_setting": "constant",
        "scheduler": "step",
        "l2_lambda": 1e-5,
        "early_stop": False,
    },
    "reg_early_stop": {
        "optimizer_setting": "constant",
        "scheduler": "step",
        "l2_lambda": 0.0,
        "early_stop": True,
        "early_stop_patience": 5,
    },
    "reg_l2_early_stop": {
        "optimizer_setting": "constant",
        "scheduler": "step",
        "l2_lambda": 1e-5,
        "early_stop": True,
        "early_stop_patience": 5,
    },
}


def apply_preset(args: argparse.Namespace) -> None:
    # English: Map high-level experiment name to concrete CLI fields (overrides current namespace values).
    key = (getattr(args, "preset", "") or "").strip().lower()
    if not key or key == "none":
        return
    if key not in PRESETS:
        raise ValueError(f"Unknown --preset {key!r}. Choose from: {', '.join(sorted(PRESETS))}.")
    for k, v in PRESETS[key].items():
        if not hasattr(args, k):
            raise AttributeError(f"preset {key!r} sets unknown arg {k!r}; add it to argparse.")
        setattr(args, k, v)


def load_mnist_train_val(
    train_images_path: Path,
    train_labels_path: Path,
    seed: int,
    val_size: int = 10_000,
    idx_pickle_path: Path | None = None,
):
    with gzip.open(train_images_path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(train_labels_path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    # --- Train/val index: reuse on disk if present, else draw from RNG and persist ---
    # If ``idx_pickle_path`` already exists, we **load** it so every run (MLP vs CNN, ablations, etc.)
    # sees the **same** 60k-sample ordering and thus the **same** 50k train / 10k val split. That keeps
    # comparisons fair and reproducible without relying on everyone typing the same ``--seed``.
    if idx_pickle_path is not None and idx_pickle_path.is_file():
        # ``idx`` must be a length-``num`` permutation of row indices into the raw MNIST train file.
        with open(idx_pickle_path, "rb") as f:
            idx = pickle.load(f)
        idx = np.asarray(idx).reshape(-1)
        # Guard against wrong dataset version, truncated files, or accidental non-index pickles.
        if idx.shape[0] != num:
            raise ValueError(
                f"Loaded idx length {idx.shape[0]} does not match MNIST train count {num} "
                f"(file: {idx_pickle_path})."
            )
        # Require a true permutation of 0..num-1 so we do not silently apply a corrupt or stale index.
        if idx.size != num or np.unique(idx).size != num or int(idx.min()) != 0 or int(idx.max()) != num - 1:
            raise ValueError(f"Loaded idx is not a permutation of 0..{num - 1}: {idx_pickle_path}")
        print(f"Loaded train/val shuffle index from {idx_pickle_path.resolve()}")
    else:
        # No index file yet (or ``idx_pickle_path`` is None): build split from ``seed`` and optionally persist.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(np.arange(num))
        if idx_pickle_path is not None:
            idx_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with open(idx_pickle_path, "wb") as f:
                pickle.dump(idx, f)
            print(
                f"Generated new train/val shuffle index (seed={seed}), saved to "
                f"{idx_pickle_path.resolve()}"
            )

    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:val_size]
    valid_labs = train_labs[:val_size]
    train_imgs = train_imgs[val_size:]
    train_labs = train_labs[val_size:]

    train_imgs = (train_imgs / train_imgs.max()).astype(np.float64)
    valid_imgs = (valid_imgs / valid_imgs.max()).astype(np.float64)
    return train_imgs, train_labs, valid_imgs, valid_labs, rows, cols


def load_mnist_test(test_images_path: Path, test_labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # English: Official MNIST test set (10k) for **curve logging only** (same cadence as validation).
    with gzip.open(test_images_path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    with gzip.open(test_labels_path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)
    test_imgs = (test_imgs / test_imgs.max()).astype(np.float64)
    return test_imgs, test_labs


def build_model(
    name: str,
    mlp_hidden: int,
    rows: int,
    cols: int,
    l2_lambda: float,
    init=np.random.normal,
):
    name_u = name.upper()
    if name_u == "MLP":
        in_dim = rows * cols
        size_list = [in_dim, mlp_hidden, 10]
        lambda_list = None
        if l2_lambda > 0:
            lambda_list = [l2_lambda] * (len(size_list) - 1)
        return nn.models.Model_MLP(size_list, "ReLU", lambda_list=lambda_list, initialize_method=init)

    if name_u == "CNN":
        size_list = [1, 32, 10]
        lambda_list = None
        if l2_lambda > 0:
            lambda_list = [l2_lambda] * 4
        return nn.models.Model_CNN(
            size_list,
            "ReLU",
            lambda_list=lambda_list,
            conv_kernel_size=3,
            conv_stride=1,
            conv_padding=1,
            input_height=rows,
            input_width=cols,
            initialize_method=init,
        )

    raise ValueError(f"Unknown model name: {name!r} (use MLP or CNN)")


def build_optimizer_and_scheduler(model, args):
    setting = args.optimizer_setting
    scheduler = None

    if setting == "constant":
        opt = nn.optimizer.SGD(init_lr=args.lr, model=model)
    elif setting == "momentum":
        opt = nn.optimizer.MomentGD(init_lr=args.lr, model=model, mu=args.momentum)
    elif setting == "lr_schedule":
        opt = nn.optimizer.SGD(init_lr=args.lr, model=model)
        if args.scheduler == "step":
            scheduler = StepLR(opt, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == "exponential":
            scheduler = ExponentialLR(opt, gamma=args.gamma)
        else:
            raise ValueError(args.scheduler)
    elif setting == "momentum_lr_schedule":
        opt = nn.optimizer.MomentGD(init_lr=args.lr, model=model, mu=args.momentum)
        if args.scheduler == "step":
            scheduler = StepLR(opt, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == "exponential":
            scheduler = ExponentialLR(opt, gamma=args.gamma)
        else:
            raise ValueError(args.scheduler)
    else:
        raise ValueError(setting)

    return opt, scheduler


def parse_args():
    p = argparse.ArgumentParser(description="Train MLP or CNN on MNIST with configurable optimization.")
    p.add_argument("--model", choices=["MLP", "CNN", "mlp", "cnn"], default="MLP", help="Model architecture")
    p.add_argument(
        "--preset",
        type=str,
        default="",
        help="Optional recipe: baseline | lr_step | lr_exp | momentum | reg_l2 | reg_early_stop | "
        "reg_l2_early_stop (overrides matching flags). Leave empty for fully manual --optimizer_setting.",
    )
    p.add_argument(
        "--experiment_slug",
        type=str,
        default="",
        help="Optional final path segment for a known --preset (replaces the table default leaf), or folder under "
        "codes/{figs,best_models,traininglogs}/<mlp|cnn>/Custom/<slug>/ when --preset is empty. "
        "If both --preset and slug are empty, legacy codes/best_models_<model> only.",
    )
    p.add_argument(
        "--optimizer_setting",
        choices=["constant", "momentum", "lr_schedule", "momentum_lr_schedule"],
        default="constant",
        help="constant | momentum | lr_schedule | momentum_lr_schedule",
    )
    p.add_argument(
        "--scheduler",
        choices=["step", "exponential"],
        default="step",
        help="Used when optimizer_setting is lr_schedule or momentum_lr_schedule (MultiStepLR not used).",
    )
    p.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum coefficient for MomentGD")
    p.add_argument(
        "--step_size",
        type=int,
        default=2000,
        help="For StepLR: decay every this many optimizer steps (tuned for batch 128 ~5 epochs on 50k MNIST)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="StepLR: multiply lr by gamma at each decay. ExponentialLR: lr *= gamma every step.",
    )
    p.add_argument("--l2_lambda", type=float, default=0.0, help="If > 0, enable L2 weight decay on all trainable layers")
    p.add_argument("--early_stop", action="store_true", help="Stop when validation does not improve for several epochs")
    p.add_argument("--early_stop_patience", type=int, default=5, help="Epochs without val improvement before stop")
    p.add_argument("--epochs", type=int, default=40, help="Default 40 for MLP and CNN (same for fair comparison)")
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Default 128 for MLP and CNN (same for fair comparison)",
    )
    p.add_argument("--log_iters", type=int, default=100)
    p.add_argument(
        "--eval_every",
        type=int,
        default=50,
        help="Same for MLP and CNN: full dev (and test) eval every N training steps (default 50). "
        "Use 1 for densest curves; each epoch still ends with a full eval when N>1.",
    )
    p.add_argument("--dev_batch_size", type=int, default=0, help="0 = full dev set in one forward; else chunk size")
    p.add_argument("--mlp_hidden", type=int, default=260, help="MLP hidden size for [784, H, 10]")
    p.add_argument("--seed", type=int, default=309)
    p.add_argument("--save_dir", type=str, default=None, help="Checkpoint directory (default: best_models_<model>)")
    p.add_argument(
        "--idx_pickle",
        type=str,
        default="idx.pickle",
        help="Train/val shuffle index file. Relative paths use cwd: if the file exists it is loaded; "
        "otherwise a permutation is drawn with --seed and saved here.",
    )
    p.add_argument("--plot", action="store_true", help="Show learning curves at end")
    return p.parse_args()


def resolve_idx_pickle_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _args_snapshot(args: argparse.Namespace) -> dict:
    d = vars(args).copy()
    for k, v in list(d.items()):
        if callable(v):
            d[k] = repr(v)
    return d


def main():
    args = parse_args()
    apply_preset(args)
    np.random.seed(args.seed)

    img_path = _CODE_DIR / "dataset" / "MNIST" / "train-images-idx3-ubyte.gz"
    lab_path = _CODE_DIR / "dataset" / "MNIST" / "train-labels-idx1-ubyte.gz"
    idx_pickle_path = resolve_idx_pickle_path(args.idx_pickle)
    train_imgs, train_labs, valid_imgs, valid_labs, rows, cols = load_mnist_train_val(
        img_path,
        lab_path,
        args.seed,
        idx_pickle_path=idx_pickle_path,
    )

    model_name = args.model.upper()
    model = build_model(
        model_name,
        mlp_hidden=args.mlp_hidden,
        rows=rows,
        cols=cols,
        l2_lambda=args.l2_lambda,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(train_labs.max()) + 1)
    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=args.batch_size,
        scheduler=scheduler,
    )

    preset_key = (args.preset or "").strip().lower()
    if preset_key == "none":
        preset_key = ""
    slug_cli = (args.experiment_slug or "").strip()
    use_artifacts = bool(preset_key) or bool(slug_cli)

    dev_bs = None if args.dev_batch_size <= 0 else args.dev_batch_size

    test_set = None
    paths: dict | None = None
    if use_artifacts:
        # English: Hierarchical tree under ``codes/`` (see ``exp_artifacts.PRESET_TO_SUBPATH``).
        paths = layout_paths(model_name.lower(), preset_key, slug_cli)
        save_dir = str(paths["weights"])
        test_img_path = _CODE_DIR / "dataset" / "MNIST" / "t10k-images-idx3-ubyte.gz"
        test_lab_path = _CODE_DIR / "dataset" / "MNIST" / "t10k-labels-idx1-ubyte.gz"
        if test_img_path.is_file() and test_lab_path.is_file():
            t_imgs, t_labs = load_mnist_test(test_img_path, test_lab_path)
            test_set = [t_imgs, t_labs]
        else:
            print(f"Warning: MNIST test files not found under {test_img_path.parent}; test curves disabled.")
    else:
        save_dir = args.save_dir or str(_CODE_DIR / f"best_models_{model_name.lower()}")

    train_kw: dict = dict(
        num_epochs=args.epochs,
        log_iters=args.log_iters,
        save_dir=save_dir,
        eval_every=args.eval_every,
        dev_batch_size=dev_bs,
        early_stop=args.early_stop,
        early_stop_patience=args.early_stop_patience,
    )
    if test_set is not None:
        train_kw["test_set"] = test_set

    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], **train_kw)
    print(f"Training finished. best_score={runner.best_score:.5f}, checkpoints in {save_dir}")

    if paths is not None:
        meta = {
            "preset": preset_key or None,
            "experiment_slug_cli": slug_cli or None,
            "artifact_relative": paths["rel"].as_posix(),
            "model": model_name,
            "best_val_accuracy": float(runner.best_score),
            "idx_pickle_resolved": str(idx_pickle_path.resolve()),
            "artifacts": {k: str(v.resolve()) for k, v in paths.items() if k != "rel"},
            "args": _args_snapshot(args),
        }
        save_experiment_json(meta, paths["traininglogs"] / "experiment.json")
        save_metrics_npz(runner, paths["traininglogs"] / "curves.npz")
        save_val_test_curves(runner, paths["figs"] / "val_test_curves.png")
        log_lines = [
            f"artifact_relative={paths['rel'].as_posix()}",
            f"model={model_name}",
            f"preset={preset_key or 'manual'}",
            f"experiment_slug_cli={slug_cli or '(default leaf from preset)'}",
            f"best_val_accuracy={runner.best_score:.6f}",
            f"optimizer_setting={args.optimizer_setting}",
            f"scheduler={args.scheduler}",
            f"lr={args.lr} batch_size={args.batch_size} epochs={args.epochs}",
            f"eval_every={args.eval_every} dev_batch_size={dev_bs}",
            f"l2_lambda={args.l2_lambda} early_stop={args.early_stop} patience={args.early_stop_patience}",
            f"weights_dir={paths['weights']}",
        ]
        write_training_log_txt(log_lines, paths["traininglogs"] / "training_log.txt")
        print(f"Saved artifacts under codes/: {paths['rel'].as_posix()} (figs, traininglogs, best_models).")

    if args.plot:
        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        plt.tight_layout()
        plot(runner, axes)
        plt.show()


if __name__ == "__main__":
    main()
