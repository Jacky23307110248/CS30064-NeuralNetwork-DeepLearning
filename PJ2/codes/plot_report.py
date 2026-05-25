"""
One-off plotting for report.ipynb figures.

Reads training outputs under outputs/CIFAR10_NeuroNet and outputs/VGG_BatchNorm,
writes PNGs to pic/ (overwrites existing files).

Usage:
    python codes/plot_report.py all
    python codes/plot_report.py 1_3_2
    python codes/plot_report.py 1_4_2 2_3_2
    python codes/plot_report.py 1_3_2 --cifar-root path/to/CIFAR10_NeuroNet
    python codes/plot_report.py 1_3_2 --out-dir my_pic --name custom.png
    python codes/plot_report.py 1_3_2 --output my_pic/custom.png

Figure ids (default output under pic/):
    1_3_2, 1_4_2, 1_5_2, 1_6_2_1, 1_6_2_2, 1_7_2, 1_8_2, 1_8_2_best, 1_9_2,
    1_10_2_loss_landscape_sgd, 1_10_2_kernels, 1_10_2_confmat, 1_10_2_top10_errors, 1_10_2_gradcam,
    2_3_2, 2_3_3_feature_maps, 2_3_3_activation_stats,
    2_4_2_loss_landscape, 2_4_2_grad_predictability_landscape, 2_4_2_grad_diff_landscape

Grad sweep figures (2.5.2) read grad_sweep.npz from grad_probe runs:
    python codes/plot_report.py 2_4_2_grad_predictability_landscape 2_4_2_grad_diff_landscape
    python codes/plot_report.py 2_4_2_grad_diff_landscape \\
        --grad-vgg-a-run-dir outputs/VGG_BatchNorm/VGG_A_grad_probe_fix_... \\
        --grad-vgg-a-bn-run-dir outputs/VGG_BatchNorm/VGG_A_BatchNorm_grad_probe_fix_...
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CIFAR_ROOT = PROJECT_ROOT / "outputs" / "CIFAR10_NeuroNet"
DEFAULT_VGG_ROOT = PROJECT_ROOT / "outputs" / "VGG_BatchNorm"
DEFAULT_PIC_DIR = PROJECT_ROOT / "pic"

CIFAR_CODE_DIR = SCRIPT_DIR / "CIFAR10_NeuroNet"
VGG_CODE_DIR = SCRIPT_DIR / "VGG_BatchNorm"
if str(CIFAR_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CIFAR_CODE_DIR))

from data_split import get_dataloaders  # noqa: E402
from models.model import build_model  # noqa: E402
from paths import get_device as get_cifar_device  # noqa: E402

BASELINE_PREFIX = "baseline_full-adamw1e-3_wd5e4_ep200"
LANDSCAPE_LRS = [1e-3, 2e-3, 1e-4, 5e-4]

WARMUP_STEPS = 20
Y_TOP_PERCENTILE = 99.5
Y_TOP_MARGIN = 1.08
PLOT_LW = 0.15
PLOT_FILL_ALPHA = 0.28
LEGEND_LW = 1.2
LANDSCAPE_STRIDE = 11  # ~88 steps/epoch -> 8 points per epoch (loss landscape only)
GRAD_PRED_YLIM = (0.990, 1.020)
GRAD_PRED_SEGMENTS_PER_EPOCH = 1   # grad predictability: 1 point per epoch (~100 total)
GRAD_DIFF_SEGMENTS_PER_EPOCH = 2   # grad difference: 2 points per epoch (~200 total)

# grad_probe distance-sweep runs (2 models, single lr)
GRAD_SWEEP_LR = 1e-3
GRAD_SWEEP_EPOCHS = 100
GRAD_SWEEP_STRIDE = 44          # ~2 points/epoch for display (less hairy than stride 11)
GRAD_SWEEP_SMOOTH_WINDOW = 11   # rolling mean on downsampled curves
GRAD_SWEEP_LW = 1.15
GRAD_SWEEP_FILL_ALPHA = 0.20
GRAD_SWEEP_Y_PERCENTILE = 97.5
ANALYSIS_SGD_LRS = (0.05, 0.1, 0.15, 0.2)
ANALYSIS_TOPK_ERRORS = 10
ANALYSIS_GRADCAM_K = 10
ANALYSIS_GRADCAM_LAYER = "layer3.0.conv1"
CIFAR10_CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

DEFAULT_GRAD_SWEEP_RUN_DIRS: dict[str, Path] = {
    "VGG_A": (
        DEFAULT_VGG_ROOT
        / "VGG_A_grad_probe_fix_1e-3_ep100_seed2020_20260523-173434"
    ),
    "VGG_A_BatchNorm": (
        DEFAULT_VGG_ROOT
        / "VGG_A_BatchNorm_grad_probe_fix_1e-3_ep100_seed2020_20260523-181529"
    ),
}

VGG_BN_COMPARE_PREFIX = {
    "VGG_A": "VGG_A_bn_compare_adamw1e-3_cos_ep200_seed2020",
    "VGG_A_BatchNorm": "VGG_A_BatchNorm_bn_compare_adamw1e-3_cos_ep200_seed2020",
}
DEFAULT_VGG_BN_COMPARE_RUN_DIRS: dict[str, Path] = {
    "VGG_A": (
        DEFAULT_VGG_ROOT
        / "VGG_A_bn_compare_adamw1e-3_cos_ep200_seed2020_20260520-121556"
    ),
    "VGG_A_BatchNorm": (
        DEFAULT_VGG_ROOT
        / "VGG_A_BatchNorm_bn_compare_adamw1e-3_cos_ep200_seed2020_20260520-123453"
    ),
}
VGG_FEATURE_TOPK = 4
VGG_FEATURE_LAYER_SPECS = (("S1", 2, 1), ("S3", 10, 6))
VGG_STATS_LAYER_SPECS = (("S1", 2, 1), ("S3", 10, 6), ("S5", 20, 12))
VGG_STATS_CSV_PATH = DEFAULT_VGG_ROOT / "feature_compare" / "stats.csv"
VGG_HIST_SAMPLE_MAX = 50000

MODEL_LANDSCAPE_SPECS = [
    ("VGG_A", "Standard VGG", "#2ca02c", "#1a6b1a"),
    ("VGG_A_BatchNorm", "Standard VGG + BatchNorm", "#d62728", "#8b0000"),
]

PanelKind = Literal["loss", "acc"]
SourceKind = Literal[
    "cifar",
    "vgg_curves",
    "vgg_landscape_loss",
    "vgg_landscape_grad_pred",
    "vgg_landscape_grad_diff",
    "vgg_grad_sweep_pred",
    "vgg_grad_sweep_diff",
    "analysis_landscape_sgd",
    "analysis_kernels",
    "analysis_confusion",
    "analysis_top_errors",
    "analysis_gradcam",
    "vgg_feature_maps",
    "vgg_activation_stats",
]


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    prefix: str


@dataclass(frozen=True)
class FigureSpec:
    fig_id: str
    default_name: str
    title: str
    source: SourceKind
    panels: tuple[PanelKind, ...]
    series: tuple[SeriesSpec, ...]


def _lr_tag(lr: float) -> str:
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def discover_run_dir(root: Path, prefix: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Output root not found: {root}")
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No run under {root} matching prefix '{prefix}*'")
    return candidates[-1]


def load_curves(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "curves.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing curves.npz: {path}")
    data = np.load(path)
    required = ("epoch", "train_loss", "val_loss", "train_acc", "val_acc")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return {k: np.asarray(data[k]) for k in required}


def load_steps(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "steps.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing steps.npz: {path}")
    data = np.load(path)
    if "global_step" not in data or "train_loss" not in data:
        raise KeyError(f"{path} must contain global_step and train_loss")
    order = np.argsort(data["global_step"])
    steps = np.asarray(data["global_step"][order], dtype=np.int64)
    loss = np.asarray(data["train_loss"][order], dtype=np.float64)
    return steps, loss


def _grad_sweep_run_dir(
    vgg_root: Path,
    model_name: str,
    lr: float = GRAD_SWEEP_LR,
    max_epochs: int = GRAD_SWEEP_EPOCHS,
) -> Path:
    tag = f"fix_{_lr_tag(lr)}_ep{max_epochs}"
    prefix = f"{model_name}_grad_probe_{tag}_"
    return discover_run_dir(vgg_root, prefix)


def resolve_grad_sweep_run_dirs(
    vgg_root: Path,
    *,
    overrides: dict[str, Path] | None = None,
    lr: float = GRAD_SWEEP_LR,
    max_epochs: int = GRAD_SWEEP_EPOCHS,
) -> dict[str, Path]:
    """Resolve run directory per model: CLI override > DEFAULT_GRAD_SWEEP_RUN_DIRS > newest under vgg_root."""
    resolved: dict[str, Path] = {}
    for model_name, _, _, _ in MODEL_LANDSCAPE_SPECS:
        if overrides and model_name in overrides and overrides[model_name] is not None:
            resolved[model_name] = Path(overrides[model_name])
            continue
        default_dir = DEFAULT_GRAD_SWEEP_RUN_DIRS.get(model_name)
        if default_dir is not None:
            resolved[model_name] = default_dir
            continue
        resolved[model_name] = _grad_sweep_run_dir(vgg_root, model_name, lr=lr, max_epochs=max_epochs)
    return resolved


def load_grad_sweep(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps and per-step min/max of ||g'-g|| over distance alphas."""
    path = run_dir / "grad_sweep.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing grad_sweep.npz: {path}")
    data = np.load(path)
    required = ("global_step", "grad_diff_min", "grad_diff_max")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    order = np.argsort(data["global_step"])
    steps = np.asarray(data["global_step"][order], dtype=np.int64)
    lo = np.asarray(data["grad_diff_min"][order], dtype=np.float64)
    hi = np.asarray(data["grad_diff_max"][order], dtype=np.float64)
    return steps, lo, hi


def load_grad_probe(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run_dir / "grad_probe.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing grad_probe.npz: {path}")
    data = np.load(path)
    required = ("global_step", "grad_norm_w", "grad_norm_perturbed", "grad_diff")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    order = np.argsort(data["global_step"])
    steps = np.asarray(data["global_step"][order], dtype=np.int64)
    gn_w = np.asarray(data["grad_norm_w"][order], dtype=np.float64)
    gn_p = np.asarray(data["grad_norm_perturbed"][order], dtype=np.float64)
    g_diff = np.asarray(data["grad_diff"][order], dtype=np.float64)
    ratio = np.divide(
        gn_p,
        gn_w,
        out=np.full_like(gn_p, np.nan),
        where=gn_w > 0,
    )
    return steps, ratio, g_diff


def load_step_epochs(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "steps.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing steps.npz: {path}")
    data = np.load(path)
    if "global_step" not in data or "epoch" not in data:
        raise KeyError(f"{path} must contain global_step and epoch")
    order = np.argsort(data["global_step"])
    steps = np.asarray(data["global_step"][order], dtype=np.int64)
    epochs = np.asarray(data["epoch"][order], dtype=np.int64)
    return steps, epochs


def _epochs_for_probe_steps(run_dir: Path, probe_steps: np.ndarray) -> np.ndarray:
    ref_steps, epochs = load_step_epochs(run_dir)
    if len(probe_steps) == len(ref_steps) and np.array_equal(probe_steps, ref_steps):
        return epochs
    step_to_epoch = {int(s): int(e) for s, e in zip(ref_steps, epochs)}
    missing = [int(s) for s in probe_steps if int(s) not in step_to_epoch]
    if missing:
        raise ValueError(f"{run_dir}: probe steps not found in steps.npz (e.g. {missing[:3]})")
    return np.asarray([step_to_epoch[int(s)] for s in probe_steps], dtype=np.int64)


def _aggregate_probe_run(
    run_dir: Path,
    value_key: Literal["ratio", "grad_diff"],
    segments_per_epoch: int = GRAD_DIFF_SEGMENTS_PER_EPOCH,
) -> tuple[np.ndarray, np.ndarray]:
    probe_steps, ratio, g_diff = load_grad_probe(run_dir)
    epochs = _epochs_for_probe_steps(run_dir, probe_steps)
    values = ratio if value_key == "ratio" else g_diff

    agg_steps: list[float] = []
    agg_vals: list[float] = []
    for epoch in np.unique(epochs):
        mask = epochs == epoch
        ep_steps = probe_steps[mask]
        ep_vals = values[mask]
        order = np.argsort(ep_steps)
        ep_steps = ep_steps[order]
        ep_vals = ep_vals[order]
        n = len(ep_steps)
        if n == 0:
            continue
        n_seg = max(1, min(segments_per_epoch, n))
        bounds = np.linspace(0, n, n_seg + 1, dtype=int)
        for i in range(n_seg):
            lo, hi = bounds[i], bounds[i + 1]
            if lo >= hi:
                continue
            agg_steps.append(float(np.median(ep_steps[lo:hi])))
            agg_vals.append(float(np.nanmedian(ep_vals[lo:hi])))
    return np.asarray(agg_steps, dtype=np.int64), np.asarray(agg_vals, dtype=np.float64)


def _analysis_lr_tag(lr: float) -> str:
    if lr >= 0.01:
        return f"{lr:.2f}".rstrip("0").rstrip(".")
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def _load_run_config(run_dir: Path) -> dict:
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["channels"] = tuple(cfg["channels"])
    cfg["blocks_per_stage"] = tuple(cfg["blocks_per_stage"])
    return cfg


def _load_model_from_run(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    cfg = _load_run_config(run_dir)
    model = build_model(cfg).to(device)
    ckpt_path = run_dir / "best.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def _resolve_analysis_run_dir(cifar_root: Path, override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    return discover_run_dir(cifar_root, "combine-cutmix_w96_sgd0.1-")


def _resolve_analysis_baseline_dir(cifar_root: Path, override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    return discover_run_dir(cifar_root, BASELINE_PREFIX)


def _resolve_analysis_landscape_runs(
    cifar_root: Path,
    *,
    use_cutmix: bool,
    run_dirs: list[Path] | None = None,
) -> list[tuple[float, Path]]:
    if run_dirs is not None:
        if len(run_dirs) != len(ANALYSIS_SGD_LRS):
            raise ValueError(
                f"Expected {len(ANALYSIS_SGD_LRS)} run dirs for analysis landscape, "
                f"got {len(run_dirs)}."
            )
        return [(lr, Path(rd)) for lr, rd in zip(ANALYSIS_SGD_LRS, run_dirs)]

    runs = []
    variant = "cutmix_w96" if use_cutmix else "nocutmix_w96"
    for lr in ANALYSIS_SGD_LRS:
        tag = _analysis_lr_tag(lr)
        prefix = f"analysis_landscape_sgd-{variant}_sgd{tag}_ep50-"
        runs.append((lr, discover_run_dir(cifar_root, prefix)))
    return runs


def _denormalize_cifar_tensor(x: torch.Tensor) -> np.ndarray:
    img = x.detach().cpu().permute(1, 2, 0).numpy()
    img = img * 0.5 + 0.5
    return np.clip(img, 0.0, 1.0)


def _get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    module = model
    for token in module_name.split("."):
        if token.isdigit():
            module = module[int(token)]
        else:
            module = getattr(module, token)
    return module


def _collect_test_predictions(run_dir: Path) -> dict:
    device = get_cifar_device()
    model, cfg = _load_model_from_run(run_dir, device)
    cfg["num_workers"] = 0
    _, _, test_loader = get_dataloaders(cfg)
    y_true = []
    y_pred = []
    confidence = []
    sample_idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for b in range(x.size(0)):
                y_true.append(int(y[b].item()))
                y_pred.append(int(pred[b].item()))
                confidence.append(float(conf[b].item()))
                sample_idx += 1
    return {
        "run_dir": str(run_dir),
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "confidence": np.asarray(confidence, dtype=np.float64),
    }


def _collect_topk_confidence_errors(run_dir: Path, k: int) -> list[dict]:
    device = get_cifar_device()
    model, cfg = _load_model_from_run(run_dir, device)
    cfg["num_workers"] = 0
    _, _, test_loader = get_dataloaders(cfg)
    top_errors: list[dict] = []
    sample_idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for b in range(x.size(0)):
                y_true = int(y[b].item())
                y_pred = int(pred[b].item())
                conf_val = float(conf[b].item())
                if y_true != y_pred:
                    top_errors.append(
                        {
                            "sample_idx": sample_idx,
                            "true": y_true,
                            "pred": y_pred,
                            "confidence": conf_val,
                            "image": x[b].detach().cpu(),
                        }
                    )
                sample_idx += 1
    top_errors.sort(key=lambda item: item["confidence"], reverse=True)
    return top_errors[:k]


def _analysis_envelope_from_runs(
    cifar_root: Path,
    *,
    use_cutmix: bool,
    run_dirs: list[Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_pairs = _resolve_analysis_landscape_runs(
        cifar_root,
        use_cutmix=use_cutmix,
        run_dirs=run_dirs,
    )
    loaded = []
    for lr, run_dir in run_pairs:
        steps, losses = load_steps(run_dir)
        loaded.append((lr, run_dir, steps, losses))
    ref_steps = loaded[0][2]
    for _, _, steps, _ in loaded[1:]:
        if len(steps) != len(ref_steps) or not np.array_equal(steps, ref_steps):
            raise ValueError("analysis landscape runs must have identical global_step sequences")
    stack = np.stack([item[3] for item in loaded], axis=0)
    max_curve = stack.max(axis=0)
    min_curve = stack.min(axis=0)
    # Keep the same readability settings as before.
    stride = 8
    ref_steps = ref_steps[::stride]
    max_curve = _smooth_1d(max_curve[::stride], window=9)
    min_curve = _smooth_1d(min_curve[::stride], window=9)
    min_curve = np.minimum(min_curve, max_curve)
    return ref_steps, max_curve, min_curve


def plot_analysis_loss_landscape(
    cifar_root: Path,
    out_path: Path,
    *,
    cutmix_run_dirs: list[Path] | None = None,
    nocutmix_run_dirs: list[Path] | None = None,
) -> None:
    cutmix_steps, cutmix_max, cutmix_min = _analysis_envelope_from_runs(
        cifar_root,
        use_cutmix=True,
        run_dirs=cutmix_run_dirs,
    )
    nocutmix_steps, nocutmix_max, nocutmix_min = _analysis_envelope_from_runs(
        cifar_root,
        use_cutmix=False,
        run_dirs=nocutmix_run_dirs,
    )

    _setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(9, 9.6), sharex=True)

    # Top: CutMix (red palette), keep old styling.
    axes[0].plot(cutmix_steps, cutmix_max, color="#8b0000", lw=0.7, label="max_curve")
    axes[0].plot(cutmix_steps, cutmix_min, color="#8b0000", lw=0.7, linestyle="--", label="min_curve")
    axes[0].fill_between(cutmix_steps, cutmix_min, cutmix_max, color="#d62728", alpha=0.40)
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Task1 Loss Landscape (SGD lr sweep, CutMix)")
    axes[0].set_ylim(0.0, 3.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    # Bottom: No CutMix (blue palette), same alpha/linewidth/scale settings.
    axes[1].plot(nocutmix_steps, nocutmix_max, color="#0b3d91", lw=0.7, label="max_curve")
    axes[1].plot(nocutmix_steps, nocutmix_min, color="#0b3d91", lw=0.7, linestyle="--", label="min_curve")
    axes[1].fill_between(nocutmix_steps, nocutmix_min, nocutmix_max, color="#1f77b4", alpha=0.40)
    axes[1].set_xlabel("Global step")
    axes[1].set_ylabel("Train loss")
    axes[1].set_title("Task1 Loss Landscape (SGD lr sweep, No CutMix)")
    axes[1].set_ylim(0.0, 3.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_filter_grid(ax, weight: torch.Tensor, title: str, max_filters: int = 64) -> None:
    w = weight.detach().cpu()
    n = min(max_filters, w.shape[0])
    grid_size = int(np.ceil(np.sqrt(n)))
    k_h, k_w = w.shape[2], w.shape[3]
    upscale = 10 if max(k_h, k_w) <= 3 else 1
    cell_h, cell_w = k_h * upscale, k_w * upscale
    canvas = np.ones((grid_size * cell_h, grid_size * cell_w, 3), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, grid_size)
        filt = w[i].numpy()  # [in_channels, k_h, k_w]
        if filt.shape[0] == 3:
            vis = np.transpose(filt, (1, 2, 0))
        else:
            # For deeper conv layers, project top-3 strongest input channels to RGB.
            channel_strength = np.mean(np.abs(filt), axis=(1, 2))
            top_idx = np.argsort(channel_strength)[-3:]
            rgb = filt[top_idx, :, :]
            if rgb.shape[0] < 3:
                rgb = np.pad(rgb, ((0, 3 - rgb.shape[0]), (0, 0), (0, 0)), mode="edge")
            vis = np.transpose(rgb, (1, 2, 0))
        f_min, f_max = float(vis.min()), float(vis.max())
        if f_max > f_min:
            vis = (vis - f_min) / (f_max - f_min)
        else:
            vis = np.zeros_like(vis)
        if upscale > 1:
            vis = np.repeat(np.repeat(vis, upscale, axis=0), upscale, axis=1)
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w, :] = vis
    ax.imshow(canvas)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_analysis_kernels(cifar_root: Path, out_path: Path, run_dir: Path | None, baseline_run_dir: Path | None) -> None:
    device = get_cifar_device()
    best_run = _resolve_analysis_run_dir(cifar_root, run_dir)
    base_run = _resolve_analysis_baseline_dir(cifar_root, baseline_run_dir)
    model_best, _ = _load_model_from_run(best_run, device)
    model_base, _ = _load_model_from_run(base_run, device)

    _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _plot_filter_grid(axes[0, 0], model_base.conv1.weight.data, "Baseline conv1")
    _plot_filter_grid(axes[0, 1], model_best.conv1.weight.data, "Best conv1")
    _plot_filter_grid(axes[1, 0], model_base.layer3[0].conv1.weight.data, "Baseline layer3.0.conv1")
    _plot_filter_grid(axes[1, 1], model_best.layer3[0].conv1.weight.data, "Best layer3.0.conv1")
    fig.suptitle("Task1 Kernel Visualization (conv1 + layer3.0.conv1)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_analysis_confusion(cifar_root: Path, out_path: Path, run_dir: Path | None) -> None:
    run = _resolve_analysis_run_dir(cifar_root, run_dir)
    pred_data = _collect_test_predictions(run)
    conf = np.zeros((10, 10), dtype=np.int64)
    for y_t, y_p in zip(pred_data["y_true"], pred_data["y_pred"]):
        conf[int(y_t), int(y_p)] += 1
    row_sum = conf.sum(axis=1, keepdims=True).clip(min=1)
    conf_norm = conf / row_sum

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CIFAR10_CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Task1 Confusion Matrix (normalized by true class)")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            val = int(conf[i, j])
            text_color = "white" if conf_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_analysis_top_errors(cifar_root: Path, out_path: Path, run_dir: Path | None, topk: int = ANALYSIS_TOPK_ERRORS) -> list[dict]:
    run = _resolve_analysis_run_dir(cifar_root, run_dir)
    top_errors = _collect_topk_confidence_errors(run, topk)

    cols = 5
    rows_n = int(np.ceil(len(top_errors) / cols))
    _setup_matplotlib()
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.7, rows_n * 2.7))
    axes = np.array(axes).reshape(rows_n, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for i, err in enumerate(top_errors):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(_denormalize_cifar_tensor(err["image"]))
        ax.set_title(
            f"idx={err['sample_idx']}\n{CIFAR10_CLASS_NAMES[err['true']]}->{CIFAR10_CLASS_NAMES[err['pred']]}\nconf={err['confidence']:.3f}",
            fontsize=7,
        )
        ax.axis("off")
    fig.suptitle(f"Top-{len(top_errors)} Highest-Confidence Errors", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return top_errors


def _compute_gradcam_overlay(model: torch.nn.Module, layer: torch.nn.Module, image: torch.Tensor, class_idx: int) -> np.ndarray:
    acts = []
    grads = []

    def _fwd_hook(_module, _inp, out):
        acts.append(out.detach())

    def _bwd_hook(_module, _gin, gout):
        grads.append(gout[0].detach())

    h1 = layer.register_forward_hook(_fwd_hook)
    h2 = layer.register_full_backward_hook(_bwd_hook)
    try:
        logits = model(image)
        score = logits[0, class_idx]
        model.zero_grad(set_to_none=True)
        score.backward()
    finally:
        h1.remove()
        h2.remove()

    act = acts[-1]
    grad = grads[-1]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    heat = cam.detach().cpu().numpy()
    img = _denormalize_cifar_tensor(image[0].detach().cpu())
    color = plt.get_cmap("jet")(heat)[..., :3]
    overlay = np.clip(0.45 * color + 0.55 * img, 0, 1)
    return overlay


def plot_analysis_gradcam(
    cifar_root: Path,
    out_path: Path,
    run_dir: Path | None,
    topk: int = ANALYSIS_TOPK_ERRORS,
    cam_k: int = ANALYSIS_GRADCAM_K,
    layer_name: str = ANALYSIS_GRADCAM_LAYER,
) -> None:
    run = _resolve_analysis_run_dir(cifar_root, run_dir)
    device = get_cifar_device()
    model, _cfg = _load_model_from_run(run, device)
    layer = _get_module_by_name(model, layer_name)
    top_errors = _collect_topk_confidence_errors(run, topk)[:cam_k]
    if not top_errors:
        raise ValueError("No misclassified samples available for Grad-CAM.")

    _setup_matplotlib()
    fig, axes = plt.subplots(1, len(top_errors), figsize=(len(top_errors) * 3.0, 3.6))
    if len(top_errors) == 1:
        axes = [axes]
    for ax, err in zip(axes, top_errors):
        x = err["image"].unsqueeze(0).to(device)
        overlay = _compute_gradcam_overlay(model, layer, x, err["pred"])
        ax.imshow(overlay)
        ax.set_title(
            f"idx={err['sample_idx']}\n{CIFAR10_CLASS_NAMES[err['true']]}->{CIFAR10_CLASS_NAMES[err['pred']]}\nconf={err['confidence']:.3f}",
            fontsize=7,
        )
        ax.axis("off")
    fig.suptitle(f"Grad-CAM on Top-{len(top_errors)} High-Confidence Errors", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _import_vgg_modules():
    """Load VGG_BatchNorm helpers without clobbering CIFAR10_NeuroNet imports."""
    global _VGG_MODULE_CACHE
    if _VGG_MODULE_CACHE is not None:
        return _VGG_MODULE_CACHE

    import importlib.util
    import types

    saved_modules = {name: sys.modules[name] for name in ("utils", "paths") if name in sys.modules}
    try:
        nn_spec = importlib.util.spec_from_file_location(
            "vgg_plot_utils_nn",
            VGG_CODE_DIR / "utils" / "nn.py",
        )
        vgg_nn = importlib.util.module_from_spec(nn_spec)
        nn_spec.loader.exec_module(vgg_nn)

        vgg_utils = types.ModuleType("utils")
        vgg_utils.__path__ = [str(VGG_CODE_DIR / "utils")]
        sys.modules["utils"] = vgg_utils
        sys.modules["utils.nn"] = vgg_nn

        paths_spec = importlib.util.spec_from_file_location(
            "vgg_plot_paths",
            VGG_CODE_DIR / "paths.py",
        )
        vgg_paths = importlib.util.module_from_spec(paths_spec)
        paths_spec.loader.exec_module(vgg_paths)
        sys.modules["paths"] = vgg_paths

        vgg_spec = importlib.util.spec_from_file_location(
            "vgg_plot_models_vgg",
            VGG_CODE_DIR / "models" / "vgg.py",
        )
        vgg_mod = importlib.util.module_from_spec(vgg_spec)
        vgg_spec.loader.exec_module(vgg_mod)

        data_spec = importlib.util.spec_from_file_location(
            "vgg_plot_data_split",
            VGG_CODE_DIR / "data_split.py",
        )
        vgg_data = importlib.util.module_from_spec(data_spec)
        data_spec.loader.exec_module(vgg_data)
    finally:
        for name, module in saved_modules.items():
            sys.modules[name] = module
        for name in ("utils", "paths"):
            if name not in saved_modules and name in sys.modules:
                del sys.modules[name]
        if "utils" not in saved_modules and "utils.nn" in sys.modules:
            del sys.modules["utils.nn"]

    _VGG_MODULE_CACHE = (vgg_mod.build_model, vgg_data.get_dataloaders, vgg_paths.get_device)
    return _VGG_MODULE_CACHE


_VGG_MODULE_CACHE = None


def resolve_vgg_bn_compare_run_dirs(
    vgg_root: Path,
    overrides: dict[str, Path] | None = None,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for model_name in ("VGG_A", "VGG_A_BatchNorm"):
        if overrides and model_name in overrides and overrides[model_name] is not None:
            resolved[model_name] = Path(overrides[model_name])
            continue
        default_dir = DEFAULT_VGG_BN_COMPARE_RUN_DIRS.get(model_name)
        if default_dir is not None and default_dir.is_dir():
            resolved[model_name] = default_dir
            continue
        resolved[model_name] = discover_run_dir(vgg_root, VGG_BN_COMPARE_PREFIX[model_name])
    return resolved


def _load_vgg_from_run(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    build_vgg_model, _, _ = _import_vgg_modules()
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_vgg_model(cfg["model_name"], num_classes=cfg["num_classes"])
    ckpt_path = run_dir / "best.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, cfg


def _get_vgg_dataloaders(cfg: dict):
    _, get_vgg_dataloaders, _ = _import_vgg_modules()
    loader_cfg = dict(cfg)
    loader_cfg["num_workers"] = 0
    return get_vgg_dataloaders(loader_cfg)


def _collect_vgg_topk_errors(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    k: int,
) -> list[dict]:
    errors: list[dict] = []
    sample_idx = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for b in range(x.size(0)):
                y_true = int(y[b].item())
                y_pred = int(pred[b].item())
                if y_true != y_pred:
                    errors.append(
                        {
                            "sample_idx": sample_idx,
                            "true": y_true,
                            "pred": y_pred,
                            "confidence": float(conf[b].item()),
                            "image": x[b].detach().cpu(),
                        }
                    )
                sample_idx += 1
    errors.sort(key=lambda item: item["confidence"], reverse=True)
    return errors[:k]


def _extract_vgg_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    layer_indices: dict[str, int],
) -> dict[str, torch.Tensor]:
    storage: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            storage[name] = out.detach()
        return hook

    for name, idx in layer_indices.items():
        handles.append(model.features[idx].register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return storage


def _feat_mean_map(act: torch.Tensor) -> np.ndarray:
    m = act[0].mean(dim=0).cpu().numpy()
    lo, hi = float(m.min()), float(m.max())
    if hi > lo:
        m = (m - lo) / (hi - lo)
    else:
        m = np.zeros_like(m)
    return m


@dataclass
class _VggStageStatsAcc:
    sum_abs: float = 0.0
    sum_val: float = 0.0
    sum_sq: float = 0.0
    count: int = 0
    positive: int = 0
    channel_std_sum: float = 0.0
    n_samples: int = 0


def _mean_channel_std(tensor: torch.Tensor) -> float:
    batch_size = tensor.shape[0]
    spatial = tensor.view(batch_size, tensor.shape[1], -1)
    if spatial.shape[2] > 1:
        return spatial.std(dim=2).mean().item()
    # 1x1 maps: per-channel std across the batch.
    return spatial.squeeze(2).std(dim=0).mean().item()


def _update_vgg_stage_stats(acc: _VggStageStatsAcc, tensor: torch.Tensor) -> None:
    t = tensor.detach()
    acc.sum_abs += t.abs().sum().item()
    acc.sum_val += t.sum().item()
    acc.sum_sq += (t * t).sum().item()
    acc.count += t.numel()
    acc.positive += int((t > 0).sum().item())
    batch_size = t.shape[0]
    acc.channel_std_sum += _mean_channel_std(t) * batch_size
    acc.n_samples += batch_size


def _finalize_vgg_stage_stats(acc: _VggStageStatsAcc) -> dict[str, float]:
    if acc.count == 0:
        return {
            "mean_abs": 0.0,
            "std_global": 0.0,
            "mean_channel_std": 0.0,
            "positive_ratio": 0.0,
        }
    mean = acc.sum_val / acc.count
    var = acc.sum_sq / acc.count - mean * mean
    return {
        "mean_abs": acc.sum_abs / acc.count,
        "std_global": float(np.sqrt(max(var, 0.0))),
        "mean_channel_std": acc.channel_std_sum / max(acc.n_samples, 1),
        "positive_ratio": acc.positive / acc.count,
    }


def _compute_vgg_activation_stats(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    layer_specs: tuple[tuple[str, int, int], ...],
    model_label: str,
    hist_stage: str = "S3",
    hist_max: int = VGG_HIST_SAMPLE_MAX,
) -> tuple[list[dict], np.ndarray]:
    use_bn = "BatchNorm" in model_label
    accs = {name: _VggStageStatsAcc() for name, _, _ in layer_specs}
    hist_vals: list[np.ndarray] = []
    hist_count = 0

    model.eval()
    with torch.no_grad():
        for x, _y in val_loader:
            x = x.to(device)
            feats = _extract_vgg_features(
                model,
                x,
                {name: (bn_idx if use_bn else idx) for name, idx, bn_idx in layer_specs},
            )
            for name, _, _ in layer_specs:
                _update_vgg_stage_stats(accs[name], feats[name])
            if hist_stage in feats and hist_count < hist_max:
                flat = feats[hist_stage].reshape(-1).cpu().numpy()
                remaining = hist_max - hist_count
                if flat.size > remaining:
                    rng = np.random.default_rng(2020)
                    pick = rng.choice(flat.size, size=remaining, replace=False)
                    flat = flat[pick]
                hist_vals.append(flat)
                hist_count += flat.size

    rows = []
    for name, _, _ in layer_specs:
        stats = _finalize_vgg_stage_stats(accs[name])
        rows.append({"model": model_label, "stage": name, **stats})
    hist_array = np.concatenate(hist_vals) if hist_vals else np.array([], dtype=np.float64)
    return rows, hist_array


def _save_vgg_stats_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["model", "stage", "mean_abs", "std_global", "mean_channel_std", "positive_ratio"]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["model"],
                    row["stage"],
                    f"{row['mean_abs']:.6f}",
                    f"{row['std_global']:.6f}",
                    f"{row['mean_channel_std']:.6f}",
                    f"{row['positive_ratio']:.6f}",
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_vgg_pair(
    vgg_root: Path,
    vgg_run_dirs: dict[str, Path] | None,
) -> tuple[torch.nn.Module, torch.nn.Module, dict, torch.device]:
    _, _, get_vgg_device = _import_vgg_modules()
    device = get_vgg_device()
    run_dirs = resolve_vgg_bn_compare_run_dirs(vgg_root, overrides=vgg_run_dirs)
    model_a, cfg_a = _load_vgg_from_run(run_dirs["VGG_A"], device)
    model_bn, _cfg_bn = _load_vgg_from_run(run_dirs["VGG_A_BatchNorm"], device)
    return model_a, model_bn, cfg_a, device


def plot_vgg_feature_maps(
    vgg_root: Path,
    out_path: Path,
    *,
    vgg_run_dirs: dict[str, Path] | None = None,
    topk: int = VGG_FEATURE_TOPK,
) -> list[dict]:
    model_a, model_bn, cfg_a, device = _load_vgg_pair(vgg_root, vgg_run_dirs)
    _, _, test_loader, _ = _get_vgg_dataloaders(cfg_a)
    top_errors = _collect_vgg_topk_errors(model_a, test_loader, device, topk)
    if not top_errors:
        raise ValueError("No misclassified samples found for VGG-A on the test set.")

    layer_a = {name: idx for name, idx, _ in VGG_FEATURE_LAYER_SPECS}
    layer_bn = {name: bn_idx for name, _, bn_idx in VGG_FEATURE_LAYER_SPECS}

    _setup_matplotlib()
    n_rows = len(top_errors)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    col_titles = [
        "Input",
        "VGG-A S1",
        "VGG-A+BN S1",
        "VGG-A S3",
        "VGG-A+BN S3",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9)

    for row, err in enumerate(top_errors):
        x = err["image"].unsqueeze(0).to(device)
        feats_a = _extract_vgg_features(model_a, x, layer_a)
        feats_bn = _extract_vgg_features(model_bn, x, layer_bn)

        ax_in = axes[row, 0]
        ax_in.imshow(_denormalize_cifar_tensor(err["image"]))
        ax_in.set_ylabel(
            f"idx={err['sample_idx']}\n"
            f"{CIFAR10_CLASS_NAMES[err['true']]}->{CIFAR10_CLASS_NAMES[err['pred']]}\n"
            f"conf={err['confidence']:.3f}",
            fontsize=7,
        )
        ax_in.set_xticks([])
        ax_in.set_yticks([])

        panels = [
            (1, feats_a["S1"], "viridis"),
            (2, feats_bn["S1"], "viridis"),
            (3, feats_a["S3"], "magma"),
            (4, feats_bn["S3"], "magma"),
        ]
        for col, feat, cmap in panels:
            ax = axes[row, col]
            ax.imshow(_feat_mean_map(feat), cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "2.3.3 Feature maps on VGG-A top high-confidence test errors (channel mean)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return top_errors


def plot_vgg_activation_stats(
    vgg_root: Path,
    out_path: Path,
    *,
    vgg_run_dirs: dict[str, Path] | None = None,
    csv_path: Path | None = None,
) -> list[dict]:
    model_a, model_bn, cfg_a, device = _load_vgg_pair(vgg_root, vgg_run_dirs)
    _, val_loader, _, _ = _get_vgg_dataloaders(cfg_a)

    rows_a, hist_a = _compute_vgg_activation_stats(
        model_a, val_loader, device, VGG_STATS_LAYER_SPECS, "VGG_A",
    )
    rows_bn, hist_bn = _compute_vgg_activation_stats(
        model_bn, val_loader, device, VGG_STATS_LAYER_SPECS, "VGG_A_BatchNorm",
    )
    all_rows = rows_a + rows_bn
    stats_path = csv_path if csv_path is not None else VGG_STATS_CSV_PATH
    _save_vgg_stats_csv(all_rows, stats_path)

    stages = [name for name, _, _ in VGG_STATS_LAYER_SPECS]
    x_pos = np.arange(len(stages))

    def _metric_series(model_label: str, metric: str) -> list[float]:
        lookup = {(r["model"], r["stage"]): r[metric] for r in all_rows}
        return [lookup[(model_label, stage)] for stage in stages]

    _setup_matplotlib()
    fig = plt.figure(figsize=(11, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax_line = fig.add_subplot(gs[0, 0])
    gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 2.3], hspace=0.38)
    ax_zero = fig.add_subplot(gs_right[0, 0])
    ax_hist = fig.add_subplot(gs_right[1, 0])

    ax_line.plot(
        x_pos,
        _metric_series("VGG_A", "mean_channel_std"),
        color="#2ca02c",
        marker="o",
        lw=1.8,
        label="VGG-A",
    )
    ax_line.plot(
        x_pos,
        _metric_series("VGG_A_BatchNorm", "mean_channel_std"),
        color="#d62728",
        marker="o",
        lw=1.8,
        label="VGG-A+BN",
    )
    ax_line.set_xticks(x_pos)
    ax_line.set_xticklabels(stages)
    ax_line.set_xlabel("Stage (post-pool)")
    ax_line.set_ylabel("Mean channel std")
    ax_line.set_title("Activation spread across stages (validation set)")
    ax_line.grid(True, alpha=0.3)
    ax_line.legend(loc="best", fontsize=8)

    if hist_a.size and hist_bn.size:
        zero_a = float((hist_a <= 0).mean())
        zero_bn = float((hist_bn <= 0).mean())
        x_bar = np.arange(2)
        ax_zero.bar(
            x_bar - 0.18,
            [zero_a, zero_bn],
            width=0.36,
            color=["#2ca02c", "#d62728"],
            alpha=0.85,
        )
        ax_zero.set_xticks(x_bar)
        ax_zero.set_xticklabels(["VGG-A", "VGG-A+BN"], fontsize=8)
        ax_zero.set_ylim(0.0, 1.0)
        ax_zero.set_ylabel("Fraction at zero")
        ax_zero.set_title("S3 zero activation fraction (validation set)")
        ax_zero.grid(True, axis="y", alpha=0.3)

        pos_a = hist_a[hist_a > 0]
        pos_bn = hist_bn[hist_bn > 0]
        if pos_a.size and pos_bn.size:
            combined_pos = np.concatenate([pos_a, pos_bn])
            hi = float(np.percentile(combined_pos, 99))
            if hi <= 0:
                hi = max(float(combined_pos.max()), 1e-6)
            bins = np.linspace(0.0, hi, 40)
            ax_hist.hist(
                pos_a[pos_a <= hi],
                bins=bins,
                density=True,
                alpha=0.55,
                color="#2ca02c",
                label="VGG-A",
            )
            ax_hist.hist(
                pos_bn[pos_bn <= hi],
                bins=bins,
                density=True,
                alpha=0.55,
                color="#d62728",
                label="VGG-A+BN",
            )
            ax_hist.set_xlim(0.0, hi)
    ax_hist.set_xlabel("Activation value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("S3 histogram for activations > 0 (x capped at p99)")
    ax_hist.legend(loc="best", fontsize=8)
    ax_hist.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return all_rows


def _setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _series_colors(n: int) -> list:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _resolve_output_path(
    fig: FigureSpec,
    out_dir: Path | None,
    name: str | None,
    output: Path | None,
) -> Path:
    if output is not None:
        return output
    directory = out_dir if out_dir is not None else DEFAULT_PIC_DIR
    filename = name if name is not None else fig.default_name
    return directory / filename


def plot_curves_figure(
    fig: FigureSpec,
    cifar_root: Path,
    vgg_root: Path,
    out_path: Path,
) -> None:
    root = cifar_root if fig.source == "cifar" else vgg_root
    curves_list: list[tuple[str, dict[str, np.ndarray]]] = []
    for spec in fig.series:
        run_dir = discover_run_dir(root, spec.prefix)
        curves_list.append((spec.label, load_curves(run_dir)))

    _setup_matplotlib()
    n_panels = len(fig.panels)
    fig_obj, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    colors = _series_colors(len(curves_list))
    legend_handles: list[Line2D] = []

    for panel_idx, panel in enumerate(fig.panels):
        ax = axes[panel_idx]
        for color, (label, curves) in zip(colors, curves_list):
            epoch = curves["epoch"]
            if panel == "loss":
                ax.plot(epoch, curves["train_loss"], color=color, lw=1.2, ls="-")
                ax.plot(epoch, curves["val_loss"], color=color, lw=1.2, ls="--")
                ax.set_ylabel("Loss")
            else:
                ax.plot(epoch, curves["train_acc"], color=color, lw=1.2, ls="-")
                ax.plot(epoch, curves["val_acc"], color=color, lw=1.2, ls="--")
                ax.set_ylabel("Accuracy")
            if panel_idx == 0:
                legend_handles.append(Line2D([0], [0], color=color, lw=1.5, label=label))

        ax.grid(True, alpha=0.3)
        ax.set_title(f"{fig.title} — {'Loss' if panel == 'loss' else 'Accuracy'}")

    style_note = Line2D([0], [0], color="gray", lw=0, label="solid=train, dashed=val")
    legend_handles.append(style_note)
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(handles=legend_handles, loc="best", fontsize=8)
    fig_obj.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_obj.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_obj)


def _landscape_run_dirs(vgg_root: Path, model_name: str) -> list[Path]:
    dirs = []
    for lr in LANDSCAPE_LRS:
        tag = _lr_tag(lr)
        prefix = f"{model_name}_loss_landscape_fix_{tag}_"
        dirs.append(discover_run_dir(vgg_root, prefix))
    return dirs


def _loss_envelope(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = [load_steps(rd) for rd in run_dirs]
    ref_steps = series[0][0]
    losses = [series[0][1]]
    for steps, loss in series[1:]:
        if len(steps) != len(ref_steps) or not np.array_equal(steps, ref_steps):
            raise ValueError("global_step mismatch across landscape runs")
        losses.append(loss)
    stacked = np.stack(losses, axis=0)
    return ref_steps, stacked.max(axis=0), stacked.min(axis=0)


def _probe_envelope(
    run_dirs: list[Path],
    value_key: Literal["ratio", "grad_diff"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = [load_grad_probe(rd) for rd in run_dirs]
    ref_steps = series[0][0]
    idx = 1 if value_key == "ratio" else 2
    values = [series[0][idx]]
    for item in series[1:]:
        steps, ratio, g_diff = item
        if len(steps) != len(ref_steps) or not np.array_equal(steps, ref_steps):
            raise ValueError("global_step mismatch across landscape runs")
        values.append(item[idx])
    stacked = np.stack(values, axis=0)
    return ref_steps, stacked.max(axis=0), stacked.min(axis=0)


def _probe_envelope_by_epoch(
    run_dirs: list[Path],
    value_key: Literal["ratio", "grad_diff"],
    segments_per_epoch: int = GRAD_DIFF_SEGMENTS_PER_EPOCH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = [
        _aggregate_probe_run(rd, value_key, segments_per_epoch=segments_per_epoch)
        for rd in run_dirs
    ]
    ref_steps = series[0][0]
    vals = [series[0][1]]
    for steps, v in series[1:]:
        if len(steps) != len(ref_steps) or not np.array_equal(steps, ref_steps):
            raise ValueError("epoch-aligned step sequences differ across landscape runs")
        vals.append(v)
    stacked = np.stack(vals, axis=0)
    return ref_steps, stacked.max(axis=0), stacked.min(axis=0)


def _downsample_envelope(
    steps: np.ndarray,
    max_curve: np.ndarray,
    min_curve: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stride <= 1:
        return steps, max_curve, min_curve
    sl = slice(None, None, stride)
    return steps[sl], max_curve[sl], min_curve[sl]


def _smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean; dampens per-step noise in grad_sweep plots."""
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or len(y) < 3:
        return y
    w = int(window)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(y, kernel, mode="same")


def _prepare_grad_sweep_curves(
    steps: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    stride: int,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, lo, hi = _downsample_envelope(steps, lo, hi, stride)
    lo = _smooth_1d(lo, smooth_window)
    hi = _smooth_1d(hi, smooth_window)
    lo = np.minimum(lo, hi)
    return steps, lo, hi


def _robust_ylim_grad_sweep(
    envelopes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    warmup: int = WARMUP_STEPS,
    percentile: float = GRAD_SWEEP_Y_PERCENTILE,
) -> tuple[float, float]:
    """Cap y-axis using high percentiles so rare spikes do not flatten the curves."""
    highs, lows = [], []
    for steps, max_curve, min_curve in envelopes:
        mask = steps >= warmup
        if mask.any():
            highs.append(max_curve[mask])
            lows.append(min_curve[mask])
    if not highs:
        return 0.0, 1.0
    y_top = float(np.percentile(np.concatenate(highs), percentile)) * 1.06
    y_bot = float(np.percentile(np.concatenate(lows), 100 - percentile))
    y_bot = max(0.0, min(y_bot, y_top * 0.35))
    return y_bot, max(y_top, 0.05)


def _robust_ylim_loss(
    envelopes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    highs = []
    for steps, max_curve, _ in envelopes:
        mask = steps >= WARMUP_STEPS
        if mask.any():
            highs.append(max_curve[mask])
    if not highs:
        return 0.0, 2.5
    pooled = np.concatenate(highs)
    y_top = float(np.percentile(pooled, Y_TOP_PERCENTILE)) * Y_TOP_MARGIN
    return 0.0, max(y_top, 0.5)


def _robust_ylim_generic(
    envelopes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    default_top: float,
    warmup: int = WARMUP_STEPS,
) -> tuple[float, float]:
    highs, lows = [], []
    for steps, max_curve, min_curve in envelopes:
        mask = steps >= warmup
        if mask.any():
            highs.append(max_curve[mask])
            lows.append(min_curve[mask])
    if not highs:
        return 0.0, default_top
    y_top = float(np.percentile(np.concatenate(highs), Y_TOP_PERCENTILE)) * Y_TOP_MARGIN
    y_bot = float(np.percentile(np.concatenate(lows), 100 - Y_TOP_PERCENTILE))
    y_bot = min(y_bot, y_top * 0.5)
    return max(0.0, y_bot * 0.92), max(y_top, default_top * 0.1)


def plot_landscape_envelope(
    fig: FigureSpec,
    vgg_root: Path,
    out_path: Path,
    *,
    value_key: Literal["loss", "ratio", "grad_diff"],
    ylabel: str,
    title_suffix: str,
    ylim_fn: Callable[[list], tuple[float, float]],
    stride: int = LANDSCAPE_STRIDE,
    ylim_fixed: tuple[float, float] | None = None,
    aggregate_by_epoch: bool = False,
    segments_per_epoch: int = GRAD_DIFF_SEGMENTS_PER_EPOCH,
) -> None:
    groups: dict[str, list[Path]] = {}
    missing: list[str] = []
    for model_name, _, _, _ in MODEL_LANDSCAPE_SPECS:
        try:
            dirs = _landscape_run_dirs(vgg_root, model_name)
            for rd in dirs:
                if value_key == "loss":
                    load_steps(rd)
                else:
                    load_grad_probe(rd)
            groups[model_name] = dirs
        except (FileNotFoundError, KeyError) as exc:
            missing.append(str(exc))
    if missing:
        raise FileNotFoundError(
            "Missing landscape runs or npz files:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    _setup_matplotlib()
    fig_obj, ax = plt.subplots(figsize=(9, 5))
    envelopes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    legend_handles: list[Line2D] = []

    for model_name, legend_label, fill_color, line_color in MODEL_LANDSCAPE_SPECS:
        run_dirs = groups[model_name]
        if value_key == "loss":
            env = _loss_envelope(run_dirs)
            steps, max_curve, min_curve = _downsample_envelope(*env, stride)
        elif aggregate_by_epoch:
            env = _probe_envelope_by_epoch(run_dirs, value_key, segments_per_epoch=segments_per_epoch)
            steps, max_curve, min_curve = env
        elif value_key == "ratio":
            env = _probe_envelope(run_dirs, "ratio")
            steps, max_curve, min_curve = _downsample_envelope(*env, stride)
        else:
            env = _probe_envelope(run_dirs, "grad_diff")
            steps, max_curve, min_curve = _downsample_envelope(*env, stride)
        envelopes.append((steps, max_curve, min_curve))
        ax.plot(steps, max_curve, color=line_color, lw=PLOT_LW)
        ax.plot(steps, min_curve, color=line_color, lw=PLOT_LW, linestyle="--")
        ax.fill_between(steps, min_curve, max_curve, color=fill_color, alpha=PLOT_FILL_ALPHA, linewidth=0)
        legend_handles.append(
            Line2D([0], [0], color=line_color, lw=LEGEND_LW, label=f"{legend_label} max")
        )
        legend_handles.append(
            Line2D(
                [0], [0], color=line_color, lw=LEGEND_LW, linestyle="--", label=f"{legend_label} min"
            )
        )

    if ylim_fixed is not None:
        y0, y1 = ylim_fixed
    else:
        y0, y1 = ylim_fn(envelopes)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{fig.title} — {title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig_obj.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_obj.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_obj)


def plot_grad_sweep_envelope(
    fig: FigureSpec,
    vgg_root: Path,
    out_path: Path,
    *,
    use_max_only: bool,
    ylabel: str,
    title_suffix: str,
    stride: int = GRAD_SWEEP_STRIDE,
    smooth_window: int = GRAD_SWEEP_SMOOTH_WINDOW,
    grad_run_dirs: dict[str, Path] | None = None,
    lr: float = GRAD_SWEEP_LR,
    max_epochs: int = GRAD_SWEEP_EPOCHS,
) -> None:
    """Plot distance-sweep grad_diff min/max (or max-only) from grad_sweep.npz."""
    run_dirs = resolve_grad_sweep_run_dirs(
        vgg_root, overrides=grad_run_dirs, lr=lr, max_epochs=max_epochs,
    )
    missing: list[str] = []
    series: list[tuple[str, str, str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for model_name, legend_label, fill_color, line_color in MODEL_LANDSCAPE_SPECS:
        rd = run_dirs[model_name]
        try:
            steps, lo, hi = load_grad_sweep(rd)
            steps, lo, hi = _prepare_grad_sweep_curves(
                steps, lo, hi, stride=stride, smooth_window=smooth_window,
            )
            series.append(
                (legend_label, fill_color, line_color, model_name, steps, lo, hi)
            )
        except (FileNotFoundError, KeyError) as exc:
            missing.append(f"{model_name} ({rd}): {exc}")
    if missing:
        raise FileNotFoundError(
            "Missing grad_sweep.npz in grad_probe runs:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    _setup_matplotlib()
    fig_obj, ax = plt.subplots(figsize=(9, 4.8))
    legend_handles: list[Line2D] = []
    envelopes = []

    # Draw Standard VGG under +BN so the red curve stays visible on top.
    for legend_label, fill_color, line_color, _mn, steps, lo, hi in series:
        envelopes.append((steps, hi, lo))
        z = 3 if "BatchNorm" in legend_label else 2
        if use_max_only:
            ax.plot(
                steps, hi, color=line_color, lw=GRAD_SWEEP_LW,
                solid_capstyle="round", zorder=z,
            )
            legend_handles.append(
                Line2D([0], [0], color=line_color, lw=LEGEND_LW, label=legend_label)
            )
        else:
            ax.fill_between(
                steps, lo, hi, color=fill_color, alpha=GRAD_SWEEP_FILL_ALPHA,
                linewidth=0, zorder=z - 1,
            )
            ax.plot(
                steps, hi, color=line_color, lw=GRAD_SWEEP_LW,
                solid_capstyle="round", zorder=z,
            )
            legend_handles.append(
                Line2D([0], [0], color=line_color, lw=LEGEND_LW, label=legend_label)
            )

    y0, y1 = _robust_ylim_grad_sweep(envelopes)
    ax.set_ylim(y0, y1)
    ax.set_xlim(envelopes[0][0].min(), envelopes[0][0].max())
    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title_suffix, fontsize=12, pad=10)
    ax.grid(True, alpha=0.22, linestyle="-", linewidth=0.6)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.92)
    fig_obj.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_obj.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_obj)


def build_figure_registry() -> dict[str, FigureSpec]:
    s = SeriesSpec
    return {
        "1_3_2": FigureSpec(
            "1_3_2", "1_3_2.png", "1.3.2 Baseline",
            "cifar", ("loss", "acc"), (s("Baseline", BASELINE_PREFIX),),
        ),
        "1_4_2": FigureSpec(
            "1_4_2", "1_4_2.png", "1.4.2 Width Ablation",
            "cifar", ("loss", "acc"),
            (
                s("Narrow", "width_full-w32-64-128-256_ep200"),
                s("Baseline", BASELINE_PREFIX),
                s("Wide", "width_full-w96-192-384-768_ep200"),
            ),
        ),
        "1_5_2": FigureSpec(
            "1_5_2", "1_5_2.png", "1.5.2 Loss Function Ablation",
            "cifar", ("acc",),
            (
                s("CE (baseline)", BASELINE_PREFIX),
                s("Focal Loss", "loss_full-focal_g2_wd5e4_ep200"),
                s("Multi-Margin Loss", "loss_full-multimargin_m1_wd5e4_ep200"),
            ),
        ),
        "1_6_2_1": FigureSpec(
            "1_6_2_1", "1_6_2_1.png", "1.6.2.1 L2 Regularization Strength",
            "cifar", ("loss", "acc"),
            (
                s("No weight decay", "loss_full-ce_wd0_ep200"),
                s("Baseline (wd=5e-4)", BASELINE_PREFIX),
                s("Strong weight decay (wd=1e-3)", "loss_full-ce_wd1e3_ep200"),
            ),
        ),
        "1_6_2_2": FigureSpec(
            "1_6_2_2", "1_6_2_2.png", "1.6.2.2 Regularization Methods",
            "cifar", ("loss", "acc"),
            (
                s("Baseline", BASELINE_PREFIX),
                s("Label smoothing", "loss_full-ls0.1_wd5e4_ep200"),
                s("Mixup", "loss_full-mixup0.2_wd5e4_ep200"),
                s("CutMix", "loss_full-cutmix1.0_wd5e4_ep200"),
                s("Dropout", "loss_full-ce_wd5e4_drop0.5_ep200"),
            ),
        ),
        "1_7_2": FigureSpec(
            "1_7_2", "1_7_2.png", "1.7.2 Activation Functions",
            "cifar", ("loss", "acc"),
            (
                s("ReLU (baseline)", BASELINE_PREFIX),
                s("GELU", "act_full-gelu_ep200"),
                s("Leaky ReLU", "act_full-leaky_relu_ep200"),
            ),
        ),
        "1_8_2": FigureSpec(
            "1_8_2", "1_8_2.png", "1.8.2 Optimizers",
            "cifar", ("loss", "acc"),
            (
                s("SGD lr=0.05", "optim_full-sgd_lr0.05_ep200"),
                s("SGD lr=0.1", "optim_full-sgd_lr0.1_ep200"),
                s("SGD lr=0.2", "optim_full-sgd_lr0.2_ep200"),
                s("Adam lr=3e-4", "optim_full-adam_lr3e-4_ep200"),
                s("Adam lr=1e-3", "optim_full-adam_lr1e-3_ep200"),
                s("Adam lr=3e-3", "optim_full-adam_lr3e-3_ep200"),
                s("AdamW lr=3e-4", "optim_full-adamw_lr3e-4_ep200"),
                s("AdamW lr=1e-3 (baseline)", BASELINE_PREFIX),
                s("AdamW lr=3e-3", "optim_full-adamw_lr3e-3_ep200"),
            ),
        ),
        "1_8_2_best": FigureSpec(
            "1_8_2_best", "1_8_2_best.png", "1.8.2 Optimizers (Best Val Acc per Optimizer)",
            "cifar", ("loss", "acc"),
            (
                s("SGD lr=0.1 (best)", "optim_full-sgd_lr0.1_ep200"),
                s("Adam lr=1e-3 (best)", "optim_full-adam_lr1e-3_ep200"),
                s("AdamW lr=3e-3 (best)", "optim_full-adamw_lr3e-3_ep200"),
            ),
        ),
        "1_9_2": FigureSpec(
            "1_9_2", "1_9_2.png", "1.9.2 Optimal Model Exploration",
            "cifar", ("loss", "acc"),
            (
                s("基线", BASELINE_PREFIX),
                s("实验1", "combine-cutmix_w96_sgd0.1"),
                s("实验2", "combine-cutmix_w96_sgd0.05"),
                s("实验3", "combine-cutmix_w64_sgd0.1"),
                s("实验4", "combine-mixup_w64_sgd0.1"),
            ),
        ),
        "2_3_2": FigureSpec(
            "2_3_2", "2_3_2.png", "2.3.2 VGG-A vs VGG-A+BN",
            "vgg_curves", ("loss", "acc"),
            (
                s("VGG-A", "VGG_A_bn_compare_adamw1e-3_cos_ep200_seed2020"),
                s("VGG-A + BN", "VGG_A_BatchNorm_bn_compare_adamw1e-3_cos_ep200_seed2020"),
            ),
        ),
        "2_3_3_feature_maps": FigureSpec(
            "2_3_3_feature_maps",
            "2_3_3_feature_maps.png",
            "2.3.3 Feature Maps (VGG-A vs VGG-A+BN)",
            "vgg_feature_maps",
            (),
            (),
        ),
        "2_3_3_activation_stats": FigureSpec(
            "2_3_3_activation_stats",
            "2_3_3_activation_stats.png",
            "2.3.3 Activation Statistics",
            "vgg_activation_stats",
            (),
            (),
        ),
        "2_4_2_loss_landscape": FigureSpec(
            "2_4_2_loss_landscape", "2_4_2_loss_landscape.png", "2.4.2 Loss Landscape",
            "vgg_landscape_loss", (), (),
        ),
        "2_4_2_grad_predictability_landscape": FigureSpec(
            "2_4_2_grad_predictability_landscape",
            "2_4_2_grad_predictability_landscape.png",
            "2.4.2 Gradient Predictability",
            "vgg_grad_sweep_pred", (), (),
        ),
        "2_4_2_grad_diff_landscape": FigureSpec(
            "2_4_2_grad_diff_landscape",
            "2_4_2_grad_diff_landscape.png",
            "2.4.2 Gradient Difference (max over distance)",
            "vgg_grad_sweep_diff", (), (),
        ),
        "1_10_2_loss_landscape_sgd": FigureSpec(
            "1_10_2_loss_landscape_sgd",
            "1_10_3_loss_landscape_sgd.png",
            "1.10.2 Loss Landscape (SGD)",
            "analysis_landscape_sgd",
            (),
            (),
        ),
        "1_10_2_kernels": FigureSpec(
            "1_10_2_kernels",
            "1_10_2_kernels.png",
            "1.10.2 Kernel Visualization",
            "analysis_kernels",
            (),
            (),
        ),
        "1_10_2_confmat": FigureSpec(
            "1_10_2_confmat",
            "1_10_1_confmat.png",
            "1.10.2 Confusion Matrix",
            "analysis_confusion",
            (),
            (),
        ),
        "1_10_2_top10_errors": FigureSpec(
            "1_10_2_top10_errors",
            "1_10_1_top10_errors.png",
            "1.10.2 Top-10 Errors",
            "analysis_top_errors",
            (),
            (),
        ),
        "1_10_2_gradcam": FigureSpec(
            "1_10_2_gradcam",
            "1_10_1_gradcam.png",
            "1.10.2 Grad-CAM",
            "analysis_gradcam",
            (),
            (),
        ),
    }


FIGURES = build_figure_registry()
ALL_FIGURE_IDS = list(FIGURES.keys())


def _build_grad_run_dir_overrides(args: argparse.Namespace) -> dict[str, Path] | None:
    vgg_a = getattr(args, "grad_vgg_a_run_dir", None)
    vgg_bn = getattr(args, "grad_vgg_a_bn_run_dir", None)
    if vgg_a is None and vgg_bn is None:
        return None
    return {
        "VGG_A": Path(vgg_a) if vgg_a is not None else DEFAULT_GRAD_SWEEP_RUN_DIRS["VGG_A"],
        "VGG_A_BatchNorm": (
            Path(vgg_bn) if vgg_bn is not None else DEFAULT_GRAD_SWEEP_RUN_DIRS["VGG_A_BatchNorm"]
        ),
    }


def _build_vgg_bn_compare_run_dir_overrides(args: argparse.Namespace) -> dict[str, Path] | None:
    vgg_a = getattr(args, "vgg_a_run_dir", None)
    vgg_bn = getattr(args, "vgg_a_bn_run_dir", None)
    if vgg_a is None and vgg_bn is None:
        return None
    return {
        "VGG_A": Path(vgg_a) if vgg_a is not None else DEFAULT_VGG_BN_COMPARE_RUN_DIRS["VGG_A"],
        "VGG_A_BatchNorm": (
            Path(vgg_bn) if vgg_bn is not None else DEFAULT_VGG_BN_COMPARE_RUN_DIRS["VGG_A_BatchNorm"]
        ),
    }


def plot_one(
    fig_id: str,
    *,
    cifar_root: Path,
    vgg_root: Path,
    out_dir: Path | None,
    name: str | None,
    output: Path | None,
    grad_run_dirs: dict[str, Path] | None = None,
    vgg_bn_compare_run_dirs: dict[str, Path] | None = None,
    analysis_run_dir: Path | None = None,
    analysis_baseline_run_dir: Path | None = None,
    analysis_landscape_cutmix_run_dirs: list[Path] | None = None,
    analysis_landscape_nocutmix_run_dirs: list[Path] | None = None,
    topk_errors: int = ANALYSIS_TOPK_ERRORS,
    gradcam_k: int = ANALYSIS_GRADCAM_K,
    gradcam_layer: str = ANALYSIS_GRADCAM_LAYER,
    vgg_topk_errors: int = VGG_FEATURE_TOPK,
) -> Path:
    if fig_id not in FIGURES:
        raise KeyError(f"Unknown figure id: {fig_id}. Choose from: {', '.join(ALL_FIGURE_IDS)}")
    fig = FIGURES[fig_id]
    out_path = _resolve_output_path(fig, out_dir, name, output)

    if fig.source in ("cifar", "vgg_curves"):
        plot_curves_figure(fig, cifar_root, vgg_root, out_path)
    elif fig.source == "vgg_landscape_loss":
        plot_landscape_envelope(
            fig, vgg_root, out_path,
            value_key="loss", ylabel="Loss Landscape", title_suffix="Loss Landscape",
            ylim_fn=_robust_ylim_loss,
        )
    elif fig.source == "vgg_landscape_grad_pred":
        plot_landscape_envelope(
            fig, vgg_root, out_path,
            value_key="ratio", ylabel="Gradient Predictability",
            title_suffix="Gradient Predictability (legacy 4-lr)",
            ylim_fn=lambda env: _robust_ylim_generic(env, default_top=2.0),
            ylim_fixed=GRAD_PRED_YLIM,
            aggregate_by_epoch=True,
            segments_per_epoch=GRAD_PRED_SEGMENTS_PER_EPOCH,
        )
    elif fig.source == "vgg_landscape_grad_diff":
        plot_landscape_envelope(
            fig, vgg_root, out_path,
            value_key="grad_diff", ylabel="Gradient Difference",
            title_suffix="Gradient Difference (legacy 4-lr)",
            ylim_fn=lambda env: _robust_ylim_generic(env, default_top=1.0),
            aggregate_by_epoch=True,
            segments_per_epoch=GRAD_DIFF_SEGMENTS_PER_EPOCH,
        )
    elif fig.source == "vgg_grad_sweep_pred":
        plot_grad_sweep_envelope(
            fig, vgg_root, out_path,
            use_max_only=False,
            ylabel=r"$\|g' - g\|$",
            title_suffix="2.5.2 Gradient predictability (envelope over distance)",
            grad_run_dirs=grad_run_dirs,
        )
    elif fig.source == "vgg_grad_sweep_diff":
        plot_grad_sweep_envelope(
            fig, vgg_root, out_path,
            use_max_only=True,
            ylabel=r"$\max_{\alpha}\|g' - g\|$",
            title_suffix="2.5.2 Max gradient difference over distance",
            grad_run_dirs=grad_run_dirs,
        )
    elif fig.source == "analysis_landscape_sgd":
        plot_analysis_loss_landscape(
            cifar_root,
            out_path,
            cutmix_run_dirs=analysis_landscape_cutmix_run_dirs,
            nocutmix_run_dirs=analysis_landscape_nocutmix_run_dirs,
        )
    elif fig.source == "analysis_kernels":
        plot_analysis_kernels(
            cifar_root, out_path,
            run_dir=analysis_run_dir,
            baseline_run_dir=analysis_baseline_run_dir,
        )
    elif fig.source == "analysis_confusion":
        plot_analysis_confusion(cifar_root, out_path, run_dir=analysis_run_dir)
    elif fig.source == "analysis_top_errors":
        plot_analysis_top_errors(cifar_root, out_path, run_dir=analysis_run_dir, topk=topk_errors)
    elif fig.source == "analysis_gradcam":
        plot_analysis_gradcam(
            cifar_root, out_path,
            run_dir=analysis_run_dir,
            topk=topk_errors,
            cam_k=gradcam_k,
            layer_name=gradcam_layer,
        )
    elif fig.source == "vgg_feature_maps":
        plot_vgg_feature_maps(
            vgg_root,
            out_path,
            vgg_run_dirs=vgg_bn_compare_run_dirs,
            topk=vgg_topk_errors,
        )
    elif fig.source == "vgg_activation_stats":
        plot_vgg_activation_stats(
            vgg_root,
            out_path,
            vgg_run_dirs=vgg_bn_compare_run_dirs,
        )
    else:
        raise ValueError(f"Unknown figure source: {fig.source}")

    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot report.ipynb figures from experiment outputs.")
    parser.add_argument(
        "figures",
        nargs="+",
        help=f"Figure id(s) or 'all'. Available: {', '.join(ALL_FIGURE_IDS)}",
    )
    parser.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR_ROOT,
                        help=f"CIFAR10_NeuroNet output root (default: {DEFAULT_CIFAR_ROOT})")
    parser.add_argument("--vgg-root", type=Path, default=DEFAULT_VGG_ROOT,
                        help=f"VGG_BatchNorm output root (default: {DEFAULT_VGG_ROOT})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"Output directory (default: {DEFAULT_PIC_DIR})")
    parser.add_argument("--name", type=str, default=None,
                        help="Output filename for a single figure (requires exactly one figure id)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Full output path for a single figure (requires exactly one figure id)")
    parser.add_argument(
        "--grad-vgg-a-run-dir",
        type=Path,
        default=None,
        help=(
            "grad_probe run dir for VGG_A (must contain grad_sweep.npz); "
            f"default: {DEFAULT_GRAD_SWEEP_RUN_DIRS['VGG_A']}"
        ),
    )
    parser.add_argument(
        "--grad-vgg-a-bn-run-dir",
        type=Path,
        default=None,
        help=(
            "grad_probe run dir for VGG_A_BatchNorm (grad_sweep.npz); "
            f"default: {DEFAULT_GRAD_SWEEP_RUN_DIRS['VGG_A_BatchNorm']}"
        ),
    )
    parser.add_argument(
        "--analysis-run-dir",
        type=Path,
        default=None,
        help="Target model run dir for Task1 analysis figures (default: latest combine-cutmix_w96_sgd0.1-*)",
    )
    parser.add_argument(
        "--analysis-baseline-run-dir",
        type=Path,
        default=None,
        help=f"Baseline run dir for kernel comparison (default: latest {BASELINE_PREFIX}-*)",
    )
    parser.add_argument(
        "--analysis-landscape-cutmix-run-dirs",
        type=Path,
        nargs=4,
        default=None,
        help=(
            "Optional 4 run dirs (lr=0.05 0.1 0.15 0.2) for top CutMix landscape panel; "
            "default: latest analysis_landscape_sgd-cutmix_w96_sgd*_ep50-*"
        ),
    )
    parser.add_argument(
        "--analysis-landscape-nocutmix-run-dirs",
        type=Path,
        nargs=4,
        default=None,
        help=(
            "Optional 4 run dirs (lr=0.05 0.1 0.15 0.2) for bottom no-CutMix panel; "
            "default: latest analysis_landscape_sgd-nocutmix_w96_sgd*_ep50-*"
        ),
    )
    parser.add_argument(
        "--topk-errors",
        type=int,
        default=ANALYSIS_TOPK_ERRORS,
        help=f"Top-K high-confidence errors for analysis figures (default: {ANALYSIS_TOPK_ERRORS})",
    )
    parser.add_argument(
        "--gradcam-k",
        type=int,
        default=ANALYSIS_GRADCAM_K,
        help=f"How many top errors to render Grad-CAM for (default: {ANALYSIS_GRADCAM_K})",
    )
    parser.add_argument(
        "--gradcam-layer",
        type=str,
        default=ANALYSIS_GRADCAM_LAYER,
        help=f"Layer path for Grad-CAM (default: {ANALYSIS_GRADCAM_LAYER})",
    )
    parser.add_argument(
        "--vgg-a-run-dir",
        type=Path,
        default=None,
        help=(
            "bn_compare run dir for VGG_A (best.pt); "
            f"default: {DEFAULT_VGG_BN_COMPARE_RUN_DIRS['VGG_A']}"
        ),
    )
    parser.add_argument(
        "--vgg-a-bn-run-dir",
        type=Path,
        default=None,
        help=(
            "bn_compare run dir for VGG_A_BatchNorm (best.pt); "
            f"default: {DEFAULT_VGG_BN_COMPARE_RUN_DIRS['VGG_A_BatchNorm']}"
        ),
    )
    parser.add_argument(
        "--vgg-topk-errors",
        type=int,
        default=VGG_FEATURE_TOPK,
        help=f"Top-K high-confidence VGG-A test errors for 2.3.3 feature maps (default: {VGG_FEATURE_TOPK})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fig_ids = ALL_FIGURE_IDS if args.figures == ["all"] else args.figures

    if args.name is not None and len(fig_ids) != 1:
        print("ERROR: --name applies to a single figure only.", file=sys.stderr)
        return 1
    if args.output is not None and len(fig_ids) != 1:
        print("ERROR: --output applies to a single figure only.", file=sys.stderr)
        return 1

    grad_run_dirs = _build_grad_run_dir_overrides(args)
    vgg_bn_compare_run_dirs = _build_vgg_bn_compare_run_dir_overrides(args)

    ok, failed = 0, 0
    for fig_id in fig_ids:
        try:
            out_path = plot_one(
                fig_id,
                cifar_root=args.cifar_root,
                vgg_root=args.vgg_root,
                out_dir=args.out_dir,
                name=args.name,
                output=args.output,
                grad_run_dirs=grad_run_dirs,
                vgg_bn_compare_run_dirs=vgg_bn_compare_run_dirs,
                analysis_run_dir=args.analysis_run_dir,
                analysis_baseline_run_dir=args.analysis_baseline_run_dir,
                analysis_landscape_cutmix_run_dirs=args.analysis_landscape_cutmix_run_dirs,
                analysis_landscape_nocutmix_run_dirs=args.analysis_landscape_nocutmix_run_dirs,
                topk_errors=args.topk_errors,
                gradcam_k=args.gradcam_k,
                gradcam_layer=args.gradcam_layer,
                vgg_topk_errors=args.vgg_topk_errors,
            )
            print(f"OK  {fig_id} -> {out_path}")
            ok += 1
        except Exception as exc:
            print(f"FAIL {fig_id}: {exc}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"\nDone: {ok} saved, {failed} failed.", file=sys.stderr)
        return 1
    print(f"\nDone: {ok} figure(s) saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
