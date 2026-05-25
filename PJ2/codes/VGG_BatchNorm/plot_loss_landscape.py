"""
Plot loss-landscape envelopes (section 2.4.2, approach A).

Reads steps.npz from 8 loss_landscape runs under outputs/VGG_BatchNorm,
aggregates per-step train_loss min/max over 4 fixed learning rates,
and saves pic/2_4_2_loss_landscape.png.

Usage (from repo root or this directory):
    python codes/VGG_BatchNorm/plot_loss_landscape.py
    python plot_loss_landscape.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import paths  # noqa: F401 — sets up sys.path
from config import LANDSCAPE_LRS
from paths import OUTPUT_ROOT, PROJECT_ROOT

# Match folder names: fix_1e-3_ep100, fix_2e-3_ep100, ...
LR_TAGS = []
for lr in LANDSCAPE_LRS:
    s = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    LR_TAGS.append(s)

MODEL_SPECS = [
    ("VGG_A", "Standard VGG", "#2ca02c", "#1a6b1a"),
    ("VGG_A_BatchNorm", "Standard VGG + BatchNorm", "#d62728", "#8b0000"),
]

OUT_PATH = PROJECT_ROOT / "pic" / "2_4_2_loss_landscape.png"

# y-axis: ignore first WARMUP_STEPS when auto-scaling (large-lr spikes at start)
WARMUP_STEPS = 20
Y_TOP_PERCENTILE = 99.5
Y_TOP_MARGIN = 1.08

PLOT_LW = 0.15
PLOT_FILL_ALPHA = 0.28
LEGEND_LW = 1.2  # legend only; curves stay at PLOT_LW


def _lr_tag(lr: float) -> str:
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def discover_run_dir(model_name: str, lr: float) -> Path:
    """Pick the newest loss_landscape run folder for (model, lr)."""
    tag = _lr_tag(lr)
    prefix = f"{model_name}_loss_landscape_fix_{tag}_"
    candidates = sorted(
        (p for p in OUTPUT_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directory under {OUTPUT_ROOT} matching prefix '{prefix}*'"
        )
    return candidates[-1]


def require_steps_npz(run_dir: Path) -> Path:
    path = run_dir / "steps.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def load_train_loss(steps_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(steps_path)
    if "global_step" not in data or "train_loss" not in data:
        raise KeyError(f"{steps_path} must contain global_step and train_loss")
    order = np.argsort(data["global_step"])
    steps = np.asarray(data["global_step"][order], dtype=np.int64)
    loss = np.asarray(data["train_loss"][order], dtype=np.float64)
    return steps, loss


def envelope(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step min/max train_loss across multiple lr runs."""
    series = []
    for rd in run_dirs:
        steps, loss = load_train_loss(require_steps_npz(rd))
        series.append((steps, loss))

    ref_steps = series[0][0]
    losses = [series[0][1]]
    for steps, loss in series[1:]:
        if len(steps) != len(ref_steps) or not np.array_equal(steps, ref_steps):
            raise ValueError(
                "global_step sequences differ across runs; "
                "ensure identical data split, batch size, and max_epochs."
            )
        losses.append(loss)

    stacked = np.stack(losses, axis=0)
    return ref_steps, stacked.max(axis=0), stacked.min(axis=0)


def collect_runs() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    missing = []
    for model_name, _, _, _ in MODEL_SPECS:
        dirs = []
        for lr in LANDSCAPE_LRS:
            try:
                rd = discover_run_dir(model_name, lr)
                require_steps_npz(rd)
                dirs.append(rd)
            except FileNotFoundError as e:
                missing.append(str(e))
        groups[model_name] = dirs

    if missing:
        msg = "Cannot plot loss landscape; missing runs or steps.npz:\n" + "\n".join(
            f"  - {m}" for m in missing
        )
        raise FileNotFoundError(msg)
    return groups


def _robust_ylim(
    envelopes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """Upper y from post-warmup envelope highs; avoids step-0/1 lr spikes blowing the scale."""
    highs = []
    for steps, max_curve, _ in envelopes:
        mask = steps >= WARMUP_STEPS
        if mask.any():
            highs.append(max_curve[mask])
    if not highs:
        return 0.0, 2.5
    pooled = np.concatenate(highs)
    y_top = float(np.percentile(pooled, Y_TOP_PERCENTILE)) * Y_TOP_MARGIN
    y_top = max(y_top, 0.5)
    return 0.0, y_top


def plot(groups: dict[str, list[Path]], out_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 5))
    envelopes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    legend_handles: list[Line2D] = []

    for model_name, legend_label, fill_color, line_color in MODEL_SPECS:
        steps, max_curve, min_curve = envelope(groups[model_name])
        envelopes.append((steps, max_curve, min_curve))
        ax.plot(steps, max_curve, color=line_color, lw=PLOT_LW)
        ax.plot(steps, min_curve, color=line_color, lw=PLOT_LW, linestyle="--")
        ax.fill_between(
            steps, min_curve, max_curve, color=fill_color, alpha=PLOT_FILL_ALPHA, linewidth=0
        )
        legend_handles.append(
            Line2D([0], [0], color=line_color, lw=LEGEND_LW, label=f"{legend_label} max")
        )
        legend_handles.append(
            Line2D(
                [0], [0], color=line_color, lw=LEGEND_LW, linestyle="--", label=f"{legend_label} min"
            )
        )

    y0, y1 = _robust_ylim(envelopes)
    ax.set_ylim(y0, y1)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss Landscape")
    ax.set_title("Loss Landscape")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not OUTPUT_ROOT.is_dir():
        print(f"ERROR: output root not found: {OUTPUT_ROOT}", file=sys.stderr)
        return 1

    try:
        groups = collect_runs()
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    for model_name, dirs in groups.items():
        print(f"{model_name}: {len(dirs)} runs")
        for d in dirs:
            print(f"  {d.name}")

    plot(groups, OUT_PATH)
    print(f"Saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
