"""
Experiment artifacts live under ``codes/`` (same directory as this file's package root).

# English: Hierarchical layout (example for MLP + ``--preset reg_l2_early_stop``):

  codes/figs/mlp/Regularization/l2earlyStop/val_test_curves.png
  codes/best_models/mlp/Regularization/l2earlyStop/best_model.pickle
  codes/traininglogs/mlp/Regularization/l2earlyStop/experiment.json
  codes/traininglogs/mlp/Regularization/l2earlyStop/curves.npz
  codes/traininglogs/mlp/Regularization/l2earlyStop/training_log.txt

Baseline (no extra "研究方向" segment) uses a single folder under the model, e.g.
``codes/figs/mlp/baseline/val_test_curves.png``.

CNN runs use ``cnn/`` instead of ``mlp/``. If ``--experiment_slug`` is set together with a known
``--preset``, it replaces only the **leaf** path component (defaults come from the table below).
If only ``--experiment_slug`` is set, files go under ``{model}/Custom/<slug>/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

# English: Map ``--preset`` keys to path segments under ``{mlp|cnn}/`` (研究方向/…/叶子 for ablations;
# baseline is only ``{model}/baseline/`` with no extra category folder).
PRESET_TO_SUBPATH: dict[str, tuple[str, ...]] = {
    "baseline": ("baseline",),
    "lr_step": ("Optimization", "LRscheduling", "StepLR"),
    "lr_exp": ("Optimization", "LRscheduling", "ExponentialLR"),
    "momentum": ("Optimization", "momentum"),
    "reg_l2": ("Regularization", "l2"),
    "reg_early_stop": ("Regularization", "earlyStop"),
    "reg_l2_early_stop": ("Regularization", "l2earlyStop"),
}


def codes_root() -> Path:
    """``PJ1/codes/`` — directory that contains ``exp_artifacts.py``."""
    return Path(__file__).resolve().parent


def artifact_relative_dir(model_lower: str, preset_key: str, experiment_slug: str) -> Path:
    """
    # English: Relative directory under ``figs`` / ``traininglogs`` / ``best_models`` (same relpath for all).
    """
    m = model_lower.strip().lower()
    if m not in ("mlp", "cnn"):
        raise ValueError(f"model must be 'mlp' or 'cnn', got {model_lower!r}")
    pk = (preset_key or "").strip().lower()
    slug = (experiment_slug or "").strip().replace(" ", "_")

    if pk in PRESET_TO_SUBPATH:
        parts = list(PRESET_TO_SUBPATH[pk])
        if slug:
            parts[-1] = slug
        return Path(m, *parts)
    if slug:
        return Path(m) / "Custom" / slug
    raise ValueError("Need a non-empty --preset or --experiment_slug to build artifact paths.")


def layout_paths(model_lower: str, preset_key: str, experiment_slug: str) -> dict[str, Path]:
    """
    # English: Resolve ``figs``, ``traininglogs``, and ``best_models`` folders under ``codes/``.
    """
    root = codes_root()
    rel = artifact_relative_dir(model_lower, preset_key, experiment_slug)
    figs = root / "figs" / rel
    tlogs = root / "traininglogs" / rel
    weights = root / "best_models" / rel
    for p in (figs, tlogs, weights):
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root, "rel": rel, "figs": figs, "traininglogs": tlogs, "weights": weights}


def save_val_test_curves(runner, out_path: Path) -> None:
    """
    # English: Save loss/accuracy vs training iteration for train, validation (dev), and optional test.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    it = np.arange(len(runner.train_loss))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(it, runner.train_loss, label="train loss", color="#2E86AB")
    axes[0].plot(it, runner.dev_loss, label="val loss", color="#A23B72", linestyle="--")
    if getattr(runner, "test_loss", None) and len(runner.test_loss) == len(it):
        axes[0].plot(it, runner.test_loss, label="test loss", color="#F18F01", linestyle=":")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Loss")

    axes[1].plot(it, runner.train_scores, label="train acc", color="#2E86AB")
    axes[1].plot(it, runner.dev_scores, label="val acc", color="#A23B72", linestyle="--")
    if getattr(runner, "test_scores", None) and len(runner.test_scores) == len(it):
        axes[1].plot(it, runner.test_scores, label="test acc", color="#F18F01", linestyle=":")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(loc="lower right")
    axes[1].set_title("Accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_npz(runner, out_path: Path) -> None:
    """# English: Store raw curve arrays for the report / further plotting."""
    payload: dict[str, Any] = {
        "train_loss": np.asarray(runner.train_loss, dtype=np.float64),
        "dev_loss": np.asarray(runner.dev_loss, dtype=np.float64),
        "train_scores": np.asarray(runner.train_scores, dtype=np.float64),
        "dev_scores": np.asarray(runner.dev_scores, dtype=np.float64),
    }
    if getattr(runner, "test_loss", None):
        payload["test_loss"] = np.asarray(runner.test_loss, dtype=np.float64)
        payload["test_scores"] = np.asarray(runner.test_scores, dtype=np.float64)
    np.savez_compressed(out_path, **payload)


def save_experiment_json(meta: Mapping[str, Any], out_path: Path) -> None:
    """# English: Serializable hyper-parameters and outcomes (JSON-safe values only)."""
    def _jsonify(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x.resolve())
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, (list, tuple)):
            return [_jsonify(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _jsonify(v) for k, v in x.items()}
        return repr(x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(dict(meta)), f, indent=2, ensure_ascii=False)


def write_training_log_txt(lines: list[str], out_path: Path) -> None:
    """# English: Short human-readable run summary alongside JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
