"""Evaluate best checkpoint on the official CIFAR-10 test set."""
import json
from pathlib import Path

import torch

from data_split import get_dataloaders
from losses import get_criterion
from models.model import build_model
from paths import get_device
from train_utils import evaluate


def run_test_eval(run_dir: Path, test_loader=None, device=None, checkpoint_name: str = "best.pt"):
    run_dir = Path(run_dir)
    device = device or get_device()

    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["channels"] = tuple(cfg["channels"])
    cfg["blocks_per_stage"] = tuple(cfg["blocks_per_stage"])

    ckpt_path = run_dir / checkpoint_name
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    if test_loader is None:
        _, _, test_loader = get_dataloaders(cfg)

    criterion = get_criterion(cfg)
    test_metrics = evaluate(model, test_loader, criterion, device)

    results_path = run_dir / "results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    results.update({
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_error": 1.0 - test_metrics["acc"],
        "best_epoch": results.get("best_epoch", ckpt.get("epoch")),
        "best_val_acc": results.get("best_val_acc", ckpt.get("val_acc")),
    })

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"  test eval | loss {test_metrics['loss']:.4f} "
        f"acc {test_metrics['acc']:.4f} error {1 - test_metrics['acc']:.4f}"
    )
    return results
