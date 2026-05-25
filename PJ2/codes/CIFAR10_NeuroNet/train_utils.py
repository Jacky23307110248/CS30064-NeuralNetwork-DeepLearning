"""Training helpers: optimizer, metrics, IO."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["optimizer"].lower()
    lr = cfg["lr"]
    wd = cfg["weight_decay"]
    params = model.parameters()
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=wd,
            nesterov=True,
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    if not cfg.get("use_cosine_lr", True):
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["max_epochs"],
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


def save_config(cfg: dict, run_dir: Path):
    serializable = deepcopy(cfg)
    serializable["channels"] = list(serializable["channels"])
    serializable["blocks_per_stage"] = list(serializable["blocks_per_stage"])
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_curves(run_dir: Path, history: dict):
    np.savez(run_dir / "curves.npz", **history)


def save_steps(run_dir: Path, step_history: dict):
    np.savez(run_dir / "steps.npz", **step_history)


def save_checkpoint(run_dir: Path, model, optimizer, epoch: int, val_acc: float, cfg: dict):
    ch = list(cfg["channels"]) if isinstance(cfg["channels"], tuple) else cfg["channels"]
    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {**cfg, "channels": ch},
        },
        run_dir / "best.pt",
    )
