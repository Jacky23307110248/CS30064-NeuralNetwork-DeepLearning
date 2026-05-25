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
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg: dict):
    if not cfg.get("use_cosine_lr", False):
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["max_epochs"],
    )


def current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


@torch.no_grad()
def global_grad_norm(model: nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().pow(2).sum().item()
    return sq ** 0.5


def grad_sweep(model, x, y, criterion, alphas, eta: float, cap_mult: float):
    """
    Multi-distance gradient probe along +grad (scheme A).

    For each alpha: arc = alpha * eta, ||delta|| = min(arc, cap_mult * ||g||),
    w' = w + delta * (g / ||g||). Records ||g(w')-g(w)|| and L(w') per distance.

    Restores weights and leaves gradients at w for optimizer.step().
    """
    params = [p for p in model.parameters() if p.requires_grad]
    flat_g = torch.cat([p.grad.detach().reshape(-1) for p in params])
    g_norm = flat_g.norm().item()

    k = len(alphas)
    zeros = [0.0] * k
    if g_norm < 1e-12:
        with torch.no_grad():
            loss_w = float(criterion(model(x), y).item())
        return {
            "grad_norm_w": g_norm,
            "loss_w": loss_w,
            "arc_length": zeros,
            "delta_norm": zeros,
            "grad_diff": zeros,
            "loss_perturbed": zeros,
            "grad_diff_min": 0.0,
            "grad_diff_max": 0.0,
            "loss_perturbed_min": 0.0,
            "loss_perturbed_max": 0.0,
        }

    direction = flat_g / g_norm
    arc_lengths = []
    delta_norms = []
    grad_diffs = []
    loss_perturbed = []

    for alpha in alphas:
        arc = float(alpha) * float(eta)
        step_len = min(arc, cap_mult * g_norm)
        arc_lengths.append(arc)
        delta_norms.append(step_len)

        offset = 0
        for p in params:
            n = p.numel()
            delta = direction[offset:offset + n].view_as(p)
            p.data.add_(delta, alpha=step_len)
            offset += n

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss_p = criterion(logits, y)
        loss_p.backward()
        flat_g2 = torch.cat([p.grad.detach().reshape(-1) for p in params])
        grad_diffs.append((flat_g2 - flat_g).norm().item())
        loss_perturbed.append(loss_p.item())

        offset = 0
        for p in params:
            n = p.numel()
            delta = direction[offset:offset + n].view_as(p)
            p.data.sub_(delta, alpha=step_len)
            offset += n

    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    loss_w_scalar = loss.item()

    return {
        "grad_norm_w": g_norm,
        "loss_w": loss_w_scalar,
        "arc_length": arc_lengths,
        "delta_norm": delta_norms,
        "grad_diff": grad_diffs,
        "loss_perturbed": loss_perturbed,
        "grad_diff_min": min(grad_diffs),
        "grad_diff_max": max(grad_diffs),
        "loss_perturbed_min": min(loss_perturbed),
        "loss_perturbed_max": max(loss_perturbed),
    }


def grad_probe(model, x, y, criterion, device, eps: float):
    """
    Measure ||g(w+eps*d)|| and ||g(w+eps*d)-g(w)|| with d = g/||g||.
    Restores weights and leaves fresh gradients for optimizer.step().
    """
    params = [p for p in model.parameters() if p.requires_grad]
    flat_g = torch.cat([p.grad.detach().reshape(-1) for p in params])
    g_norm = flat_g.norm().item()
    if g_norm < 1e-12:
        return g_norm, g_norm, 0.0

    direction = flat_g / g_norm
    offset = 0
    for p in params:
        n = p.numel()
        delta = direction[offset:offset + n].view_as(p)
        p.data.add_(delta, alpha=eps)
        offset += n

    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss_p = criterion(logits, y)
    loss_p.backward()
    flat_g2 = torch.cat([p.grad.detach().reshape(-1) for p in params])
    g2_norm = flat_g2.norm().item()
    grad_diff = (flat_g2 - flat_g).norm().item()

    offset = 0
    for p in params:
        n = p.numel()
        delta = direction[offset:offset + n].view_as(p)
        p.data.sub_(delta, alpha=eps)
        offset += n

    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    return g_norm, g2_norm, grad_diff


@torch.no_grad()
def parameter_update_norm(model: nn.Module, param_snap) -> float:
    sq = 0.0
    for p, s in zip(model.parameters(), param_snap):
        sq += (p.data - s).pow(2).sum().item()
    return sq ** 0.5


@torch.no_grad()
def snapshot_parameters(model: nn.Module):
    return [p.data.clone() for p in model.parameters()]


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
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(deepcopy(cfg), f, indent=2)


def save_split(split_info: dict, run_dir: Path):
    np.savez(
        run_dir / "split.npz",
        train_indices=np.array(split_info["train_indices"], dtype=np.int64),
        val_indices=np.array(split_info["val_indices"], dtype=np.int64),
    )


def save_curves(run_dir: Path, history: dict):
    np.savez(run_dir / "curves.npz", **{k: np.array(v) for k, v in history.items()})


def save_steps(run_dir: Path, steps: dict):
    np.savez(run_dir / "steps.npz", **{k: np.array(v) for k, v in steps.items()})


def save_grad_probe(run_dir: Path, probe: dict):
    np.savez(run_dir / "grad_probe.npz", **{k: np.array(v) for k, v in probe.items()})


def save_grad_sweep(run_dir: Path, sweep: dict, cfg: dict):
    """Persist per-step distance sweep (2D arrays for K alphas)."""
    alphas = np.asarray(cfg["grad_sweep_alphas"], dtype=np.float64)
    k = len(alphas)
    t = len(sweep["global_step"])
    arc = np.asarray(sweep["arc_length"], dtype=np.float64).reshape(t, k)
    delta = np.asarray(sweep["delta_norm"], dtype=np.float64).reshape(t, k)
    grad_diff = np.asarray(sweep["grad_diff"], dtype=np.float64).reshape(t, k)
    loss_p = np.asarray(sweep["loss_perturbed"], dtype=np.float64).reshape(t, k)

    np.savez(
        run_dir / "grad_sweep.npz",
        global_step=np.asarray(sweep["global_step"], dtype=np.int64),
        grad_norm_w=np.asarray(sweep["grad_norm_w"], dtype=np.float64),
        loss_w=np.asarray(sweep["loss_w"], dtype=np.float64),
        alphas=alphas,
        grad_sweep_cap_mult=np.float64(cfg["grad_sweep_cap_mult"]),
        grad_sweep_eta=np.float64(cfg["lr"]),
        arc_length=arc,
        delta_norm=delta,
        grad_diff=grad_diff,
        loss_perturbed=loss_p,
        grad_diff_min=np.asarray(sweep["grad_diff_min"], dtype=np.float64),
        grad_diff_max=np.asarray(sweep["grad_diff_max"], dtype=np.float64),
        loss_perturbed_min=np.asarray(sweep["loss_perturbed_min"], dtype=np.float64),
        loss_perturbed_max=np.asarray(sweep["loss_perturbed_max"], dtype=np.float64),
    )


def save_checkpoint(run_dir: Path, model, optimizer, epoch: int, val_acc: float, cfg: dict):
    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": deepcopy(cfg),
        },
        run_dir / "best.pt",
    )


def append_summary_row(summary_path: Path, row: dict):
    import csv
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
