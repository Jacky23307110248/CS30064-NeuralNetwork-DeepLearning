"""Training with per-step logging; grad_probe (landscape) or grad_sweep (grad)."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_split import get_dataloaders
from evaluate import run_test_eval
from models.vgg import build_model, get_number_of_parameters
from paths import get_device, make_run_dir, print_device_info
from train_utils import (
    build_optimizer,
    build_scheduler,
    current_lr,
    evaluate,
    global_grad_norm,
    grad_probe,
    grad_sweep,
    parameter_update_norm,
    save_checkpoint,
    save_config,
    save_curves,
    save_grad_probe,
    save_grad_sweep,
    save_split,
    save_steps,
    set_seed,
    snapshot_parameters,
)


def _new_step_buffers():
    return {
        "global_step": [],
        "epoch": [],
        "batch_idx": [],
        "train_loss": [],
        "lr": [],
        "grad_norm": [],
        "update_norm": [],
    }


def _new_probe_buffers():
    return {
        "global_step": [],
        "grad_norm_w": [],
        "grad_norm_perturbed": [],
        "grad_diff": [],
    }


def _new_grad_sweep_buffers():
    return {
        "global_step": [],
        "grad_norm_w": [],
        "loss_w": [],
        "arc_length": [],
        "delta_norm": [],
        "grad_diff": [],
        "loss_perturbed": [],
        "grad_diff_min": [],
        "grad_diff_max": [],
        "loss_perturbed_min": [],
        "loss_perturbed_max": [],
    }


def _append_grad_sweep(sweep: dict, step: int, row: dict):
    sweep["global_step"].append(step)
    sweep["grad_norm_w"].append(row["grad_norm_w"])
    sweep["loss_w"].append(row["loss_w"])
    sweep["arc_length"].append(row["arc_length"])
    sweep["delta_norm"].append(row["delta_norm"])
    sweep["grad_diff"].append(row["grad_diff"])
    sweep["loss_perturbed"].append(row["loss_perturbed"])
    sweep["grad_diff_min"].append(row["grad_diff_min"])
    sweep["grad_diff_max"].append(row["grad_diff_max"])
    sweep["loss_perturbed_min"].append(row["loss_perturbed_min"])
    sweep["loss_perturbed_max"].append(row["loss_perturbed_max"])


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    cfg,
    epoch,
    global_step_start,
    steps,
    probe,
    sweep,
):
    model.train()
    epoch_train_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    eps = cfg["grad_probe_eps"]
    do_probe = cfg.get("grad_probe", False)
    do_sweep = cfg.get("grad_sweep", False)
    alphas = cfg["grad_sweep_alphas"]
    cap_mult = cfg["grad_sweep_cap_mult"]
    sweep_eta = cfg["lr"]

    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False, unit="batch")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        g_norm = global_grad_norm(model)
        step = global_step_start + batch_idx

        if do_sweep:
            row = grad_sweep(model, x, y, criterion, alphas, sweep_eta, cap_mult)
            _append_grad_sweep(sweep, step, row)
            loss_val = row["loss_w"]
        elif do_probe:
            gn_w, gn_p, g_diff = grad_probe(model, x, y, criterion, device, eps)
            probe["global_step"].append(step)
            probe["grad_norm_w"].append(gn_w)
            probe["grad_norm_perturbed"].append(gn_p)
            probe["grad_diff"].append(g_diff)
            loss_val = loss.item()
        else:
            loss_val = loss.item()

        snap = snapshot_parameters(model)
        optimizer.step()
        upd_norm = parameter_update_norm(model, snap)

        lr = current_lr(optimizer)

        steps["global_step"].append(step)
        steps["epoch"].append(epoch)
        steps["batch_idx"].append(batch_idx)
        steps["train_loss"].append(loss_val)
        steps["lr"].append(lr)
        steps["grad_norm"].append(g_norm)
        steps["update_norm"].append(upd_norm)

        epoch_train_loss += loss_val * x.size(0)
        epoch_correct += (logits.detach().argmax(1) == y).sum().item()
        epoch_total += x.size(0)
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    n_batches = len(loader)
    return {
        "loss": epoch_train_loss / epoch_total,
        "acc": epoch_correct / epoch_total,
    }, global_step_start + n_batches


def run_training(cfg: dict, run_dir: Path = None) -> Path:
    set_seed(cfg["seed"])
    device = get_device()
    print_device_info(device)

    if run_dir is None:
        run_dir = make_run_dir(cfg)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir)

    train_loader, val_loader, test_loader, split_info = get_dataloaders(cfg)
    save_split(split_info, run_dir)

    model = build_model(cfg["model_name"], cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    n_params = get_number_of_parameters(model)
    print(f"[{cfg['run_id']}] params={n_params:,} -> {run_dir}")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }
    steps = _new_step_buffers()
    probe = _new_probe_buffers() if cfg.get("grad_probe") else None
    sweep = _new_grad_sweep_buffers() if cfg.get("grad_sweep") else None

    log_path = run_dir / "train.log"
    best_val_acc = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            epoch,
            global_step,
            steps,
            probe,
            sweep,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        lr_ep = current_lr(optimizer)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(lr_ep)

        line = (
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['acc']:.4f} | val loss {val_metrics['loss']:.4f} "
            f"acc {val_metrics['acc']:.4f} | lr {lr_ep:.6e}\n"
        )
        print(f"  {line.strip()}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            save_checkpoint(run_dir, model, optimizer, epoch, best_val_acc, cfg)

    save_curves(run_dir, history)
    save_steps(run_dir, steps)
    if probe is not None:
        save_grad_probe(run_dir, probe)
    if sweep is not None:
        save_grad_sweep(run_dir, sweep, cfg)

    results = {
        "run_id": cfg["run_id"],
        "model_name": cfg["model_name"],
        "exp_type": cfg["exp_type"],
        "hyper_tag": cfg["hyper_tag"],
        "run_dir": str(run_dir),
        "num_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "max_epochs": cfg["max_epochs"],
        "total_steps": global_step,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    run_test_eval(run_dir, test_loader=test_loader, device=device)
    return run_dir
