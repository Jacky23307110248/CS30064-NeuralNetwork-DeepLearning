"""Single-run training with early stopping."""
import json
from pathlib import Path

import numpy as np
import torch

from data_split import get_dataloaders
from evaluate import run_test_eval
from losses import cutmix_data, get_criterion, mixup_data, mixed_criterion
from models.model import build_model, count_parameters
from paths import get_device, make_run_dir, print_device_info
from run_resolve import list_matching_run_dirs, remove_run_dir, resolve_run_dir
from train_utils import (
    build_optimizer,
    build_scheduler,
    evaluate,
    save_checkpoint,
    save_config,
    save_curves,
    save_steps,
    set_seed,
)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    cfg,
    epoch_idx: int,
    global_step: int,
    record_steps: bool = False,
):
    model.train()
    use_mixup = float(cfg.get("mixup_alpha", 0)) > 0
    use_cutmix = float(cfg.get("cutmix_alpha", 0)) > 0
    total_loss = 0.0
    correct = 0
    total = 0
    step_records = None
    if record_steps:
        step_records = {"global_step": [], "epoch": [], "batch_idx": [], "train_loss": [], "lr": []}

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_cutmix:
            x, y_a, y_b, lam = cutmix_data(x, y, cfg["cutmix_alpha"], device)
            logits = model(x)
            loss = mixed_criterion(criterion, logits, y_a, y_b, lam)
            pred = logits.argmax(1)
            correct += (lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()).sum().item()
        elif use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, cfg["mixup_alpha"], device)
            logits = model(x)
            loss = mixed_criterion(criterion, logits, y_a, y_b, lam)
            pred = logits.argmax(1)
            correct += (lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()).sum().item()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            correct += (logits.argmax(1) == y).sum().item()

        loss.backward()
        optimizer.step()
        global_step += 1
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        if record_steps:
            step_records["global_step"].append(global_step)
            step_records["epoch"].append(epoch_idx)
            step_records["batch_idx"].append(batch_idx)
            step_records["train_loss"].append(loss.item())
            step_records["lr"].append(float(optimizer.param_groups[0]["lr"]))

    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "global_step": global_step,
        "step_records": step_records,
    }


def run_training(cfg: dict, run_dir: Path = None, force: bool = False) -> Path:
    record_steps = cfg.get("exp_type") == "analysis_landscape_sgd"
    set_seed(cfg["seed"])
    device = get_device()
    print_device_info(device)

    if run_dir is None:
        if force:
            for old in list_matching_run_dirs(cfg):
                remove_run_dir(old)
            run_dir = make_run_dir(cfg["exp_type"], cfg["hyper_tag"])
            skipped = False
        else:
            run_dir, skipped = resolve_run_dir(cfg)
    else:
        run_dir = Path(run_dir)
        skipped = False

    run_dir = Path(run_dir)

    if skipped:
        results_path = run_dir / "results.json"
        needs_test = True
        if results_path.exists():
            with open(results_path, encoding="utf-8") as f:
                results = json.load(f)
            needs_test = results.get("test_acc") is None
        if needs_test:
            _, _, test_loader = get_dataloaders(cfg)
            run_test_eval(run_dir, test_loader=test_loader, device=device)
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = get_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    n_params = count_parameters(model)
    print(f"[{cfg['run_id']}] params={n_params:,} device={device} -> {run_dir}")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    step_history = None
    if record_steps:
        step_history = {
            "global_step": [],
            "epoch": [],
            "batch_idx": [],
            "train_loss": [],
            "lr": [],
        }

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            epoch_idx=epoch,
            global_step=global_step,
            record_steps=record_steps,
        )
        global_step = train_metrics["global_step"]
        val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        if record_steps:
            for key in step_history:
                step_history[key].extend(train_metrics["step_records"][key])

        print(
            f"  epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} "
            f"| val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(run_dir, model, optimizer, epoch, best_val_acc, cfg)
        else:
            epochs_no_improve += 1

        use_early_stop = not cfg.get("no_early_stop", False)
        if use_early_stop and epoch >= cfg["min_epochs"] and epochs_no_improve >= cfg["patience"]:
            print(f"  early stop at epoch {epoch} (patience={cfg['patience']})")
            break

    save_curves(run_dir, {k: np.array(v) for k, v in history.items()})
    if record_steps:
        save_steps(run_dir, {k: np.array(v) for k, v in step_history.items()})

    stopped_epoch = history["epoch"][-1] if history["epoch"] else 0
    interim = {
        "run_id": cfg["run_id"],
        "exp_type": cfg["exp_type"],
        "hyper_tag": cfg["hyper_tag"],
        "run_dir": str(run_dir),
        "num_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "stopped_epoch": stopped_epoch,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(interim, f, indent=2)

    run_test_eval(run_dir, test_loader=test_loader, device=device)
    return run_dir
