"""CLI entry for VGG BatchNorm experiments (Task 2)."""
import argparse
import json
from pathlib import Path

import paths  # noqa: F401 — sets up sys.path
from config import (
    DEFAULT_GRAD_EPOCHS,
    DEFAULT_GRAD_LR,
    get_all_experiment_configs,
    get_bn_compare_configs,
    get_grad_probe_configs,
    get_landscape_configs,
)
from evaluate import run_test_eval
from paths import OUTPUT_ROOT
from train import run_training
from train_utils import append_summary_row


def _record_summary(cfg: dict, run_dir: Path):
    results_path = Path(run_dir) / "results.json"
    if not results_path.exists():
        return
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    row = {
        "run_id": cfg["run_id"],
        "model_name": cfg["model_name"],
        "exp_type": cfg["exp_type"],
        "hyper_tag": cfg["hyper_tag"],
        "lr": cfg["lr"],
        "run_dir": str(run_dir),
        "best_val_acc": results.get("best_val_acc"),
        "test_loss": results.get("test_loss"),
        "test_acc": results.get("test_acc"),
        "test_error": results.get("test_error"),
    }
    append_summary_row(OUTPUT_ROOT / "experiment_summary.csv", row)


def _run_configs(configs, label: str):
    print(f"\n=== {label}: {len(configs)} run(s) ===\n")
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg['run_id']}")
        run_dir = run_training(cfg)
        _record_summary(cfg, run_dir)
    print(f"\n{label} finished. Summary: {OUTPUT_ROOT / 'experiment_summary.csv'}")


def cmd_compare(_args):
    _run_configs(get_bn_compare_configs(), "bn_compare")


def cmd_landscape(_args):
    _run_configs(get_landscape_configs(), "loss_landscape")


def cmd_grad(args):
    configs = get_grad_probe_configs(lr=args.lr, max_epochs=args.epochs)
    if args.run_id:
        configs = [c for c in configs if c["run_id"] == args.run_id]
        if not configs:
            raise SystemExit(f"Unknown run_id for grad: {args.run_id}")
    elif args.model:
        configs = [c for c in configs if c["model_name"] == args.model]
        if not configs:
            raise SystemExit(f"Unknown model for grad: {args.model}")
    _run_configs(configs, "grad_probe (distance sweep)")


def cmd_all(_args):
    _run_configs(get_all_experiment_configs(), "all experiments")


def cmd_single(args):
    configs = get_all_experiment_configs() + get_grad_probe_configs()
    cfg = next((c for c in configs if c["run_id"] == args.run_id), None)
    if cfg is None:
        raise SystemExit(f"Unknown run_id: {args.run_id}")
    run_dir = run_training(cfg)
    _record_summary(cfg, run_dir)
    print(f"Done. Outputs: {run_dir}")


def cmd_eval(args):
    run_test_eval(Path(args.run_dir))
    print(f"Updated {args.run_dir}/results.json")


def _add_grad_parser(sub):
    p = sub.add_parser(
        "grad",
        help="Run grad_probe experiment with per-step distance sweep (grad_sweep.npz)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_GRAD_LR,
        help=f"Fixed training learning rate (default {DEFAULT_GRAD_LR})",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_GRAD_EPOCHS,
        help=f"Training epochs (default {DEFAULT_GRAD_EPOCHS})",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run a single config by run_id (e.g. grad_probe_VGG_A_1e-3)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        choices=("VGG_A", "VGG_A_BatchNorm"),
        help="Run a single model (alternative to --run-id)",
    )


def main():
    parser = argparse.ArgumentParser(description="VGG BatchNorm on CIFAR-10 (Task 2)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("compare", help="Run 2 bn_compare experiments (200 ep, cosine)")
    sub.add_parser("landscape", help="Run 8 loss_landscape experiments (100 ep, fixed lr)")
    sub.add_parser("all", help="Run compare + landscape (10 experiments)")
    _add_grad_parser(sub)

    p_single = sub.add_parser("run", help="Run one experiment by run_id")
    p_single.add_argument("--run-id", required=True, type=str)

    p_ev = sub.add_parser("eval", help="Re-run test eval for an existing run directory")
    p_ev.add_argument("--run-dir", required=True, type=str)

    args = parser.parse_args()
    if args.command == "compare":
        cmd_compare(args)
    elif args.command == "landscape":
        cmd_landscape(args)
    elif args.command == "all":
        cmd_all(args)
    elif args.command == "grad":
        cmd_grad(args)
    elif args.command == "run":
        cmd_single(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
