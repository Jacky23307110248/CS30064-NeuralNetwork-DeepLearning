"""CLI entry for CIFAR-10 NeuroNet experiments."""
import argparse
from pathlib import Path

import paths  # noqa: F401 — sets up sys.path
from config import (
    COMBINE_RECIPE_NAMES,
    get_analysis_landscape_configs,
    get_combine_configs,
    get_refinement_configs,
)
from evaluate import run_test_eval
from train import run_training


def cmd_eval(args):
    run_test_eval(Path(args.run_dir))
    print(f"Updated {args.run_dir}/results.json")


def cmd_refine(args):
    """Ablation grids: 200 epochs, no early stop -> {group}_full output dirs."""
    configs = get_refinement_configs(args.plan)
    print(f"\n=== Refine plan '{args.plan}': {len(configs)} run(s) ===\n")
    for cfg in configs:
        print(f"\n=== Running {cfg['run_id']} ({cfg['exp_type']}) ===")
        run_training(cfg, force=args.force)
    print(f"\nRefine '{args.plan}' finished.")


def cmd_combine(args):
    """Multi-factor combo runs (CutMix/MixUp + SGD + width)."""
    configs = get_combine_configs(args.recipe)
    print(f"\n=== Combine recipe '{args.recipe}': {len(configs)} run(s) ===\n")
    for cfg in configs:
        print(f"\n=== Running {cfg['run_id']} ({cfg['exp_type']}/{cfg['hyper_tag']}) ===")
        run_training(cfg, force=args.force)
    print(f"\nCombine '{args.recipe}' finished.")


def cmd_analysis(args):
    if args.task != "landscape_sgd":
        raise ValueError(f"Unknown analysis task: {args.task}")
    configs = get_analysis_landscape_configs(
        optimizer=args.optimizer,
        selected_lrs=args.lrs,
        use_cutmix=not args.no_cutmix,
    )
    print(f"\n=== Analysis task '{args.task}': {len(configs)} run(s) ===\n")
    for cfg in configs:
        print(f"\n=== Running {cfg['run_id']} ({cfg['exp_type']}/{cfg['hyper_tag']}) ===")
        run_training(cfg, force=args.force)
    print(f"\nAnalysis '{args.task}' finished.")


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 NeuroNet (Task 1): refine (200-ep ablations) or combine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ev = sub.add_parser("eval", help="Re-run test eval for an existing run directory")
    p_ev.add_argument("--run-dir", required=True, type=str)

    p_ref = sub.add_parser(
        "refine",
        help="200-epoch ablation runs (no early stop) -> {group}_full output dirs",
    )
    p_ref.add_argument(
        "--plan",
        choices=["baseline", "optim", "width", "loss", "act", "all"],
        default="all",
        help=(
            "baseline=baseline_full; optim/width/loss/act=*_full grids; "
            "all=21 runs"
        ),
    )
    p_ref.add_argument(
        "--force",
        action="store_true",
        help="Delete matching output dirs and retrain",
    )

    combine_choices = list(COMBINE_RECIPE_NAMES)
    p_com = sub.add_parser(
        "combine",
        help="Combo runs (aug + SGD + width) -> outputs/combine-{hyper_tag}-*",
    )
    p_com.add_argument(
        "--recipe",
        choices=combine_choices,
        default="all",
        help=(
            "cutmix_w96_sgd0.1 | cutmix_w96_sgd0.05 | cutmix_w64_sgd0.1 | "
            "mixup_w64_sgd0.1 | all (4 runs)"
        ),
    )
    p_com.add_argument(
        "--force",
        action="store_true",
        help="Delete matching output dirs and retrain even if complete",
    )

    p_analysis = sub.add_parser(
        "analysis",
        help="Task1 intrinsic analysis training runs",
    )
    p_analysis.add_argument(
        "--task",
        choices=["landscape_sgd"],
        default="landscape_sgd",
        help="analysis training task",
    )
    p_analysis.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw"],
        default="sgd",
        help="optimizer for landscape_sgd (default: sgd)",
    )
    p_analysis.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=None,
        help=(
            "optional learning rates for landscape_sgd; "
            "default: 0.05 0.1 0.15 0.2"
        ),
    )
    p_analysis.add_argument(
        "--no-cutmix",
        action="store_true",
        help="Disable CutMix in analysis runs (default: enabled)",
    )
    p_analysis.add_argument(
        "--force",
        action="store_true",
        help="Delete matching output dirs and retrain even if complete",
    )

    args = parser.parse_args()
    if args.command == "eval":
        cmd_eval(args)
    elif args.command == "refine":
        cmd_refine(args)
    elif args.command == "combine":
        cmd_combine(args)
    elif args.command == "analysis":
        cmd_analysis(args)


if __name__ == "__main__":
    main()
