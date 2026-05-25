"""Deprecated: use `python main.py refine` instead."""
import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from main import cmd_refine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run refine plan (200-epoch, no early stop). Prefer: python main.py refine"
    )
    parser.add_argument(
        "--plan",
        choices=["baseline", "optim", "width", "loss", "act", "all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cmd_refine(args)


if __name__ == "__main__":
    main()
