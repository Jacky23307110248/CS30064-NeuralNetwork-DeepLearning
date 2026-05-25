"""Resolve output run directories: skip complete runs or clean and recreate."""
import json
import shutil
from pathlib import Path
from typing import List, Tuple

from config import config_signature
from paths import OUTPUT_ROOT, make_run_dir

BASE_REQUIRED_ARTIFACTS = ("config.json", "curves.npz", "best.pt", "results.json")
ANALYSIS_REQUIRED_EXTRA = ("steps.npz",)


def run_dir_prefix(cfg: dict) -> str:
    return f"{cfg['exp_type']}-{cfg['hyper_tag']}"


def list_matching_run_dirs(cfg: dict) -> List[Path]:
    needle = run_dir_prefix(cfg) + "-"
    if not OUTPUT_ROOT.exists():
        return []
    dirs = [
        p for p in OUTPUT_ROOT.iterdir()
        if p.is_dir() and p.name.startswith(needle)
    ]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def required_artifacts_for_cfg(cfg: dict) -> tuple[str, ...]:
    if cfg.get("exp_type") == "analysis_landscape_sgd":
        return BASE_REQUIRED_ARTIFACTS + ANALYSIS_REQUIRED_EXTRA
    return BASE_REQUIRED_ARTIFACTS


def artifacts_complete(run_dir: Path, cfg: dict) -> bool:
    required = required_artifacts_for_cfg(cfg)
    return all((run_dir / name).is_file() for name in required)


def _normalize_cfg_dict(cfg: dict) -> dict:
    out = dict(cfg)
    if "channels" in out:
        out["channels"] = tuple(out["channels"])
    if "blocks_per_stage" in out:
        out["blocks_per_stage"] = tuple(out["blocks_per_stage"])
    return out


def config_matches_saved(run_dir: Path, cfg: dict) -> bool:
    try:
        with open(run_dir / "config.json", encoding="utf-8") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return config_signature(_normalize_cfg_dict(saved)) == config_signature(_normalize_cfg_dict(cfg))


def remove_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"[clean] removed {run_dir}")


def resolve_run_dir(cfg: dict) -> Tuple[Path, bool]:
    """
    Scan outputs for directories starting with {exp_type}-{hyper_tag}-.

    - All required files present and config matches -> skip (return dir, True).
    - Otherwise remove matching dirs and create a new timestamped run directory.
    """
    matches = list_matching_run_dirs(cfg)

    for run_dir in matches:
        required = required_artifacts_for_cfg(cfg)
        if artifacts_complete(run_dir, cfg) and config_matches_saved(run_dir, cfg):
            print(
                f"[skip] {cfg['run_id']}: complete run "
                f"({', '.join(required)}) -> {run_dir}"
            )
            return run_dir, True

    for run_dir in matches:
        required = required_artifacts_for_cfg(cfg)
        if not artifacts_complete(run_dir, cfg):
            missing = [n for n in required if not (run_dir / n).is_file()]
            print(f"[clean] {cfg['run_id']}: missing {missing}")
        else:
            print(f"[clean] {cfg['run_id']}: config mismatch")
        remove_run_dir(run_dir)

    new_dir = make_run_dir(cfg["exp_type"], cfg["hyper_tag"])
    print(f"[new] {cfg['run_id']}: -> {new_dir}")
    return new_dir, False
