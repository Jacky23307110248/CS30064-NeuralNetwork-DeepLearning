"""Project paths and runtime environment."""
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "VGG_BatchNorm"

for path in (PROJECT_ROOT, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def print_device_info(device):
    import torch
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        print(f"device={device} ({name})")
    else:
        print(f"device={device}")


def make_run_dir(cfg: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = (
        f"{cfg['model_name']}_{cfg['exp_type']}_{cfg['hyper_tag']}"
        f"_seed{cfg['seed']}_{timestamp}"
    )
    run_dir = OUTPUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
