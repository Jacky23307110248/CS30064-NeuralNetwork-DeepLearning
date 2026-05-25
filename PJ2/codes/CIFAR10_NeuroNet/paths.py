"""Project paths and runtime environment."""
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "CIFAR10_NeuroNet"

for path in (PROJECT_ROOT, PACKAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def get_device():
    """Use cuda:0. ROCm PyTorch on AMD also exposes the CUDA-compatible API."""
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


def make_run_dir(exp_type: str, hyper_tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUT_ROOT / f"{exp_type}-{hyper_tag}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
