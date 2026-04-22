from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from struct import unpack

import numpy as np

import mynn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved MLP/CNN checkpoint on MNIST test set.")
    p.add_argument("--model", choices=["MLP", "CNN", "mlp", "cnn"], default="MLP", help="Model type")
    p.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Checkpoint path. Default: ./best_models/<mlp|cnn>/baseline/best_model.pickle",
    )
    p.add_argument(
        "--test_images",
        type=str,
        default="./dataset/MNIST/t10k-images-idx3-ubyte.gz",
        help="Path to MNIST test images gzip file",
    )
    p.add_argument(
        "--test_labels",
        type=str,
        default="./dataset/MNIST/t10k-labels-idx1-ubyte.gz",
        help="Path to MNIST test labels gzip file",
    )
    return p.parse_args()


def resolve_ckpt_path(model_name: str, ckpt_arg: str) -> Path:
    if ckpt_arg.strip():
        p = Path(ckpt_arg).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)

    model_key = model_name.lower()
    return Path.cwd() / "best_models" / model_key / "baseline" / "best_model.pickle"


def load_mnist_test(test_images_path: Path, test_labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(test_images_path, "rb") as f:
        _, num, _, _ = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, "rb") as f:
        _, _ = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = (test_imgs / test_imgs.max()).astype(np.float64)
    return test_imgs, test_labs


def build_model(model_name: str):
    if model_name.upper() == "MLP":
        return nn.models.Model_MLP()
    if model_name.upper() == "CNN":
        return nn.models.Model_CNN()
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    args = parse_args()

    model_name = args.model.upper()
    ckpt_path = resolve_ckpt_path(model_name, args.ckpt)
    test_images_path = Path(args.test_images).expanduser()
    test_labels_path = Path(args.test_labels).expanduser()

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Please place checkpoints under ./best_models or pass --ckpt explicitly."
        )
    if not test_images_path.is_file() or not test_labels_path.is_file():
        raise FileNotFoundError(
            f"MNIST test files not found.\n"
            f"images={test_images_path}\nlabels={test_labels_path}"
        )

    model = build_model(model_name)
    model.load_model(str(ckpt_path))
    test_imgs, test_labs = load_mnist_test(test_images_path, test_labels_path)

    logits = model(test_imgs)
    acc = nn.metric.accuracy(logits, test_labs)
    print(f"model={model_name} ckpt={ckpt_path}")
    print(f"test_accuracy={acc:.6f}")


if __name__ == "__main__":
    main()