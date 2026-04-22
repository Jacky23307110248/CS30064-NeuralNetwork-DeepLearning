from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


def load_mnist_test(test_images_path: Path, test_labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(test_images_path, "rb") as f:
        _, num, _, _ = unpack(">4I", f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, "rb") as f:
        _, _ = unpack(">2I", f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = (test_imgs / test_imgs.max()).astype(np.float64)
    return test_imgs, test_labs


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        cm[int(y_true), int(y_pred)] += 1
    return cm


def select_top_misclassified(
    images: np.ndarray, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, top_k: int = 10
) -> list[tuple[np.ndarray, int, int, float]]:
    wrong_mask = preds != labels
    wrong_idx = np.where(wrong_mask)[0]
    if wrong_idx.size == 0:
        return []

    # Highest-confidence wrong predictions first.
    wrong_conf = probs[wrong_idx, preds[wrong_idx]]
    order = np.argsort(-wrong_conf)
    pick = wrong_idx[order[:top_k]]

    samples = []
    for i in pick:
        samples.append((images[i].reshape(28, 28), int(labels[i]), int(preds[i]), float(probs[i, preds[i]])))
    return samples


def plot_confusion_matrix_figure(cm: np.ndarray, model_name: str, acc: float, out_path: Path) -> None:
    fig, ax_cm = plt.subplots(figsize=(7.5, 6.5))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title(f"{model_name} Confusion Matrix (test acc={acc:.4f})", fontsize=13)
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_xticks(np.arange(10))
    ax_cm.set_yticks(np.arange(10))
    fig.colorbar(im, ax=ax_cm, fraction=0.025, pad=0.02)

    # Optional small text in each cell for readability
    vmax = max(int(cm.max()), 1)
    threshold = vmax * 0.6
    for r in range(10):
        for c in range(10):
            val = int(cm[r, c])
            color = "white" if val > threshold else "black"
            ax_cm.text(c, r, str(val), ha="center", va="center", fontsize=7, color=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top10_figure(
    samples: list[tuple[np.ndarray, int, int, float]],
    model_name: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 5, hspace=0.45, wspace=0.25)
    for i in range(10):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        ax.axis("off")
        if i < len(samples):
            img, y_true, y_pred, conf = samples[i]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"T:{y_true} P:{y_pred}\nconf={conf:.3f}", fontsize=9)
        else:
            ax.set_title("N/A", fontsize=9)

    fig.suptitle(f"{model_name} Top-10 Misclassified Samples", fontsize=14, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_for_model(
    model_name: str,
    model_path: Path,
    out_dir: Path,
    test_imgs: np.ndarray,
    test_labs: np.ndarray,
) -> None:
    if model_name.upper() == "MLP":
        model = nn.models.Model_MLP()
    elif model_name.upper() == "CNN":
        model = nn.models.Model_CNN()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.load_model(str(model_path))
    logits = model(test_imgs)
    probs = softmax(logits)
    preds = np.argmax(logits, axis=1)
    acc = float(nn.metric.accuracy(logits, test_labs))

    cm = confusion_matrix(test_labs, preds, num_classes=10)
    samples = select_top_misclassified(test_imgs, test_labs, preds, probs, top_k=10)

    cm_path = out_dir / "confusion_matrix.png"
    top10_path = out_dir / "top10_misclassified.png"
    plot_confusion_matrix_figure(cm, model_name=model_name.upper(), acc=acc, out_path=cm_path)
    plot_top10_figure(samples, model_name=model_name.upper(), out_path=top10_path)
    print(f"[{model_name.upper()}] saved: {cm_path}")
    print(f"[{model_name.upper()}] saved: {top10_path}")


def main() -> None:
    code_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate confusion matrix + top-10 misclassified samples for MLP/CNN baseline models."
    )
    parser.add_argument(
        "--test-images",
        type=Path,
        default=code_dir / "dataset" / "MNIST" / "t10k-images-idx3-ubyte.gz",
    )
    parser.add_argument(
        "--test-labels",
        type=Path,
        default=code_dir / "dataset" / "MNIST" / "t10k-labels-idx1-ubyte.gz",
    )
    parser.add_argument(
        "--mlp-model",
        type=Path,
        default=code_dir / "best_models" / "mlp" / "baseline" / "best_model.pickle",
    )
    parser.add_argument(
        "--cnn-model",
        type=Path,
        default=code_dir / "best_models" / "cnn" / "baseline" / "best_model.pickle",
    )
    parser.add_argument(
        "--mlp-out-dir",
        type=Path,
        default=code_dir / "ErrorAnalysisDoc" / "mlp",
    )
    parser.add_argument(
        "--cnn-out-dir",
        type=Path,
        default=code_dir / "ErrorAnalysisDoc" / "cnn",
    )
    args = parser.parse_args()

    test_imgs, test_labs = load_mnist_test(args.test_images, args.test_labels)

    run_for_model("MLP", args.mlp_model, args.mlp_out_dir, test_imgs, test_labs)
    run_for_model("CNN", args.cnn_model, args.cnn_out_dir, test_imgs, test_labs)


if __name__ == "__main__":
    main()

