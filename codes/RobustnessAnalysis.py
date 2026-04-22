from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from struct import unpack

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


def load_mlp_model(model_path: Path) -> nn.models.Model_MLP:
    model = nn.models.Model_MLP()
    model.load_model(str(model_path))
    return model


def load_cnn_model(model_path: Path) -> nn.models.Model_CNN:
    model = nn.models.Model_CNN()
    model.load_model(str(model_path))
    return model


def accuracy_for_model(model, images: np.ndarray, labels: np.ndarray) -> float:
    logits = model(images)
    return float(nn.metric.accuracy(logits, labels))


def make_noisy_images(clean_images: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma == 0.0:
        return clean_images.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=clean_images.shape)
    return np.clip(clean_images + noise, 0.0, 1.0)


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    default_test_img = code_dir / "dataset" / "MNIST" / "t10k-images-idx3-ubyte.gz"
    default_test_lab = code_dir / "dataset" / "MNIST" / "t10k-labels-idx1-ubyte.gz"
    default_cnn = code_dir / "best_models" / "cnn" / "baseline" / "best_model.pickle"
    default_mlp = code_dir / "best_models" / "mlp" / "baseline" / "best_model.pickle"
    default_out = code_dir / "RobustnessAnalysis.json"

    parser = argparse.ArgumentParser(
        description="Robustness analysis on MNIST test set with Gaussian noise."
    )
    parser.add_argument("--test-images", type=Path, default=default_test_img)
    parser.add_argument("--test-labels", type=Path, default=default_test_lab)
    parser.add_argument("--cnn-model", type=Path, default=default_cnn)
    parser.add_argument("--mlp-model", type=Path, default=default_mlp)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[309, 310, 311, 312, 313, 314, 315, 316, 317, 318],
        help="Run analysis with multiple fixed seeds for reproducibility.",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1],
        help="Gaussian noise std list.",
    )
    parser.add_argument("--output", type=Path, default=default_out)
    args = parser.parse_args()

    test_imgs, test_labs = load_mnist_test(args.test_images, args.test_labels)
    mlp_model = load_mlp_model(args.mlp_model)
    cnn_model = load_cnn_model(args.cnn_model)

    sigma_list = [float(s) for s in args.sigmas]
    seed_list = [int(s) for s in args.seeds]
    all_runs: list[dict] = []

    for seed in seed_list:
        rng = np.random.default_rng(seed)
        run_records: list[dict] = []
        clean_mlp_acc = None
        clean_cnn_acc = None

        for sigma in sigma_list:
            noisy_imgs = make_noisy_images(test_imgs, sigma, rng)
            mlp_acc = accuracy_for_model(mlp_model, noisy_imgs, test_labs)
            cnn_acc = accuracy_for_model(cnn_model, noisy_imgs, test_labs)

            if sigma == 0.0:
                clean_mlp_acc = mlp_acc
                clean_cnn_acc = cnn_acc

            run_records.append(
                {
                    "sigma": sigma,
                    "mlp_acc": mlp_acc,
                    "cnn_acc": cnn_acc,
                    "mlp_drop_from_clean": None if clean_mlp_acc is None else float(clean_mlp_acc - mlp_acc),
                    "cnn_drop_from_clean": None if clean_cnn_acc is None else float(clean_cnn_acc - cnn_acc),
                }
            )

        all_runs.append({"seed": seed, "accuracy_table": run_records})

    summary_records: list[dict] = []
    for sigma in sigma_list:
        mlp_vals = []
        cnn_vals = []
        mlp_drop_vals = []
        cnn_drop_vals = []

        for run in all_runs:
            row = next(r for r in run["accuracy_table"] if r["sigma"] == sigma)
            mlp_vals.append(row["mlp_acc"])
            cnn_vals.append(row["cnn_acc"])
            mlp_drop_vals.append(row["mlp_drop_from_clean"])
            cnn_drop_vals.append(row["cnn_drop_from_clean"])

        summary_records.append(
            {
                "sigma": sigma,
                "mlp_acc_mean": float(np.mean(mlp_vals)),
                "mlp_acc_std": float(np.std(mlp_vals, ddof=0)),
                "cnn_acc_mean": float(np.mean(cnn_vals)),
                "cnn_acc_std": float(np.std(cnn_vals, ddof=0)),
                "mlp_drop_from_clean_mean": float(np.mean(mlp_drop_vals)),
                "cnn_drop_from_clean_mean": float(np.mean(cnn_drop_vals)),
            }
        )

    result = {
        "noise_distribution": "epsilon ~ N(0, sigma^2)",
        "sigmas": sigma_list,
        "seeds": seed_list,
        "test_set_size": int(test_imgs.shape[0]),
        "models": {
            "mlp_baseline_best": str(args.mlp_model.resolve()),
            "cnn_baseline_best": str(args.cnn_model.resolve()),
        },
        "accuracy_table_summary": summary_records,
        "accuracy_table_runs": all_runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved robustness analysis to: {args.output}")


if __name__ == "__main__":
    main()

