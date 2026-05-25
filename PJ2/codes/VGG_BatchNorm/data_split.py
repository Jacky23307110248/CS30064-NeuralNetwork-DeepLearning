"""Train/val split from official 50k; official 10k test kept separate."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from paths import DATA_ROOT


def _normalize_transform():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def get_dataloaders(cfg: dict):
    batch_size = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 12)
    seed = cfg["seed"]
    val_ratio = cfg["val_ratio"]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize_transform(),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        _normalize_transform(),
    ])

    full_train = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=train_transform,
    )
    val_count = int(len(full_train) * val_ratio)
    train_count = len(full_train) - val_count
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train,
        [train_count, val_count],
        generator=generator,
    )

    eval_train_base = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=eval_transform,
    )
    val_dataset = Subset(eval_train_base, val_subset.indices)

    test_dataset = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    split_info = {
        "train_indices": list(train_subset.indices),
        "val_indices": list(val_subset.indices),
        "train_count": train_count,
        "val_count": val_count,
    }
    return train_loader, val_loader, test_loader, split_info
