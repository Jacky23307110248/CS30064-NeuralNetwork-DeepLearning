"""Experiment configs for Task 2 (VGG + BatchNorm)."""
from copy import deepcopy

LANDSCAPE_LRS = [1e-3, 2e-3, 1e-4, 5e-4]

# Santurkar-style distance sweep (scheme A): arc length alpha * lr along +grad direction
GRAD_SWEEP_ALPHAS = [0.5, 1.0, 2.0, 4.0]
GRAD_SWEEP_CAP_MULT = 0.4
DEFAULT_GRAD_LR = 1e-3
DEFAULT_GRAD_EPOCHS = 100

_BASE = {
    "seed": 2020,
    "val_ratio": 0.1,
    "batch_size": 512,
    "num_workers": 12,
    "optimizer": "adamw",
    "weight_decay": 5e-4,
    "lr": 1e-3,
    "use_cosine_lr": True,
    "grad_probe": False,
    "grad_probe_eps": 1e-4,
    "grad_sweep": False,
    "grad_sweep_alphas": GRAD_SWEEP_ALPHAS,
    "grad_sweep_cap_mult": GRAD_SWEEP_CAP_MULT,
    "num_classes": 10,
}


def _lr_tag(lr: float) -> str:
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def make_config(
    run_id: str,
    model_name: str,
    exp_type: str,
    hyper_tag: str,
    max_epochs: int,
    lr: float = 1e-3,
    use_cosine_lr: bool = True,
    grad_probe: bool = False,
    grad_sweep: bool = False,
) -> dict:
    cfg = deepcopy(_BASE)
    cfg.update({
        "run_id": run_id,
        "model_name": model_name,
        "exp_type": exp_type,
        "hyper_tag": hyper_tag,
        "max_epochs": max_epochs,
        "lr": lr,
        "use_cosine_lr": use_cosine_lr,
        "grad_probe": grad_probe,
        "grad_sweep": grad_sweep,
    })
    return cfg


def get_bn_compare_configs():
    cfgs = []
    for model_name in ("VGG_A", "VGG_A_BatchNorm"):
        rid = f"compare_{model_name}"
        cfgs.append(make_config(
            run_id=rid,
            model_name=model_name,
            exp_type="bn_compare",
            hyper_tag="adamw1e-3_cos_ep200",
            max_epochs=200,
            lr=1e-3,
            use_cosine_lr=True,
            grad_probe=False,
        ))
    return cfgs


def get_landscape_configs():
    cfgs = []
    for model_name in ("VGG_A", "VGG_A_BatchNorm"):
        for lr in LANDSCAPE_LRS:
            tag = f"fix_{_lr_tag(lr)}_ep100"
            rid = f"landscape_{model_name}_{_lr_tag(lr)}"
            cfgs.append(make_config(
                run_id=rid,
                model_name=model_name,
                exp_type="loss_landscape",
                hyper_tag=tag,
                max_epochs=100,
                lr=lr,
                use_cosine_lr=False,
                grad_probe=True,
            ))
    return cfgs


def get_grad_probe_configs(
    lr: float = DEFAULT_GRAD_LR,
    max_epochs: int = DEFAULT_GRAD_EPOCHS,
    model_name: str | None = None,
):
    """Two-run grad distance sweep (VGG_A +/- BN); records grad_sweep.npz only."""
    models = (model_name,) if model_name else ("VGG_A", "VGG_A_BatchNorm")
    tag = f"fix_{_lr_tag(lr)}_ep{max_epochs}"
    cfgs = []
    for name in models:
        rid = f"grad_probe_{name}_{_lr_tag(lr)}"
        cfgs.append(make_config(
            run_id=rid,
            model_name=name,
            exp_type="grad_probe",
            hyper_tag=tag,
            max_epochs=max_epochs,
            lr=lr,
            use_cosine_lr=False,
            grad_probe=False,
            grad_sweep=True,
        ))
    return cfgs


def get_all_experiment_configs():
    return get_bn_compare_configs() + get_landscape_configs()
