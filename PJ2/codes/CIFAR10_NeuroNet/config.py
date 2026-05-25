"""Experiment configurations for CIFAR-10 ResNet training (200-epoch, no early stop)."""
from copy import deepcopy

BASE_CHANNELS = (64, 128, 256, 512)
W96_CHANNELS = (96, 192, 384, 768)

# All Task 1 training runs: fixed 200 epochs, early stopping disabled.
FULL_BASE = {
    "seed": 2020,
    "val_ratio": 0.1,
    "batch_size": 512,
    "num_workers": 12,
    "max_epochs": 200,
    "patience": 15,
    "min_epochs": 20,
    "channels": BASE_CHANNELS,
    "blocks_per_stage": (2, 2, 2, 2),
    "activation": "relu",
    "dropout": 0.0,
    "loss": "ce",
    "focal_gamma": 2.0,
    "margin": 1.0,
    "margin_p": 2,
    "label_smoothing": 0.0,
    "mixup_alpha": 0.0,
    "cutmix_alpha": 0.0,
    "weight_decay": 5e-4,
    "optimizer": "adamw",
    "lr": 1e-3,
    "momentum": 0.9,
    "use_cosine_lr": True,
    "no_early_stop": True,
}


def config_signature(cfg: dict) -> tuple:
    """Fields that define a unique training run (for deduplication)."""
    ch = cfg.get("channels", BASE_CHANNELS)
    if isinstance(ch, list):
        ch = tuple(ch)
    blocks = cfg.get("blocks_per_stage", (2, 2, 2, 2))
    if isinstance(blocks, list):
        blocks = tuple(blocks)
    return (
        ch,
        blocks,
        cfg.get("activation"),
        float(cfg.get("dropout", 0.0)),
        cfg.get("loss"),
        float(cfg.get("focal_gamma", 2.0)) if cfg.get("loss", "ce") == "focal" else 0.0,
        float(cfg.get("margin", 1.0)) if cfg.get("loss", "ce") == "multi_margin" else 0.0,
        int(cfg.get("margin_p", 2)) if cfg.get("loss", "ce") == "multi_margin" else 0,
        float(cfg.get("label_smoothing", 0.0)),
        float(cfg.get("mixup_alpha", 0.0)),
        float(cfg.get("cutmix_alpha", 0.0)),
        float(cfg.get("weight_decay", 0.0)),
        cfg.get("optimizer"),
        float(cfg.get("lr", 0.0)),
        bool(cfg.get("no_early_stop", True)),
    )


def resolve_channels(channels=None, width_mult=None) -> tuple:
    if channels is not None:
        return tuple(int(c) for c in channels)
    if width_mult is not None:
        return tuple(max(8, int(c * width_mult)) for c in BASE_CHANNELS)
    return BASE_CHANNELS


def make_config(
    run_id: str,
    exp_type: str,
    hyper_tag: str,
    **overrides,
) -> dict:
    cfg = deepcopy(FULL_BASE)
    cfg["run_id"] = run_id
    cfg["exp_type"] = exp_type
    cfg["hyper_tag"] = hyper_tag
    cfg.update(overrides)
    if "channels" in overrides or "width_mult" in overrides:
        cfg["channels"] = resolve_channels(
            overrides.get("channels"),
            overrides.get("width_mult"),
        )
        cfg.pop("width_mult", None)
    else:
        cfg["channels"] = resolve_channels(cfg.get("channels"))
    cfg["no_early_stop"] = True
    return cfg


def get_baseline_full_config() -> dict:
    return make_config(
        "baseline_full",
        "baseline_full",
        "adamw1e-3_wd5e4_ep200",
    )


def get_width_full_configs() -> list:
    return [
        make_config("A1_full", "width_full", "w32-64-128-256_ep200", channels=(32, 64, 128, 256)),
        make_config("A2_full", "width_full", "w96-192-384-768_ep200", channels=W96_CHANNELS),
    ]


def get_loss_full_configs() -> list:
    return [
        make_config("B1_full", "loss_full", "ce_wd0_ep200", weight_decay=0.0),
        make_config("B2_full", "loss_full", "ce_wd1e3_ep200", weight_decay=1e-3),
        make_config("B3_full", "loss_full", "ls0.1_wd5e4_ep200", label_smoothing=0.1),
        make_config("B5_full", "loss_full", "mixup0.2_wd5e4_ep200", mixup_alpha=0.2),
        make_config("B6_full", "loss_full", "cutmix1.0_wd5e4_ep200", cutmix_alpha=1.0),
        make_config("B7_full", "loss_full", "ce_wd5e4_drop0.5_ep200", dropout=0.5),
        make_config("B8_full", "loss_full", "focal_g2_wd5e4_ep200", loss="focal", focal_gamma=2.0),
        make_config(
            "B9_full",
            "loss_full",
            "multimargin_m1_wd5e4_ep200",
            loss="multi_margin",
            margin=1.0,
            margin_p=2,
        ),
    ]


def get_act_full_configs() -> list:
    return [
        make_config("C1_full", "act_full", "gelu_ep200", activation="gelu"),
        make_config("C2_full", "act_full", "leaky_relu_ep200", activation="leaky_relu"),
    ]


def _optim_lr_tag(lr: float) -> str:
    if lr == 0.1:
        return "0.1"
    if lr == 0.05:
        return "0.05"
    if lr == 0.2:
        return "0.2"
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def get_optimizer_full_configs() -> list:
    """Optimizer grid (AdamW lr=1e-3 covered by baseline_full)."""
    configs = []
    optim_lrs = {
        "sgd": [0.05, 0.1, 0.2],
        "adam": [3e-4, 1e-3, 3e-3],
        "adamw": [3e-4, 3e-3],
    }
    for optim, lrs in optim_lrs.items():
        for lr in lrs:
            lr_tag = _optim_lr_tag(lr)
            configs.append(
                make_config(
                    f"D_{optim}_{lr_tag}_full",
                    "optim_full",
                    f"{optim}_lr{lr_tag}_ep200",
                    optimizer=optim,
                    lr=lr,
                )
            )
    return configs


REFINEMENT_PLAN_NAMES = ("baseline", "optim", "width", "loss", "act", "all")


def get_refinement_configs(plan: str) -> list:
    plan_fns = {
        "baseline": lambda: [get_baseline_full_config()],
        "optim": get_optimizer_full_configs,
        "width": get_width_full_configs,
        "loss": get_loss_full_configs,
        "act": get_act_full_configs,
    }
    if plan == "all":
        configs = [get_baseline_full_config()]
        for fn in (
            get_optimizer_full_configs,
            get_width_full_configs,
            get_loss_full_configs,
            get_act_full_configs,
        ):
            configs.extend(fn())
        return configs
    if plan not in plan_fns:
        raise ValueError(
            f"Unknown plan: {plan}. Choose from {', '.join(REFINEMENT_PLAN_NAMES)}"
        )
    return plan_fns[plan]()


COMBINE_RECIPE_NAMES = (
    "cutmix_w96_sgd0.1",
    "cutmix_w96_sgd0.05",
    "cutmix_w64_sgd0.1",
    "mixup_w64_sgd0.1",
    "all",
)


def _combine_base(hyper_tag: str, **overrides) -> dict:
    return make_config(
        hyper_tag,
        "combine",
        hyper_tag,
        weight_decay=5e-4,
        use_cosine_lr=True,
        **overrides,
    )


def get_combine_config(recipe: str) -> dict:
    recipes = {
        "cutmix_w96_sgd0.1": lambda: _combine_base(
            "cutmix_w96_sgd0.1",
            channels=W96_CHANNELS,
            cutmix_alpha=1.0,
            mixup_alpha=0.0,
            optimizer="sgd",
            lr=0.1,
        ),
        "cutmix_w96_sgd0.05": lambda: _combine_base(
            "cutmix_w96_sgd0.05",
            channels=W96_CHANNELS,
            cutmix_alpha=1.0,
            mixup_alpha=0.0,
            optimizer="sgd",
            lr=0.05,
        ),
        "cutmix_w64_sgd0.1": lambda: _combine_base(
            "cutmix_w64_sgd0.1",
            channels=BASE_CHANNELS,
            cutmix_alpha=1.0,
            mixup_alpha=0.0,
            optimizer="sgd",
            lr=0.1,
        ),
        "mixup_w64_sgd0.1": lambda: _combine_base(
            "mixup_w64_sgd0.1",
            channels=BASE_CHANNELS,
            cutmix_alpha=0.0,
            mixup_alpha=0.2,
            optimizer="sgd",
            lr=0.1,
        ),
    }
    if recipe not in recipes:
        raise ValueError(
            f"Unknown combine recipe: {recipe}. "
            f"Choose from {', '.join(COMBINE_RECIPE_NAMES)}"
        )
    cfg = recipes[recipe]()
    cfg["run_id"] = recipe
    return cfg


def get_combine_configs(recipe: str) -> list:
    if recipe == "all":
        return [get_combine_config(name) for name in COMBINE_RECIPE_NAMES if name != "all"]
    return [get_combine_config(recipe)]


ANALYSIS_LANDSCAPE_DEFAULT_LRS = (0.05, 0.1, 0.15, 0.2)
ANALYSIS_LANDSCAPE_DEFAULT_EPOCHS = 50


def _format_lr_tag(lr: float) -> str:
    if lr >= 0.01:
        return f"{lr:.2f}".rstrip("0").rstrip(".")
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def get_analysis_landscape_configs(
    optimizer: str = "sgd",
    selected_lrs: tuple[float, ...] | list[float] | None = None,
    use_cutmix: bool = True,
) -> list:
    optimizer = optimizer.lower()
    if optimizer not in ("sgd", "adam", "adamw"):
        raise ValueError(
            f"Unsupported analysis optimizer: {optimizer}. Choose from sgd, adam, adamw."
        )

    if selected_lrs is None:
        lrs = ANALYSIS_LANDSCAPE_DEFAULT_LRS
    else:
        lrs = tuple(float(lr) for lr in selected_lrs)
        if not lrs:
            raise ValueError("No valid analysis learning rates selected.")

    configs = []
    analysis_tag = "cutmix_w96" if use_cutmix else "nocutmix_w96"
    cutmix_alpha = 1.0 if use_cutmix else 0.0
    for lr in lrs:
        lr_tag = _format_lr_tag(lr)
        cfg = make_config(
            run_id=f"analysis_landscape_{optimizer}_lr{lr_tag}",
            exp_type="analysis_landscape_sgd",
            hyper_tag=f"{analysis_tag}_{optimizer}{lr_tag}_ep{ANALYSIS_LANDSCAPE_DEFAULT_EPOCHS}",
            channels=W96_CHANNELS,
            cutmix_alpha=cutmix_alpha,
            mixup_alpha=0.0,
            optimizer=optimizer,
            lr=lr,
            max_epochs=ANALYSIS_LANDSCAPE_DEFAULT_EPOCHS,
            no_early_stop=True,
        )
        configs.append(cfg)
    return configs
