"""Loss functions and mixed-sample criteria."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss on logits (Lin et al., ICCV 2017)."""

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def get_criterion(cfg: dict) -> nn.Module:
    loss_name = cfg.get("loss", "ce").lower()
    if loss_name == "ce":
        label_smoothing = float(cfg.get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_name == "focal":
        gamma = float(cfg.get("focal_gamma", 2.0))
        return FocalLoss(gamma=gamma)
    if loss_name in ("multi_margin", "multimargin", "hinge"):
        margin = float(cfg.get("margin", 1.0))
        p = int(cfg.get("margin_p", 2))
        return nn.MultiMarginLoss(margin=margin, p=p)
    raise ValueError(f"Unknown loss: {loss_name}")


def mixup_data(x, y, alpha: float, device):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha: float, device):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    _, _, h, w = x.shape
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = torch.randint(0, w, (1,), device=device).item()
    cy = torch.randint(0, h, (1,), device=device).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    x_clone = x.clone()
    x_clone[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x_clone, y, y[index], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
