"""Segmentation losses for class-imbalanced crack data.

All losses accept the model's raw logits ``(B, C, H, W)`` and integer targets
``(B, H, W)`` with values in ``{0, ..., C-1}``. The crack class is assumed to
be class 1 (positive); class 0 is background.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss on the crack class.

    Dice = 2 * sum(p * y) / (sum(p) + sum(y))
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # crack-class probability per pixel
        prob = F.softmax(logits, dim=1)[:, 1]
        target_f = (target == 1).float()

        # per-image numerator/denominator so loss isn't dominated by easy batches
        intersect = (prob * target_f).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2.0 * intersect + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky loss — generalisation of Dice with separate FP / FN weights.

    Tversky = TP / (TP + alpha * FP + beta * FN)
    Loss    = 1 - Tversky

    ``alpha=beta=0.5`` recovers Dice. ``beta > alpha`` penalises FN more
    heavily, which is the typical knob for thin-crack tasks where recall is
    the bottleneck.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(logits, dim=1)[:, 1]
        target_f = (target == 1).float()

        tp = (prob * target_f).sum(dim=(1, 2))
        fp = (prob * (1.0 - target_f)).sum(dim=(1, 2))
        fn = ((1.0 - prob) * target_f).sum(dim=(1, 2))

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()


def build_loss(
    name: str,
    *,
    dice_smooth: float = 1.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> nn.Module:
    """Factory: ``name`` ∈ {'ce', 'dice', 'tversky'}."""
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return DiceLoss(smooth=dice_smooth)
    if name == "tversky":
        return TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=dice_smooth)
    raise ValueError(f"Unknown loss '{name}'. Choose from: ce, dice, tversky.")
