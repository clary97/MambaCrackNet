"""Segmentation metrics: IoU, accuracy, precision, recall, F1, MAE."""

from dataclasses import dataclass, field
from typing import Dict, List

import torch


def _confusion_counts(pred: torch.Tensor, target: torch.Tensor):
    """Return (tp, fp, fn, tn) treating class 1 as positive."""
    pred = pred.view(-1)
    target = target.view(-1)
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    return tp, fp, fn, tn


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """Mean IoU across classes (Keras-style: per-class IoU averaged)."""
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(inter / union)
    valid = [v for v in ious if v == v]  # drop NaNs
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


@dataclass
class SegmentationMetrics:
    """Per-sample metric collector, mirroring the original notebook."""

    num_classes: int = 2
    iou: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    f1: List[float] = field(default_factory=list)
    mae: List[float] = field(default_factory=list)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """`pred` and `target` are integer class-index tensors with the same shape."""
        for p, t in zip(pred, target):
            self.iou.append(mean_iou(p, t, num_classes=self.num_classes))
            tp, fp, fn, tn = _confusion_counts(p, t)
            total = tp + fp + fn + tn
            self.accuracy.append((tp + tn) / total if total else 0.0)
            self.precision.append(tp / (tp + fp) if (tp + fp) else 0.0)
            self.recall.append(tp / (tp + fn) if (tp + fn) else 0.0)
            denom = self.precision[-1] + self.recall[-1]
            self.f1.append(
                2 * self.precision[-1] * self.recall[-1] / denom if denom else 0.0
            )
            diff = (p != t).float().mean().item()
            self.mae.append(diff)

    def averages(self) -> Dict[str, float]:
        def avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "iou": avg(self.iou),
            "accuracy": avg(self.accuracy),
            "precision": avg(self.precision),
            "recall": avg(self.recall),
            "f1": avg(self.f1),
            "mae": avg(self.mae),
        }
