"""Train MambaCrackNet on a paired (image, mask) crack-segmentation dataset.

Two ways to point at data:

1. Pre-split into train / test folders (CCSD-style):
       python train.py \
           --image-dir       /path/to/rgb \
           --mask-dir        /path/to/BW \
           --image-test-dir  /path/to/Test_rgb \
           --mask-test-dir   /path/to/Test_BW

2. Single ``{images, masks}`` folder, split internally (unified-tree style):
       python train.py \
           --image-dir   /workspace/nas_200/minkyung/unified/CCSD/images \
           --mask-dir    /workspace/nas_200/minkyung/unified/CCSD/masks \
           --test-split  0.2
"""

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import Config
from data import build_dataloaders, build_dataloaders_split
from models import MambaCrackNet
from utils import SegmentationMetrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train MambaCrackNet")
    parser.add_argument("--image-dir", default=cfg.data.image_dir)
    parser.add_argument("--mask-dir", default=cfg.data.mask_dir)
    parser.add_argument("--image-test-dir", default=cfg.data.image_test_dir)
    parser.add_argument("--mask-test-dir", default=cfg.data.mask_test_dir)
    parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help=(
            "When set (e.g. 0.2), ignore --image-test-dir/--mask-test-dir and "
            "split a single {images,masks} folder deterministically using --seed."
        ),
    )
    parser.add_argument("--epochs", type=int, default=cfg.train.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.data.batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.data.num_workers)
    parser.add_argument("--lr", type=float, default=cfg.train.lr)
    parser.add_argument("--checkpoint", default=cfg.train.checkpoint_path)
    parser.add_argument("--seed", type=int, default=cfg.train.seed)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = loss_fn(logits, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_iou(model, loader, device) -> float:
    """Fast per-epoch mean-IoU only."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=model.n_labels)
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(image)
        pred = logits.argmax(dim=1)
        metrics.update(pred.cpu(), mask.cpu())
    return metrics.averages()["iou"]


@torch.no_grad()
def evaluate_full(model, loader, device) -> Dict[str, float]:
    """End-of-training evaluation: IoU, accuracy, precision, recall, F1, MAE."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=model.n_labels)
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(image)
        pred = logits.argmax(dim=1)
        metrics.update(pred.cpu(), mask.cpu())
    return metrics.averages()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    cfg = Config()

    if args.test_split is not None:
        train_loader, test_loader, n_train, n_test = build_dataloaders_split(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            test_ratio=args.test_split,
            seed=args.seed,
            image_size=cfg.model.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(
            f"Single-source split: {n_train} train / {n_test} test "
            f"(ratio={args.test_split}, seed={args.seed})",
            flush=True,
        )
    else:
        train_loader, test_loader = build_dataloaders(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            image_test_dir=args.image_test_dir,
            mask_test_dir=args.mask_test_dir,
            image_size=cfg.model.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    model = MambaCrackNet(
        input_size=cfg.model.input_size,
        in_channels=cfg.model.in_channels,
        filter_num_begin=cfg.model.filter_num_begin,
        depth=cfg.model.depth,
        patch_size=cfg.model.patch_size,
        n_labels=cfg.model.n_labels,
        d_state=cfg.model.d_state,
        expand=cfg.model.expand,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_iou = evaluate_iou(model, test_loader, device)
        print(
            f"Epoch {epoch:4d}  loss={train_loss:.6f}  valid_mean_iou={valid_iou:.6f}",
            flush=True,
        )
        if valid_iou > best_iou:
            best_iou = valid_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "valid_iou": valid_iou,
                },
                args.checkpoint,
            )
            print(f"  -> saved checkpoint ({args.checkpoint})", flush=True)

    # ---- final evaluation with the best checkpoint -------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"\nLoaded best checkpoint from epoch {ckpt['epoch']} "
        f"(valid_mean_iou={ckpt['valid_iou']:.6f}) for final evaluation.",
        flush=True,
    )
    final = evaluate_full(model, test_loader, device)
    print("\n=== Final test metrics (best checkpoint) ===")
    for k, v in final.items():
        print(f"  {k:>9s}: {v:.6f}")


if __name__ == "__main__":
    main()
