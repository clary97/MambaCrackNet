"""Fine-tune MambaCrackNet on a mixture of unified crack-segmentation datasets.

Loads weights from a baseline checkpoint (typically the CCSD reproduction) and
continues training on the naive concatenation of the requested unified subsets.
At the end, reports per-source metrics so generalisation can be inspected.

Example:
    python finetune.py \
        --unified-root /workspace/nas_200/minkyung/unified \
        --datasets     CCSD,BCL_NonSteel,BCL_Steel,NCCD,LCW \
        --resume       ./checkpoints/ccsd_baseline.pt \
        --epochs       10 \
        --lr           1e-5 \
        --checkpoint   ./checkpoints/multidataset_finetune.pt
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import Config
from data import build_multi_dataloaders_split
from models import MambaCrackNet
from utils import SegmentationMetrics


DEFAULT_DATASETS = "CCSD,BCL_NonSteel,BCL_Steel,NCCD,LCW"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Fine-tune MambaCrackNet on multi-source data")
    parser.add_argument(
        "--unified-root",
        default="/workspace/nas_200/minkyung/unified",
        help="Root that contains <name>/{images, masks} subfolders for each dataset.",
    )
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help=f"Comma-separated subset names. Default: {DEFAULT_DATASETS}",
    )
    parser.add_argument(
        "--resume",
        required=True,
        help="Path to a baseline checkpoint to initialise weights from "
        "(e.g. ./checkpoints/ccsd_baseline.pt).",
    )
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=cfg.data.batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.data.num_workers)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/multidataset_finetune.pt",
    )
    parser.add_argument("--seed", type=int, default=cfg.train.seed)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_sources(unified_root: str, dataset_names: List[str]) -> Dict[str, tuple]:
    root = Path(unified_root)
    sources = {}
    for name in dataset_names:
        img_dir = root / name / "images"
        msk_dir = root / name / "masks"
        if not img_dir.is_dir() or not msk_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset '{name}' missing under {root} "
                f"(expected {img_dir} and {msk_dir})"
            )
        sources[name] = (str(img_dir), str(msk_dir))
    return sources


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

    dataset_names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    sources = build_sources(args.unified_root, dataset_names)

    train_loader, combined_test_loader, per_source_test_loaders, counts = (
        build_multi_dataloaders_split(
            sources=sources,
            test_ratio=args.test_split,
            seed=args.seed,
            image_size=cfg.model.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    print("Per-source pair counts (test_split={:.2f}, seed={}):".format(args.test_split, args.seed))
    print(f"  {'source':<14}  {'train':>6}  {'test':>6}")
    n_train_total = n_test_total = 0
    for name in dataset_names:
        n_tr, n_te = counts[name]
        n_train_total += n_tr
        n_test_total += n_te
        print(f"  {name:<14}  {n_tr:>6}  {n_te:>6}")
    print(f"  {'TOTAL':<14}  {n_train_total:>6}  {n_test_total:>6}", flush=True)

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

    resume_path = Path(args.resume)
    if not resume_path.exists():
        raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
    base_ckpt = torch.load(resume_path, map_location=device)
    state = base_ckpt.get("model_state_dict", base_ckpt)
    model.load_state_dict(state)
    print(
        f"\nResumed weights from {resume_path} "
        f"(epoch={base_ckpt.get('epoch', '?')}, "
        f"valid_iou={base_ckpt.get('valid_iou', float('nan')):.4f})",
        flush=True,
    )

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_iou = evaluate_iou(model, combined_test_loader, device)
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
                    "sources": list(sources.keys()),
                    "resume_from": str(resume_path),
                },
                args.checkpoint,
            )
            print(f"  -> saved checkpoint ({args.checkpoint})", flush=True)

    # ---- final per-source evaluation with the best checkpoint --------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"\nLoaded best checkpoint from epoch {ckpt['epoch']} "
        f"(valid_mean_iou={ckpt['valid_iou']:.6f}) for final evaluation.",
        flush=True,
    )

    print("\n=== Per-source test metrics (best checkpoint) ===")
    for name in dataset_names:
        m = evaluate_full(model, per_source_test_loaders[name], device)
        print(f"\n  [{name}]   n_test={counts[name][1]}")
        for k, v in m.items():
            print(f"    {k:>10s}: {v:.6f}")

    print("\n=== Aggregate test metrics (all sources combined) ===")
    m = evaluate_full(model, combined_test_loader, device)
    for k, v in m.items():
        print(f"  {k:>10s}: {v:.6f}")


if __name__ == "__main__":
    main()
