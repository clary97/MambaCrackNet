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
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import Config
from data import build_multi_dataloaders_split
from models import MambaCrackNet
from utils import SegmentationMetrics, build_loss


DEFAULT_DATASETS = "CCSD,BCL_NonSteel,BCL_Steel,NCCD,LCW"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(rng: Optional[dict]) -> None:
    if not rng:
        return
    try:
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    except Exception as exc:  # best-effort; non-fatal
        print(f"[warn] could not fully restore RNG state: {exc}", flush=True)


def save_training_state(path, model, optimizer, epoch, best_iou, args, sources) -> None:
    """Full state for crash recovery — overwritten every epoch."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "sources": list(sources.keys()),
            "sampling": args.sampling,
            "loss": args.loss,
            "lr": args.lr,
            "rng": _rng_state(),
        },
        path,
    )


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
        default=None,
        help="Path to a baseline checkpoint to initialise weights from "
        "(e.g. ./checkpoints/ccsd_baseline.pt). Required unless --resume-training is given.",
    )
    parser.add_argument(
        "--resume-training",
        default=None,
        help="Continue an interrupted run from a '*.last.pt' state checkpoint "
        "(restores model + optimizer + epoch + best_iou + RNG). Other flags "
        "should match the original run.",
    )
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=cfg.data.batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.data.num_workers)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--sampling",
        choices=["naive", "balanced"],
        default="naive",
        help=(
            "naive    — pool all train pairs and shuffle (large sources dominate).\n"
            "balanced — WeightedRandomSampler so each source contributes ~equally "
            "per batch (small sources oversampled)."
        ),
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Only used with --sampling balanced. Defaults to the pooled train size.",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "dice", "tversky", "ce_dice"],
        default="ce",
        help=(
            "ce      — CrossEntropyLoss (default; matches original recipe).\n"
            "dice    — soft Dice on the crack class (class-imbalance robust).\n"
            "tversky — Tversky with alpha/beta knobs for recall vs precision.\n"
            "ce_dice — 0.5*CE + 0.5*Dice (stable gradient + imbalance robust; "
            "avoids the pure-Dice collapse)."
        ),
    )
    parser.add_argument(
        "--tversky-alpha",
        type=float,
        default=0.3,
        help="FP weight in Tversky (smaller = more FP tolerated). Only with --loss tversky.",
    )
    parser.add_argument(
        "--tversky-beta",
        type=float,
        default=0.7,
        help="FN weight in Tversky (larger = recall favoured). Only with --loss tversky.",
    )
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
            sampling=args.sampling,
            samples_per_epoch=args.samples_per_epoch,
        )
    )

    print(
        f"Sampling      : {args.sampling}"
        + (f"  (samples_per_epoch={args.samples_per_epoch})" if args.sampling == "balanced" and args.samples_per_epoch else "")
    )
    print(
        f"Loss          : {args.loss}"
        + (f"  (alpha={args.tversky_alpha}, beta={args.tversky_beta})" if args.loss == "tversky" else "")
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

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = build_loss(
        args.loss,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
    )

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)
    # full-state checkpoint for crash recovery, e.g. foo.pt -> foo.last.pt
    state_ckpt_path = ckpt_path.with_name(ckpt_path.stem + ".last" + ckpt_path.suffix)

    start_epoch = 0
    best_iou = -1.0
    resume_from = args.resume

    if args.resume_training:
        # ---- continue an interrupted run (model + optimizer + epoch + RNG) ----
        rt_path = Path(args.resume_training)
        if not rt_path.exists():
            raise FileNotFoundError(f"--resume-training checkpoint not found: {rt_path}")
        # our own trusted file; contains numpy/python RNG state that the default
        # weights_only=True loader (PyTorch >= 2.6) refuses to unpickle.
        st = torch.load(rt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        optimizer.load_state_dict(st["optimizer_state_dict"])
        start_epoch = int(st["epoch"]) + 1
        best_iou = float(st.get("best_iou", -1.0))
        _restore_rng(st.get("rng"))
        resume_from = str(rt_path)
        print(
            f"\nResumed TRAINING from {rt_path}: continuing at epoch "
            f"{start_epoch}/{args.epochs} (best_iou so far={best_iou:.6f})",
            flush=True,
        )
    else:
        # ---- fresh fine-tune: initialise weights from a baseline checkpoint ----
        if not args.resume:
            raise ValueError(
                "Provide --resume (fresh fine-tune from a baseline) "
                "or --resume-training (continue an interrupted run)."
            )
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        base_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(base_ckpt.get("model_state_dict", base_ckpt))
        print(
            f"\nResumed weights from {resume_path} "
            f"(epoch={base_ckpt.get('epoch', '?')}, "
            f"valid_iou={base_ckpt.get('valid_iou', float('nan')):.4f})",
            flush=True,
        )

    if start_epoch >= args.epochs:
        print(
            f"start_epoch ({start_epoch}) >= --epochs ({args.epochs}); "
            "skipping training and running final evaluation only.",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
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
                    "resume_from": str(resume_from),
                    "sampling": args.sampling,
                    "loss": args.loss,
                    "tversky_alpha": args.tversky_alpha if args.loss == "tversky" else None,
                    "tversky_beta": args.tversky_beta if args.loss == "tversky" else None,
                    "lr": args.lr,
                },
                args.checkpoint,
            )
            print(f"  -> saved best checkpoint ({args.checkpoint})", flush=True)

        # always persist full training state so a reboot can resume this epoch
        save_training_state(state_ckpt_path, model, optimizer, epoch, best_iou, args, sources)

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
