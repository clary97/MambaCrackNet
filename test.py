"""Evaluate a trained MambaCrackNet checkpoint on a test directory."""

import argparse
from pathlib import Path

import torch

from config import Config
from data import CrackDataset
from data.dataset import _collect_pairs, split_pairs
from models import MambaCrackNet
from utils import SegmentationMetrics


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Evaluate MambaCrackNet")
    parser.add_argument("--image-test-dir", default=cfg.data.image_test_dir)
    parser.add_argument("--mask-test-dir", default=cfg.data.mask_test_dir)
    parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help=(
            "If set, evaluate only the held-out portion of a deterministic split. "
            "Use the same value of --seed used during training to reproduce its test set."
        ),
    )
    parser.add_argument("--seed", type=int, default=cfg.train.seed)
    parser.add_argument("--checkpoint", default=cfg.train.checkpoint_path)
    parser.add_argument("--batch-size", type=int, default=cfg.data.batch_size)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = Config()

    if args.test_split is not None:
        all_pairs = _collect_pairs(args.image_test_dir, args.mask_test_dir)
        _train, test_pairs = split_pairs(all_pairs, args.test_split, args.seed)
        print(
            f"Single-source split: evaluating {len(test_pairs)} held-out pair(s) "
            f"out of {len(all_pairs)} (ratio={args.test_split}, seed={args.seed})"
        )
        dataset = CrackDataset(
            pairs=test_pairs, image_size=cfg.model.input_size, train=False
        )
    else:
        dataset = CrackDataset(
            args.image_test_dir,
            args.mask_test_dir,
            image_size=cfg.model.input_size,
            train=False,
        )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
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

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    metrics = SegmentationMetrics(num_classes=cfg.model.n_labels)
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(image)
        pred = logits.argmax(dim=1)
        metrics.update(pred.cpu(), mask.cpu())

    avg = metrics.averages()
    print("=== Test results ===")
    for k, v in avg.items():
        print(f"  {k:>9s}: {v:.6f}")


if __name__ == "__main__":
    main()
