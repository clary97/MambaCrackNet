"""Render inference visualisations for a trained MambaCrackNet checkpoint.

For each image in the chosen split, write a single PNG that tiles
    [ input | ground-truth mask | predicted mask | red-overlay on input ]
horizontally so you can browse the predictions in any image viewer / VSCode.

Examples:
    # Visualise the same 89 test pairs the CCSD baseline trained against
    python inference.py \
        --image-dir   /workspace/nas_200/minkyung/unified/CCSD/images \
        --mask-dir    /workspace/nas_200/minkyung/unified/CCSD/masks \
        --test-split  0.2 --seed 2024 \
        --checkpoint  ./checkpoints/ccsd_baseline.pt \
        --output-dir  ./predictions/ccsd_baseline

    # Visualise every pair in some folder (no split)
    python inference.py \
        --image-dir   /path/to/images --mask-dir /path/to/masks \
        --checkpoint  ./checkpoints/ccsd_baseline.pt \
        --output-dir  ./predictions/all
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFile, ImageFont

from config import Config
from data.dataset import _collect_pairs, split_pairs
from models import MambaCrackNet
from utils import mean_iou

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Render MambaCrackNet predictions")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help="If set, render only the held-out portion of a deterministic split.",
    )
    parser.add_argument("--seed", type=int, default=cfg.train.seed)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="./predictions")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on number of images to render (default: render all).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _load_pairs(image_dir: str, mask_dir: str, test_split, seed: int):
    all_pairs = _collect_pairs(image_dir, mask_dir)
    if not all_pairs:
        raise RuntimeError(f"No image/mask pairs in {image_dir} / {mask_dir}")
    if test_split is None:
        return all_pairs
    _train, test = split_pairs(all_pairs, test_split, seed)
    return test


def _load_image(path: str, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _load_mask(path: str, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize((size[1], size[0]), Image.NEAREST)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return np.round(arr).astype(np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask > 0] = 255
    return rgb


def _overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a red highlight over `image` wherever mask==1."""
    out = image.astype(np.float32).copy()
    red = np.array([220.0, 30.0, 30.0], dtype=np.float32)
    sel = mask > 0
    out[sel] = (1.0 - alpha) * out[sel] + alpha * red
    return out.clip(0, 255).astype(np.uint8)


def _label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    """Add a small caption strip on top of a panel."""
    h, w = panel.shape[:2]
    strip_h = 24
    canvas = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    canvas[strip_h:] = panel
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def _compose(image, gt_mask, pred_mask, iou_val: float) -> Image.Image:
    panels = [
        _label_panel(image, "input"),
        _label_panel(_mask_to_rgb(gt_mask), "ground truth"),
        _label_panel(_mask_to_rgb(pred_mask), f"prediction  IoU={iou_val:.3f}"),
        _label_panel(_overlay(image, pred_mask), "overlay (pred = red)"),
    ]
    return Image.fromarray(np.concatenate(panels, axis=1))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = Config()
    device = torch.device(args.device)

    pairs: List[Tuple[str, str]] = _load_pairs(
        args.image_dir, args.mask_dir, args.test_split, args.seed
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(
        f"Loaded checkpoint {args.checkpoint} "
        f"(epoch={ckpt.get('epoch', '?')}, valid_iou={ckpt.get('valid_iou', float('nan')):.4f})"
    )
    print(f"Rendering {len(pairs)} pair(s) to {out_dir}")

    H, W = cfg.model.input_size
    for idx, (img_path, mask_path) in enumerate(pairs):
        image_u8 = _load_image(img_path, (H, W))
        mask_u8 = _load_mask(mask_path, (H, W))

        # match training-time normalisation: [-1, 1], CHW float32
        img_norm = image_u8.astype(np.float32) / 255.0 * 2.0 - 1.0
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        logits = model(img_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        iou_val = mean_iou(
            torch.from_numpy(pred), torch.from_numpy(mask_u8), num_classes=cfg.model.n_labels
        )

        stem = Path(img_path).stem
        out_path = out_dir / f"{stem}.png"
        _compose(image_u8, mask_u8, pred, iou_val).save(out_path, optimize=True)

        if (idx + 1) % 10 == 0 or idx == len(pairs) - 1:
            print(f"  [{idx + 1}/{len(pairs)}] {out_path}")

    print(f"\nDone. Open the PNGs under {out_dir} to inspect predictions.")


if __name__ == "__main__":
    main()
