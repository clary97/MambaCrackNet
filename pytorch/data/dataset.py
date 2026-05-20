"""Dataset / DataLoader utilities for the concrete crack segmentation task."""

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    pairs: List[Tuple[str, str]] = []
    for entry in sorted(image_dir.iterdir()):
        if entry.suffix.lower() not in IMG_EXTS:
            continue
        # The original notebook pairs files by identical filename across folders.
        mask_path = mask_dir / entry.name
        if mask_path.exists():
            pairs.append((str(entry), str(mask_path)))
    return pairs


class CrackDataset(Dataset):
    """Paired RGB image + binary mask dataset.

    The image is rescaled to [-1, 1]; the mask is rounded to {0, 1} and returned
    as a long tensor of class indices, ready for `nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        train: bool = True,
        horizontal_flip_prob: float = 0.5,
    ):
        self.pairs = _collect_pairs(image_dir, mask_dir)
        if not self.pairs:
            raise RuntimeError(
                f"No image/mask pairs found in {image_dir} <-> {mask_dir}"
            )
        self.image_size = image_size
        self.train = train
        self.hflip_prob = horizontal_flip_prob

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            im = im.convert("RGB").resize(
                (self.image_size[1], self.image_size[0]), Image.BILINEAR
            )
            return np.asarray(im, dtype=np.float32)

    def _load_mask(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            im = im.convert("L").resize(
                (self.image_size[1], self.image_size[0]), Image.NEAREST
            )
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return np.round(arr).astype(np.int64)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if self.train and random.random() < self.hflip_prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # Normalize to [-1, 1] (matches the original notebook)
        image = image / 255.0 * 2.0 - 1.0
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


def build_dataloaders(
    image_dir: str,
    mask_dir: str,
    image_test_dir: str,
    mask_test_dir: str,
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = CrackDataset(image_dir, mask_dir, image_size=image_size, train=True)
    test_ds = CrackDataset(
        image_test_dir, mask_test_dir, image_size=image_size, train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, test_loader
