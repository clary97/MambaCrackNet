"""Dataset / DataLoader utilities for the concrete crack segmentation task."""

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# Some images in the public CCSD distribution (e.g. 593.JPG) are missing the EOI
# marker and PIL refuses to decode them by default. Allow decoding to consume
# whatever bytes are available rather than crashing the worker.
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Pair images and masks by *stem* (extension-agnostic).

    Matches the unified-tree layout where images may be ``.jpg`` / ``.JPG`` and
    the corresponding mask is ``.png`` with the same stem.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    mask_by_stem = {
        p.stem: p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }
    pairs: List[Tuple[str, str]] = []
    for entry in sorted(image_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in IMG_EXTS:
            continue
        mask_path = mask_by_stem.get(entry.stem)
        if mask_path is not None:
            pairs.append((str(entry), str(mask_path)))
    return pairs


def split_pairs(
    pairs: Sequence[Tuple[str, str]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Deterministic train/test split of an (image, mask) pair list."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    return shuffled[n_test:], shuffled[:n_test]


class CrackDataset(Dataset):
    """Paired RGB image + binary mask dataset.

    The image is rescaled to [-1, 1]; the mask is rounded to {0, 1} and returned
    as a long tensor of class indices, ready for ``nn.CrossEntropyLoss``.

    You can either point at a folder pair (``image_dir`` + ``mask_dir``) or pass
    a pre-built ``pairs`` list when you want fine-grained control over splits.
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        *,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        image_size: Tuple[int, int] = (512, 512),
        train: bool = True,
        horizontal_flip_prob: float = 0.5,
    ):
        if pairs is None:
            if image_dir is None or mask_dir is None:
                raise ValueError("provide either `pairs` or both `image_dir` and `mask_dir`")
            pairs = _collect_pairs(image_dir, mask_dir)
        if not pairs:
            raise RuntimeError(
                f"No image/mask pairs found in {image_dir} <-> {mask_dir}"
            )
        self.pairs = list(pairs)
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


def _make_loader(ds: CrackDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )


def build_dataloaders(
    image_dir: str,
    mask_dir: str,
    image_test_dir: str,
    mask_test_dir: str,
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build (train, test) loaders from two pre-split folders."""
    train_ds = CrackDataset(image_dir, mask_dir, image_size=image_size, train=True)
    test_ds = CrackDataset(image_test_dir, mask_test_dir, image_size=image_size, train=False)
    return (
        _make_loader(train_ds, batch_size, num_workers, shuffle=True),
        _make_loader(test_ds, batch_size, num_workers, shuffle=False),
    )


def build_dataloaders_split(
    image_dir: str,
    mask_dir: str,
    test_ratio: float = 0.2,
    seed: int = 2024,
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Build (train, test) loaders by splitting a single ``{images, masks}`` folder.

    Returns the loaders alongside the (train, test) pair counts so the caller can log them.
    """
    all_pairs = _collect_pairs(image_dir, mask_dir)
    if not all_pairs:
        raise RuntimeError(f"No image/mask pairs found in {image_dir} <-> {mask_dir}")
    train_pairs, test_pairs = split_pairs(all_pairs, test_ratio, seed)
    train_ds = CrackDataset(pairs=train_pairs, image_size=image_size, train=True)
    test_ds = CrackDataset(pairs=test_pairs, image_size=image_size, train=False)
    return (
        _make_loader(train_ds, batch_size, num_workers, shuffle=True),
        _make_loader(test_ds, batch_size, num_workers, shuffle=False),
        len(train_pairs),
        len(test_pairs),
    )
