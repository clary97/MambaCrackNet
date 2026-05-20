"""Centralised configuration for MambaCrackNet (PyTorch)."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    input_size: Tuple[int, int] = (512, 512)
    in_channels: int = 3
    filter_num_begin: int = 64
    depth: int = 4
    patch_size: Tuple[int, int] = (4, 4)
    n_labels: int = 2
    d_state: int = 16
    expand: int = 2


@dataclass
class DataConfig:
    image_dir: str = "./dataset/rgb"
    mask_dir: str = "./dataset/BW"
    image_test_dir: str = "./dataset/Test_rgb"
    mask_test_dir: str = "./dataset/Test_BW"
    batch_size: int = 2
    num_workers: int = 2


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    checkpoint_path: str = "./checkpoints/mamba_crack_net.pt"
    log_every: int = 10
    seed: int = 2024


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
