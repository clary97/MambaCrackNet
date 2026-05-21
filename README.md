# MambaCrackNet

Code for the published paper titled "Enhancing Pixel-Level Crack Segmentation with Visual Mamba and Convolutional Networks" on Automation in Construction.

- Original: https://github.com/ChengjiaHanSEU/MambaCrackNet
- Paper: "Enhancing Pixel-Level Crack Segmentation with Visual Mamba and CNN"

This repository ships both the original TensorFlow notebook and a re-implementation in PyTorch organised as a small package.

## Repository layout

```
MambaCrackNet/
├── MambaCrackNet-ForGithubVersion.ipynb   # original TensorFlow notebook (kept for reference)
├── README.md
├── dataset/                               # default dataset location (see dataset/README.md)
│   └── README.md
└── pytorch/                               # PyTorch re-implementation
    ├── config.py                          # ModelConfig / DataConfig / TrainConfig dataclasses
    ├── train.py                           # training entry point
    ├── test.py                            # evaluation entry point
    ├── requirements.txt
    ├── models/
    │   ├── drop_path.py                   # DropPath stochastic depth
    │   ├── mamba.py                       # RMSNorm, selective_scan, SSM, MambaBlock, MambaResidualBlock
    │   ├── patches.py                     # PatchExtract / PatchEmbedding / PatchMerging / PatchExpanding
    │   ├── blocks.py                      # Mlp, ConvResidualBlock (image-domain)
    │   └── mamba_crack_net.py             # full MambaCrackNet model
    ├── data/
    │   └── dataset.py                     # CrackDataset + build_dataloaders
    ├── utils/
    │   └── metrics.py                     # IoU / accuracy / precision / recall / F1 / MAE
    └── checkpoints/                       # default save location
```

## Datasets

The PyTorch port targets four public crack-segmentation datasets: **BCL**, **NCCD-PF**, **LCW**, and **CCSD**. By convention all data is kept under [`dataset/`](dataset/) (the default in [`pytorch/config.py`](pytorch/config.py)), though every path can be overridden via CLI flags.

See **[`dataset/README.md`](dataset/README.md)** for the full list of sources / DOIs, the expected folder layout, the NAS location of the source archives, and per-dataset preprocessing notes.

## Quick start (PyTorch)

```bash
# 1. install dependencies
pip install -r pytorch/requirements.txt

# 2a. train using the default ./dataset/{rgb,BW,Test_rgb,Test_BW} layout
python -m pytorch.train --epochs 100 --batch-size 2

# 2b. train against an arbitrary dataset location (override the defaults)
python -m pytorch.train \
    --image-dir       dataset/CCSD/rgb \
    --mask-dir        dataset/CCSD/BW \
    --image-test-dir  dataset/CCSD/Test_rgb \
    --mask-test-dir   dataset/CCSD/Test_BW \
    --epochs 100 --batch-size 2

# 3. evaluate a saved checkpoint
python -m pytorch.test \
    --image-test-dir dataset/CCSD/Test_rgb \
    --mask-test-dir  dataset/CCSD/Test_BW \
    --checkpoint ./checkpoints/mamba_crack_net.pt
```

The default model expects `512 × 512` RGB images and binary masks; image / mask pairs are matched by filename.

## Model overview

`MambaCrackNet` keeps two parallel pathways:

1. **Token pathway** — ViT-style patchify, then a stack of `MambaResidualBlock`s linked with `PatchMerging` (encoder) and `PatchExpanding` (decoder).
2. **Image pathway** — a convolutional U-Net built from `ConvResidualBlock`s.

The two streams exchange information at every scale: encoder image features are patchified and concatenated into the token stream, while decoded tokens are unpatchified into a pyramid that fuses back into the image stream. The pyramid features are upsampled, concatenated and projected to per-pixel logits with shape `(B, n_labels, H, W)`.
