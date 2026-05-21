# MambaCrackNet

PyTorch re-implementation of *"Enhancing Pixel-Level Crack Segmentation with Visual Mamba and Convolutional Networks"* (Automation in Construction).

- Original (TensorFlow) repository: https://github.com/ChengjiaHanSEU/MambaCrackNet
- Paper: "Enhancing Pixel-Level Crack Segmentation with Visual Mamba and CNN"

The PyTorch port lives at the repository root; the original TensorFlow notebook is preserved under [`tensorflow/`](tensorflow/) for reference.

## Repository layout

```
MambaCrackNet/
├── README.md
├── requirements.txt
├── config.py                              # ModelConfig / DataConfig / TrainConfig dataclasses
├── train.py                               # training entry point
├── test.py                                # evaluation entry point
├── models/
│   ├── drop_path.py                       # DropPath stochastic depth
│   ├── mamba.py                           # RMSNorm, selective_scan, SSM, MambaBlock, MambaResidualBlock
│   ├── patches.py                         # PatchExtract / PatchEmbedding / PatchMerging / PatchExpanding
│   ├── blocks.py                          # Mlp, ConvResidualBlock (image-domain)
│   └── mamba_crack_net.py                 # full MambaCrackNet model
├── data/
│   └── dataset.py                         # CrackDataset + build_dataloaders
├── utils/
│   └── metrics.py                         # IoU / accuracy / precision / recall / F1 / MAE
├── dataset/                               # default location for raw datasets
│   └── README.md                          # dataset sources, layout, NAS paths
└── tensorflow/                            # original TensorFlow notebook (kept for reference)
    └── MambaCrackNet-ForGithubVersion.ipynb
```

`checkpoints/` is created automatically the first time training saves a model.

## Datasets

The PyTorch port targets four public crack-segmentation datasets: **BCL**, **NCCD-PF**, **LCW**, and **CCSD**. By convention all data is kept under [`dataset/`](dataset/) (the default in [`config.py`](config.py)), though every path can be overridden via CLI flags.

See **[`dataset/README.md`](dataset/README.md)** for the full list of sources / DOIs, the expected folder layout, the NAS location of the source archives, and per-dataset preprocessing notes.

## Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2a. train using the default ./dataset/{rgb,BW,Test_rgb,Test_BW} layout
python train.py --epochs 100 --batch-size 2

# 2b. train against an arbitrary dataset location (override the defaults)
python train.py \
    --image-dir       dataset/CCSD/rgb \
    --mask-dir        dataset/CCSD/BW \
    --image-test-dir  dataset/CCSD/Test_rgb \
    --mask-test-dir   dataset/CCSD/Test_BW \
    --epochs 100 --batch-size 2

# 3. evaluate a saved checkpoint
python test.py \
    --image-test-dir dataset/CCSD/Test_rgb \
    --mask-test-dir  dataset/CCSD/Test_BW \
    --checkpoint ./checkpoints/mamba_crack_net.pt
```

Run the commands from the repository root so the `config`, `models`, `data`, and `utils` packages resolve correctly. The default model expects `512 × 512` RGB images and binary masks; image / mask pairs are matched by filename.

## Model overview

`MambaCrackNet` keeps two parallel pathways:

1. **Token pathway** — ViT-style patchify, then a stack of `MambaResidualBlock`s linked with `PatchMerging` (encoder) and `PatchExpanding` (decoder).
2. **Image pathway** — a convolutional U-Net built from `ConvResidualBlock`s.

The two streams exchange information at every scale: encoder image features are patchified and concatenated into the token stream, while decoded tokens are unpatchified into a pyramid that fuses back into the image stream. The pyramid features are upsampled, concatenated and projected to per-pixel logits with shape `(B, n_labels, H, W)`.
