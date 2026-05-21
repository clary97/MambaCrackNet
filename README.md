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

`checkpoints/` and `logs/` are created on demand by training runs (`mkdir -p logs checkpoints` once before the first run) and are both gitignored.

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

# 2c. train from a single {images, masks} folder with an internal train/test split
python train.py \
    --image-dir   /workspace/nas_200/minkyung/unified/CCSD/images \
    --mask-dir    /workspace/nas_200/minkyung/unified/CCSD/masks \
    --test-split  0.2 \
    --epochs 100 --batch-size 2

# 3. evaluate a saved checkpoint
python test.py \
    --image-test-dir dataset/CCSD/Test_rgb \
    --mask-test-dir  dataset/CCSD/Test_BW \
    --checkpoint ./checkpoints/mamba_crack_net.pt
```

Run the commands from the repository root so the `config`, `models`, `data`, and `utils` packages resolve correctly. The default model expects `512 × 512` RGB images and binary masks; image / mask pairs are matched by stem (extension-agnostic — e.g. `images/001.jpg` ↔ `masks/001.png`).

## Reproducing the CCSD baseline (original paper)

To replicate the original MambaCrackNet result, train on **CCSD only** using the unified data on NAS and the hyperparameters lifted from [tensorflow/MambaCrackNet-ForGithubVersion.ipynb](tensorflow/MambaCrackNet-ForGithubVersion.ipynb) (Adam `lr=1e-4`, batch=2, 100 epochs, 512×512 RGB, random horizontal flip):

```bash
python train.py \
    --image-dir   /workspace/nas_200/minkyung/unified/CCSD/images \
    --mask-dir    /workspace/nas_200/minkyung/unified/CCSD/masks \
    --test-split  0.2 \
    --seed        2024 \
    --epochs      100 \
    --batch-size  2 \
    --lr          1e-4 \
    --checkpoint  ./checkpoints/ccsd_baseline.pt
```

This:
- loads all 446 CCSD pairs from the unified folder and deterministically splits them **357 train / 89 test** (`test_ratio=0.2`, `seed=2024`);
- trains for 100 epochs with the original recipe and saves a new checkpoint every time the validation mean-IoU improves;
- after the final epoch, automatically reloads the best checkpoint and prints the full test-metric suite (`iou`, `accuracy`, `precision`, `recall`, `f1`, `mae`) — matching the metrics used in the original notebook.

To re-evaluate the saved baseline later without retraining:

```bash
python test.py \
    --image-test-dir /workspace/nas_200/minkyung/unified/CCSD/images \
    --mask-test-dir  /workspace/nas_200/minkyung/unified/CCSD/masks \
    --checkpoint     ./checkpoints/ccsd_baseline.pt
```

Note that `test.py` evaluates over **all** pairs in the folders you give it; to match the training-time test split exactly, point it at a held-out folder or symlink only the 89 test images / masks there.

## Running long training jobs (logs & background execution)

A full 100-epoch CCSD baseline takes roughly **3 hours** on a single RTX A5000 (≈14.9 GB peak GPU memory at batch=2). Run it in the background with `nohup` so it survives terminal disconnects, and redirect both stdout and stderr to a log file you can `tail -f` later.

### One-time setup

The redirect target directory has to exist *before* the shell parses the `>` operator — otherwise zsh aborts with `no such file or directory`. Create both folders once at the start:

```bash
cd /workspace/minkyung/Dron/MambaCrackNet
mkdir -p logs checkpoints
```

### Launch the training in the background

```bash
nohup python train.py \
    --image-dir   /workspace/nas_200/minkyung/unified/CCSD/images \
    --mask-dir    /workspace/nas_200/minkyung/unified/CCSD/masks \
    --test-split  0.2 --seed 2024 \
    --epochs 100 --batch-size 2 --lr 1e-4 \
    --checkpoint ./checkpoints/ccsd_baseline.pt \
    > logs/ccsd_baseline.log 2>&1 &
```

The shell prints `[1] <PID>` — note that PID, you'll use it to inspect or kill the job.

### Monitor progress

```bash
# follow the log live
tail -f logs/ccsd_baseline.log

# check the process is still alive
jobs -l                  # in the same shell where you launched it
ps -fp <PID>             # from anywhere
nvidia-smi               # GPU utilisation / memory

# stop the run cleanly
kill <PID>
```

Log lines you should see early on:

```
Single-source split: 357 train / 89 test (ratio=0.2, seed=2024)
Epoch    0  loss=...  valid_mean_iou=...
  -> saved checkpoint (./checkpoints/ccsd_baseline.pt)
...
=== Final test metrics (best checkpoint) ===
        iou: ...
   accuracy: ...
   ...
```

### Log conventions

- One log file per run, named after the experiment (`logs/ccsd_baseline.log`, `logs/ccsd_finetune.log`, …).
- The `logs/` folder is in [.gitignore](.gitignore) and never committed.
- Use `tee` instead of `>` if you want to follow the run live in the foreground while also keeping the log:
  ```bash
  python train.py ... 2>&1 | tee logs/ccsd_baseline.log
  ```

## Model overview

`MambaCrackNet` keeps two parallel pathways:

1. **Token pathway** — ViT-style patchify, then a stack of `MambaResidualBlock`s linked with `PatchMerging` (encoder) and `PatchExpanding` (decoder).
2. **Image pathway** — a convolutional U-Net built from `ConvResidualBlock`s.

The two streams exchange information at every scale: encoder image features are patchified and concatenated into the token stream, while decoded tokens are unpatchified into a pyramid that fuses back into the image stream. The pyramid features are upsampled, concatenated and projected to per-pixel logits with shape `(B, n_labels, H, W)`.
