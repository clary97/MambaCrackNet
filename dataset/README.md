# Datasets

This folder is the **default location** for all datasets consumed by MambaCrackNet (configured in [`config.py`](../config.py)). Datasets are not tracked in git — only this README is.

The path can always be overridden at run time via the training / evaluation CLI flags (`--image-dir`, `--mask-dir`, `--image-test-dir`, `--mask-test-dir`), so you are free to keep the raw data wherever is convenient (e.g. on a NAS mount).

## Supported datasets

The PyTorch port has been developed and validated against the following four public crack-segmentation datasets:

| Short name | Full name | Source |
|---|---|---|
| **BCL**     | Bridge Crack Library                       | https://doi.org/10.7910/DVN/RURXSH |
| **NCCD-PF** | Pre-failure Narrow Concrete Crack Dataset  | https://zenodo.org/records/8215100 |
| **LCW**     | Labeled Cracks in the Wild                 | https://doi.org/10.7294/16624672.v2 |
| **CCSD**    | Concrete Crack Segmentation Dataset        | https://data.mendeley.com/datasets/jwsn7tfbrp/1 |

Please cite the original publications when using these datasets — refer to each source page for the appropriate citation.

## Expected folder layout

The code follows the CCSD convention — RGB images and binary masks split into train / test directories, with paired files matched by filename:

```
dataset/
├── rgb/         # training images          (.jpg / .png / .bmp)
├── BW/          # training masks (binary)  — 0 = background, 255 = crack
├── Test_rgb/    # test images
└── Test_BW/     # test masks
```

When juggling multiple datasets, keep each one in its own subfolder and either symlink the active one into the four top-level folders, or just pass full paths via the CLI:

```
dataset/
├── BCL/
│   └── {rgb, BW, Test_rgb, Test_BW}/
├── NCCD-PF/
│   └── ...
├── LCW/
│   └── ...
└── CCSD/
    └── ...
```

Image / mask pairs are matched by filename, so `BW/0001.jpg` must correspond to `rgb/0001.jpg`.

Default geometry assumed by the model: **512 × 512 RGB images**, **binary masks** (the `CrackDataset` rounds mask pixel values to `{0, 1}`).

## Where the source archives live (internal NAS)

For internal use, the four dataset archives are stored on NAS and accessible inside the container at:

```
/workspace/nas_200/minkyung/
├── Bridge Crack Library.zip
├── NCCD-PF_Dataset.zip
├── LCW.zip
└── CCSD.zip
```

Unzip the one(s) you need into this folder, then either symlink the active dataset or point the CLI at it:

```bash
# from the repo root
mkdir -p dataset
unzip "/workspace/nas_200/minkyung/CCSD.zip" -d dataset/CCSD
# (repeat for BCL / NCCD-PF / LCW as needed)

# option A — symlink the active dataset into the default location
ln -s CCSD/rgb       dataset/rgb
ln -s CCSD/BW        dataset/BW
ln -s CCSD/Test_rgb  dataset/Test_rgb
ln -s CCSD/Test_BW   dataset/Test_BW

# option B — pass full paths through the CLI
python train.py \
    --image-dir       dataset/CCSD/rgb \
    --mask-dir        dataset/CCSD/BW \
    --image-test-dir  dataset/CCSD/Test_rgb \
    --mask-test-dir   dataset/CCSD/Test_BW
```

## Per-dataset notes (folder structure after extracting)

The four archives ship in slightly different layouts. After unzipping, you may need to rename folders / split a train-test partition so that each one ends up in the `{rgb, BW, Test_rgb, Test_BW}` shape expected by [`data/dataset.py`](../data/dataset.py). Document any per-dataset preprocessing steps you apply here so the next person can reproduce them.

---

## Training pipeline & unified layout

We replicate the original MambaCrackNet result in two stages:

1. **Baseline** — train on **CCSD only** to reproduce the published checkpoint.
2. **Fine-tune** — continue training on a **mixture of all four datasets** (CCSD + BCL + NCCD-PF + LCW) to improve generalisation.

Because the four sources ship with **different folder layouts, file extensions, mask polarities, and naming conventions**, we first normalise everything into a single unified directory tree. The normalisation script lives next to the raw archives on the NAS:

- Script: `/workspace/nas_200/minkyung/unify_datasets.py`
- Output: `/workspace/nas_200/minkyung/unified/`

### Unified directory layout

```
/workspace/nas_200/minkyung/unified/
├── CCSD/         { images/  masks/ }
├── BCL_NonSteel/ { images/  masks/ }   # BCL split into two subsets
├── BCL_Steel/    { images/  masks/ }
├── NCCD/         { images/  masks/ }
├── LCW/          { images/  masks/ }   # Train + Test merged
└── summary.json                        # pair counts and crack-present stats
```

Inside each subset:

- `images/` — **symlinks** to the original image files (no re-encoding; saves disk and avoids quality loss).
- `masks/` — **freshly written PNGs**, mode `L`, strictly binary `{0, 255}`, with **background = 0 and crack = 255**.
- Image and mask filenames share the same stem (e.g. `images/0001.jpg` ↔ `masks/0001.png`).

### Normalisation rules per dataset

| Dataset | Image handling | Mask handling | Filename rule |
|---|---|---|---|
| **CCSD** | symlink originals (`.jpg`) | decoded to `L`, then **thresholded at 127** to absorb JPEG artifacts | keep original stem |
| **BCL_NonSteel** / **BCL_Steel** | symlink originals | decoded to `L`, then **inverted** (`255 - arr`) because the source uses *white background / black crack*, then thresholded | keep original stem; the two BCL subsets (`Non-steel crack images`, `Steel crack images`) are kept separate so they can be sampled independently |
| **NCCD-PF** | symlink originals | decoded to `L` (mask is grayscale; channel 0 used) and thresholded | source uses `image_N` / `mask_N` — both **renamed to `N`** so image and mask stems match |
| **LCW** | symlink originals | decoded to `L` and thresholded (already correct polarity) | original `Train/` and `Test/` are merged with **`train_` / `test_` filename prefix** to avoid collisions while preserving provenance |

Threshold used everywhere: `pixel > 127 → 255 (crack)`, otherwise `0`.

### Adapting the unified layout to the training code

The training code currently expects the CCSD-style `{rgb, BW, Test_rgb, Test_BW}` layout (see [Expected folder layout](#expected-folder-layout)). The unified tree uses `{images, masks}` without a train/test split. To bridge the two there are two options:

**(a) Project the unified data into the CCSD layout.** Decide a train/test split per subset, then symlink:

```bash
# example: use CCSD as the baseline, holding out e.g. the last 10% as test
cd /workspace/minkyung/Dron/MambaCrackNet
mkdir -p dataset/CCSD/{rgb,BW,Test_rgb,Test_BW}
# (split logic goes here — populate the four folders with symlinks)
```

**(b) Point the CLI directly at the unified tree** and split inside the dataloader. This avoids duplicating the directory layout but requires a small change to `data/dataset.py` to accept a single `{images, masks}` folder plus a train/test split ratio. Track this as a TODO if/when we go this route.

### Baseline run (CCSD only)

```bash
# from repo root, with the unified tree visible at /workspace/nas_200/minkyung/unified/
python train.py \
    --image-dir       /workspace/nas_200/minkyung/unified/CCSD/images \
    --mask-dir        /workspace/nas_200/minkyung/unified/CCSD/masks \
    --image-test-dir  /workspace/nas_200/minkyung/unified/CCSD/images \
    --mask-test-dir   /workspace/nas_200/minkyung/unified/CCSD/masks \
    --epochs 100 --batch-size 2
```

*(The example above reuses the same folder for train and test only as a placeholder — replace with a proper split once option (a) or (b) above is in place.)*

### Fine-tuning on the four-dataset mixture

After the CCSD baseline checkpoint is saved, use [`finetune.py`](../finetune.py) to continue training on the union of all five subset folders. It splits each source 80/20 with `--seed` (independent of the others), pools the train sides into a single shuffled loader (naive concat), and reports **per-source metrics at the end** so you can see how well the fine-tuned model generalises to each dataset individually.

```bash
nohup python finetune.py \
    --unified-root /workspace/nas_200/minkyung/unified \
    --datasets     CCSD,BCL_NonSteel,BCL_Steel,NCCD,LCW \
    --resume       ./checkpoints/ccsd_baseline.pt \
    --test-split   0.2 --seed 2024 \
    --epochs       10 --batch-size 2 --lr 1e-5 \
    --checkpoint   ./checkpoints/multidataset_finetune.pt \
    > logs/multidataset_finetune.log 2>&1 &
```

Defaults (Adam, `lr=1e-5`, 10 epochs, naive concat) match what we currently have in the main README; the script will reject an unknown subset name and abort early if any `<unified-root>/<name>/{images,masks}` is missing.

Per-source train/test pair counts under `--test-split 0.2 --seed 2024`:

| source | train | test |
|---|---:|---:|
| CCSD          |    357 |    89 |
| BCL_NonSteel  |  4,615 | 1,154 |
| BCL_Steel     |  1,629 |   407 |
| NCCD          |  4,310 | 1,078 |
| LCW           |  3,018 |   755 |
| **TOTAL**     | **13,929** | **3,483** |

(LCW count assumes the unify script has completed — re-run `/workspace/nas_200/minkyung/unify_datasets.py` if any of the `images/` folders look out of sync with `masks/`.)

**Sampling / loss knobs.** `finetune.py` now supports `--sampling {naive,balanced}` and `--loss {ce,dice,tversky}` (with optional `--tversky-alpha`/`--tversky-beta`) so the 4-way ablation (CCSD baseline vs naive+CE vs balanced+CE vs naive+Dice) is a one-flag change. See the *Sampling and loss options* section in the main [README](../README.md) for ready-to-paste commands.

`balanced` sampling assigns each pair weight `1 / n_source`, so the per-batch distribution becomes uniform across sources — small sources (CCSD, LCW) get heavily oversampled and large sources (BCL_NonSteel, NCCD) are seen ~60% of unique pairs per epoch.

### Status of the unification (as of last NAS check)

| Subset | Mask PNGs written | Notes |
|---|---:|---|
| BCL_NonSteel | 5,769 | masks inverted |
| BCL_Steel    | 2,036 | masks inverted |
| CCSD         |   446 | JPG masks rebinarised |
| LCW          |   796 | Train+Test merged with `train_`/`test_` prefix |
| NCCD         | 5,388 | `image_N`/`mask_N` renamed to `N` |

Image symlinks are populated by the same script — re-run `unify_datasets.py` if the `images/` folders ever look out of sync with `masks/`. The authoritative pair counts and crack-present statistics land in `/workspace/nas_200/minkyung/unified/summary.json`.
