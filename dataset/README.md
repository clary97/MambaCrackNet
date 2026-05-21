# Datasets

This folder is the **default location** for all datasets consumed by MambaCrackNet (configured in [`pytorch/config.py`](../pytorch/config.py)). Datasets are not tracked in git — only this README is.

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
python -m pytorch.train \
    --image-dir       dataset/CCSD/rgb \
    --mask-dir        dataset/CCSD/BW \
    --image-test-dir  dataset/CCSD/Test_rgb \
    --mask-test-dir   dataset/CCSD/Test_BW
```

## Per-dataset notes (folder structure after extracting)

The four archives ship in slightly different layouts. After unzipping, you may need to rename folders / split a train-test partition so that each one ends up in the `{rgb, BW, Test_rgb, Test_BW}` shape expected by [`pytorch/data/dataset.py`](../pytorch/data/dataset.py). Document any per-dataset preprocessing steps you apply here so the next person can reproduce them.
