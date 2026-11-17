# epic-kitchens-dinov3
# EPIC-KITCHENS Action Recognition with DINOv3

This project uses a frozen DINOv3 ViT-B/16 backbone to extract
features from 16 frames per clip at 384 × 384 resolution.
A temporal transformer predicts verb and noun classes.

## Project files

- src/dinov3_base_action_model.py — model definition
- scripts/train_dinov3_base_384.py — feature extraction and training
- scripts/train_dinov3_base_384.slurm — Wice job submission
- results/splits/train_readable.csv — training annotations
- results/splits/val_readable.csv — validation annotations

Use the supplied CSV files unchanged to reproduce the same splits
and match the feature caches.

## Environment

The original Wice job uses Python 3.11.5.

The code requires PyTorch, torchvision, pandas, OpenCV, Pillow,
and the DINOv3 source code. Exact package versions and the DINOv3
source revision still need to be recorded from the working environment.

## Data and pretrained weights

The original video location is:

/scratch/leuven/380/vsc38054/epic-kitchens/videos_640x360

The required DINOv3 backbone checkpoint belongs at:

results/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

Feature caches belong under:

results/transformer_feature_cache/

Training uses original and colour-augmented features.
Validation uses original features.

The training script can generate missing feature chunks when it
has access to the videos, backbone weights, and DINOv3 source.

Videos, weights, and feature caches require separate access.
Shared locations and teammate permissions still need to be confirmed.

## Training on Wice

Before submitting, check the paths in the Python and Slurm scripts.
Set the working directory, virtual environment, video directory,
log directory, and Slurm account for your own environment.

From the repository root, run:

mkdir -p results/logs
sbatch scripts/train_dinov3_base_384.slurm

The script writes results to:

results/dinov3_base_colour_384_fullval_v1/

Choose a new output directory for another run. The script stops
if the configured output directory already exists.

## Dataset credit

This project uses EPIC-KITCHENS-100.

Dataset:
https://epic-kitchens.github.io/

Annotation source and license:
https://github.com/epic-kitchens/epic-kitchens-100-annotations

The supplied split CSVs select clips for this project.
Retain the original dataset attribution and license.
