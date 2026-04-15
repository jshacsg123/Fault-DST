# fault_dst

A semi-supervised 3D fault detection framework with a dual-student design.  
The model combines a CNN branch and a 3D Transformer branch, and introduces an EMA teacher for stable consistency learning.

## Overview

The framework contains three components:

- **Student1**: CNN-based branch
- **Student2**: 3D Transformer-based branch
- **Teacher**: EMA version of Student2

The training process is divided into two stages:

1. **Synthetic pretraining**  
   Both branches are first trained on synthetic data with full supervision.

2. **Joint training**  
   Synthetic data, sparsely labeled field data, and unlabeled field data are used together for semi-supervised learning.

## Losses

The training objective includes:

- **Supervised loss** for labeled samples
- **Hard pseudo-label loss** from Student2 to Student1
- **Soft consistency loss** between the two branches
- **Classifier consistency regularization**
- **Teacher-student consistency loss**

## Data format

### Synthetic data

Each `.npz` file should contain:

- `seis`: seismic volume
- `fault`: fault label

### Field sparse data

Each `.npz` file should contain:

- `seis`: seismic volume
- `fault`: sparse fault label
- optional `mask`: valid labeled region

If `mask` is not provided, ignored regions can be inferred from the label.

### Field unlabeled data

Each `.npz` file should contain:

- `seis`: seismic volume

Other arrays will be ignored during unlabeled training.

## Training

The code supports:

- supervised pretraining on synthetic data
- semi-supervised joint training on synthetic and field data
- mixed precision training
- configurable optimization and scheduling strategies

A typical training command looks like this:

```bash
python train.py \
  --syn_train_dir data/synthetic/train \
  --syn_val_dir data/synthetic/val \
  --field_train_dir data/field/train \
  --field_unlabeled_dir data/field/unlabeled \
  --amp

  ## Inference

The code supports inference on a single seismic volume or multiple datasets.

During inference, the model loads a trained checkpoint and predicts fault probabilities in a sliding-window manner, which allows large 3D volumes to be processed patch by patch. The output can be saved as prediction files and optionally exported as 2D slices or 3D visualization results.

A typical inference command looks like this:

```bash
python infer.py \
  --input data/F3.npy \
  --checkpoint checkpoints \
  --infer_size 128 \
  --output_root output/infer