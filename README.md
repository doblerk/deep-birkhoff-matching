# Deep Birkhoff Matching Framework

## Overview
`birkhoffnet` is the official Python implementation of the Deep Birkhoff Matching framework. The goal is to research a novel GNN-based framework for approximating *Graph Edit Distance* (GED) computation in a fully differentiable manner. 

Instead of projecting continuous matrices onto the permutation space via entropic regularization or iterative normalization (e.g., Sinkhorn), our method:

 - starts from valid permutation matrices,
 - learns weights over permutations,
 - and produces a convex combination of interpretable assignments.

The resulting matrix lies within a subspace of the *Birkhoff polytope*, enabling structured and interpretable matching between graphs.

## Installation

#### Prerequisites
 - Python >= 3.10
 - PyTorch
 - PyTorch Geometric

#### Setup
```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate environment
source venv/bin/activate

# 3. Install PyTorch (example for CUDA 12.4)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 4. Install PyTorch Geometric
pip install torch_geometric

# 5. Install birkhoffnet in editable mode
pip install -e .
```
Adjust the PyTorch installation depending on your CUDA version.

## Datasets
Graph datasets are automatically downloaded via PyTorch Geometric's `TUDataset` loader on first use.

To avoid redistributing datasets, this repository does not include the raw graph data. Instead, we provide precomputed metadata and GED ground truth required for experiments.

Directory structure:
```bash
data/
├── datasets/        # ignored completely
│   ├── AIDS/
│   │   ├── raw/
│   │   └── processed/
│   └── ...
│
└── metadata/        # tracked by git
    ├── AIDS/
    │   ├── metadata.json
    │   ├── AIDS_pairs.npy
    │   └── AIDS_ged_matrices.pt
    └── ...
```

These files contain:

 - dataset subset definition
 - train / validation / test splits
 - graph pair indices
 - precomputed Graph Edit Distances (GED)

The raw dataset will be downloaded automatically when the dataset is first loaded.

## Configuration
Training and evaluation are configured using a JSON parameter file.

Example configuration:
```json
{
    "dataset": "AIDS",
    "device": "cuda",

    "model": {
        "embedding_dim": 64,
        "k": 21,
        "num_layers": 2,
        "perm_strategy": "random"
    },

    "encoder":{
        "mode": "train",
        "checkpoint": "ckpt_encoder.pth"
    },

    "permutation_evolution": {
        "evolve": false,
        "evolve_every": 10,
        "num_replace": 2,
        "freeze_after_evolve": false
    },

    "training": {
        "epochs_triplet": 101,
        "epochs_siamese": 101,
        "lr": 1e-3,
        "weight_decay": 1e-8,
        "triplet_margin": 0.2,
        "use_entropy": true,
        "entropy_weight": 0.05
    },

    "alpha_tracker": {
        "warmup": 100,
        "window": 10,
        "ema_decay": 0.0
    },

    "dataset_dir": "./data/",
    "output_dir": "./res/"
}
```

## Creating Dataset Metadata
Metadata and graph pair lists can be generated using:
```bash
python3 script/create_datasets.py \
        --dataset_dir data/ \
        --dataset_name AIDS \
        --output_dir data/AIDS
```

To restrict experiments to a subset of graphs (default):
```bash
python3 script/create_datasets.py \
        --dataset_dir data/ \
        --dataset_name AIDS \
        --output_dir data/AIDS \
        --use_subset
```

## Precomputed Graph Edit Distances
Graph edit distances were computed offline on an HPC cluster using job arrays. 

To make experiments reproducible, we provide the merged results:
```bash
data/processed/<dataset>/ged_matrices.pt
```

This file contains:

 - full GED matrix
 - normalized similarity matrix
 - normalization factors
 - node counts
 - valid graph indices

You do not need to recompute GED.

If you wish to regenerate the data or compute GED for new subsets, see:
```bash
scripts/
hpc/
```
## Training
Run training using:
```bash
python3 scripts/run_training.py \
        --params params.json \
        --metadata data/AIDS/AIDS_metadata.json \
        --ged_data data/AIDS/AIDS_ged_matrices.pt
```

## Inference
To predict all pairwise graph distances:
```bash
python3 scripts/run_inference.py \
        --params params.json \
        --metadata data/AIDS/AIDS_metadata.json \
        --ged_data data/AIDS/AIDS_ged_matrices.pt
```

## Project structure
```bash
deep-birkhoff-matching/
├── scripts/           # training and preprocessing scripts
├── hpc/               # scripts used for large-scale GED computation
├── birkhoffnet/               # core library code
├── data/              # metadata and precomputed GED matrices
├── params.json        # experiment configuration
└── README.md
```

## Roadmap

### Milestones
#### 1. Two-stage framework for (1) learning discriminative node embeddings, and (2) learning convex combinations of permutation matrices.
> - [x] Triplet Loss + Regression Loss.
> - [x] GNN encoder + MLP.
> - [x] Learnable scaling factor.

### Next Goals
#### Explore adaptive permutation pool refinement.
> - [x] Prune underused permutation matrices.
> - [x] Apply a genetic-like algorithm to generate new permutation matrices.
#### Address the handling of unequally size graphs.
> - [ ] Integrate learnable insertion and deletion.
#### Explore another type of edit-based formulation.
> - [ ] Extend the framework to a self-supervised learning approach.

## Citation

If you use **Deep Birkhoff Matching** in your work, please cite our paper:

```bibtex
@inproceedings{DBLP:conf/acpr/DoblerR25,
  author       = {Kalvin Dobler and
                  Kaspar Riesen},
  editor       = {Christian Wallraven and
                  Ran He and
                  Brian C. Lovell and
                  Prithwi Chakraborty},
  title        = {Approximating Graph Edit Distance via Differentiable Birkhoff Decompositions},
  booktitle    = {Pattern Recognition and Computer Vision - 8th Asian Conference on
                  Pattern Recognition, {ACPR} 2025, Gold Coast, QLD, Australia, November
                  10-13, 2025, Proceedings, Part {II}},
  series       = {Lecture Notes in Computer Science},
  volume       = {16175},
  pages        = {32--47},
  publisher    = {Springer},
  year         = {2025},
  url          = {https://doi.org/10.1007/978-981-95-4398-4\_3},
  doi          = {10.1007/978-981-95-4398-4\_3},
  timestamp    = {Sun, 07 Dec 2025 22:09:20 +0100},
  biburl       = {https://dblp.org/rec/conf/acpr/DoblerR25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```