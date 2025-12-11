# Deep Birkhoff Matching Framework

## Description
`birkhoffnet` is the official Python implementation of the Deep Birkhoff Matching framework. The goal is to research a novel GNN-based framework for approximating *Graph Edit Distance* (GED) computation in a fully differentiable manner. Rather than enforcing permutation-like matrices onto a continuous matrix via entropic regularization or iterative normalization, we start with valid permutations and learn how to weigh them meaningfully. This yields a convex combination of interpretable assignments, which resides within a subspace of the *Birkhoff polytope*.

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