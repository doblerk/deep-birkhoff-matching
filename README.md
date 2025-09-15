# GNN-GED

## Description
The goal is to research a novel GNN-based framework for approximating *Graph Edit Distance* (GED) computation in a fully differentiable manner. Rather than enforcing permutation-like matrices onto a continuous matrix via entropic regularization or iterative normalization, we start with valid permutations and learn how to weigh them meaningfully. This yields a convex combination of interpretable assignments, which resides within a subspace of the *Birkhoff polytope*.

## Roadmap

### Milestones
#### 1. Two-stage framework for (1) learning discriminative node embeddings, and (2) learning convex combinations of permutation matrices.
> - [x] Triplet Loss + Regression Loss.
> - [x] GNN encoder + MLP.
> - [x] Learnable scaling factor.

### Next Goals
#### Explore adaptive permutation pool refinement.
> - [ ] Prune underused permutation matrices + genetic algorithms.
#### Address the handling of unequally size graphs.
> - [ ] Integrate learnable insertion and deletion.
#### Explore another type of edit-based formulation.
> - [ ] Extend the framework to a self-supervised learning approach.