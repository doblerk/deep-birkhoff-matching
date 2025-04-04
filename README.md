# Learning GED via Differentiable Permutation Matrix via Birkhoff Polytope Approximation

#### Questions and Remarks
- Should we implement in the beginning a pre-generated set of permutation matrices?
    - If yes, then we should get the size of the biggest graph(s) so that we can pad all cost matrices if necessary.
    - If no, then we should generate permutation matrices on the fly.
- For a cost matrix of size NxN, there are N! permutation matrices, but the size of the Birkhoff decomposition is guranteed to be at most N².
    - Moreover, Carathéodory's Theorem states that any point in the convex hull (i.e., any DSM in the Birkhoff polytope) can be expressed as a convex combination of at most k + 1 extreme points (i.e., permutation matrices), where k is the size of the cost matrix. This implies we can compute a subset of k + 1 permutation matrices instead of computing N! permutation matrices. Since we only need k + 1 permutation matrices per graph pair, we can:
        - Randomly sample k + 1 permutation matrices from the full set.
        - Deterministically sample k + 1 permutation matrices from the hull set.
    - We currently determine the size of the largest graph(s) in a batch and pad all cost matrices accordingly so that they are all square and of the same size.
        - Could we generate permutation matrices on the fly instead?

#### TODOs
- [x] Implement a function to store ground truth labels as a dictionary -> constant lookup.
- [x] Implement a function to compute cost matrices.
- [ ] Implement two different strategies to efficiently sample permutation matrices.
- [ ] Extend the framework to a self-supervised learning approach. 

#### Some References

- [ref](https://arxiv.org/pdf/2304.02458)
- [ref](https://www.pragmatic.ml/sparse-sinkhorn-attention/)