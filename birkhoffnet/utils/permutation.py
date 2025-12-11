import numpy as np

import torch

from typing import Tuple


class PermutationPool:
    def __init__(self, max_n: int, k: int, seed: int = 42):
        """
        Args:
            max_n (int): Maximum graph size (i.e., full matrix size: max_n x max_n)
            k (int): Number of permutation matrices to generate
            seed (int): RNG seed for reproducibility
        """
        self.rng = torch.Generator().manual_seed(seed) #np.random.default_rng(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_n = max_n
        self.k = k
        self.perm_vectors = self._generate_permutation_vectors()

    def _generate_permutation_vectors(self) -> torch.Tensor:
        """Generate permutation vectors of max_n length, padded/embedded."""
        perms = torch.zeros((self.k, self.max_n), dtype=torch.long)
        perms[0] = torch.arange(0, self.max_n, dtype=torch.long)
        for i in range(1, self.k):
            perms[i] = torch.randperm(self.max_n, generator=self.rng)
        perms = perms.to(self.device)
        return perms
    
    def get_vectors(self) -> torch.Tensor:
        return self.perm_vectors
    
    def _resolve(self, value: int, seg_from: torch.Tensor, seg_to: torch.Tensor) -> int:
        """Recursively resolve conflicts for a single value."""
        if value in seg_to:
            pos = (seg_to == value).nonzero(as_tuple=True)[0].item()
            return self._resolve(seg_from[pos], seg_from, seg_to)
        else:
            return value
    
    def _partially_mapped_crossover(self, p1: torch.Tensor, p2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        size = len(p1)
        c1, c2 = p1.clone(), p2.clone()

        # Random crossover break points
        bp1 = torch.randint(0, size - 1, (1,)).item()
        bp2 = torch.randint(bp1 + 1, size, (1,)).item()

        # Crossover segments
        seg1 = p1[bp1:bp2 + 1]
        seg2 = p2[bp1:bp2 + 1]

        # Exchange segments
        c1[bp1:bp2 + 1] = seg2
        c2[bp1:bp2 + 1] = seg1

        # Fill up mapping outside the crossover region
        for i in list(range(0, bp1)) + list(range(bp2 + 1, size)):
            c1[i] = self._resolve(c1[i].item(), seg1, seg2)
            c2[i] = self._resolve(c2[i].item(), seg2, seg1)

        return c1, c2
    
    def mate_permutations(self, sorted_idx: torch.Tensor, k: int = 2) -> None:
        """
        Replaces the k worst perms with offspring produced by the k best perms.
        """
        best_idx = sorted_idx[-k:]
        worst_idx = sorted_idx[:k]

        # For each worst individual, generate a new child from the best parents
        for wi in worst_idx:
            # Randomly choose 2 distinct parents from top-k
            parent_indices = torch.randperm(k)[:2]
            p1 = self.perm_vectors[best_idx[parent_indices[0]]]
            p2 = self.perm_vectors[best_idx[parent_indices[1]]]

            # Perform mating
            c1, c2 = self._partially_mapped_crossover(p1, p2)

            # Randomly choose one child
            child = c1 if torch.rand(1).item() < 0.5 else c2
            self.perm_vectors[wi] = child
    
    def get_matrix_batch(self):
        """
        Returns a batch of k permutation matrices of shape (k, max_n, max_n)
        One-hot encoded, inactive rows filled with zeros
        """
        matrices = torch.zeros((self.k, self.max_n, self.max_n), device=self.device, dtype=torch.float32)
        matrices.scatter_(2, self.perm_vectors.unsqueeze(-1), 1.0)
        return matrices
    

    # def mate_permutations(self, sorted_idx: torch.Tensor, k: int = 2) -> None:
    # """
    # Replace all permutations except the first (elite) one
    # using crossover between two random parents from the top-k best.
    # """
    # elite_idx = sorted_idx[0]          # keep this one unchanged
    # best_idx = sorted_idx[:k]          # top-k pool for parent selection
    # replace_idx = sorted_idx[1:]       # everyone except the first

    # for wi in replace_idx:
    #     # Randomly choose 2 distinct parents from top-k
    #     parent_indices = torch.randperm(k)[:2]
    #     p1 = self.perm_vectors[best_idx[parent_indices[0]]]
    #     p2 = self.perm_vectors[best_idx[parent_indices[1]]]

    #     # Perform crossover
    #     c1, c2 = self._partially_mapped_crossover(p1, p2)

    #     # Randomly pick one child
    #     child = c1 if torch.rand(1).item() < 0.5 else c2
    #     print(f'Changing perm: {self.perm_vectors[wi]} with parent indices {parent_indices}')
    #     self.perm_vectors[wi] = child
    #     print(f'Changed to: {self.perm_vectors[wi]}')

    # # ✅ Preserve elite (the first permutation)
    # self.perm_vectors[elite_idx] = self.perm_vectors[elite_idx]