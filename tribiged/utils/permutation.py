import numpy as np

import torch


class PermutationPool:
    def __init__(self, max_n: int, k: int, seed: int = 42):
        """
        Args:
            max_n (int): Maximum graph size (i.e., full matrix size: max_n x max_n)
            k (int): Number of permutation matrices to generate
            size_data (np.ndarray): Array of shape (N, 2) containing historical (n, m) size pairs
            seed (int): RNG seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.max_n = max_n
        self.k = k
        self.perm_vectors = self._generate_permutation_vectors()

    def _generate_permutation_vectors(self):
        """Generate permutation vectors of max_n length, padded/embedded."""
        perms = []

        identity = np.arange(self.max_n)
        perms.append(tuple(identity))

        for _ in range(self.k - 1):
            perms.append(tuple(self.rng.permutation(self.max_n)))

        return torch.tensor(perms, dtype=torch.long)
    
    def get_vectors(self):
        return self.perm_vectors
    
    def _mate_permutations(self, alpha_weights):
        ranking = torch.argsort(alpha_weights)
        topk = alpha_weights[ranking][:2]
        pass

    def _partially_mapped_crossover(self, p1, p2):
        size = len(p1)
        c1, c2 = p1.clone(), p2.clone()

        # Random crossover break points
        bp1 = torch.randint(0, size - 1, (1,)).item()
        bp2 = torch.randint(bp1 + 1, size, (1,)).item()

        # Crossover segments
        seg1 = p1[bp1:bp2]
        seg2 = p2[bp1:bp2]

        # Fill up mapping
        self._fill_child(c1, seg1, seg2, size)
        self._fill_child(c2, seg2, seg1, size)

        # Exchange segments
        c1[bp1:bp2 + 1] = seg2
        c2[bp1:bp2 + 1] = seg1

        return c1, c2
    
    def _resolve(self, value, seg_from, seg_to):
        """Recursively resolve conflicts for a single value."""
        if value in seg_to:
            pos = (seg_to == value).nonzero(as_tuple=True)[0].item()
            return self._resolve(seg_from[pos], seg_from, seg_to)
        else:
            return value

    def _fill_child(self, c, seg_from, seg_to, size, idx=0):
        if idx >= size:
            return c
        # If current gene is in the conflicting segment, resolve recursively
        c[idx] = self._resolve(c[idx], seg_from, seg_to)
        return self._fill_child(c, seg_from, seg_to, size, idx + 1)
    
    def get_matrix_batch(self):
        """
        Returns a batch of k permutation matrices of shape (k, max_n, max_n)
        One-hot encoded, inactive rows filled with zeros
        """
        matrices = torch.zeros((self.k, self.max_n, self.max_n))
        for idx, vec in enumerate(self.perm_vectors):
            for i, j in enumerate(vec):
                if j != -1:
                    matrices[idx, i, j] = 1.0
        return matrices