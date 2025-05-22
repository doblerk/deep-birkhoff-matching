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