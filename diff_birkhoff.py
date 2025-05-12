import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from random import sample

from scipy.stats import gaussian_kde


class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        return F.relu(pos_dist - neg_dist + self.margin).mean()


class SoftGEDLoss(nn.Module):

    def __init__(self):
        super(SoftGEDLoss, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, cost_matrices, assignment_matrices):
        return torch.sum(cost_matrices * assignment_matrices, dim=(1, 2)) * self.scale # (B,)


class PermutationPool:
    def __init__(self, max_n, k, size_data, seed: int = 42):
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
        self.size_data = size_data
        self.kde = gaussian_kde(size_data.T)
        self.perm_vectors = self._generate_permutation_vectors()
    
    def _sample_size(self):
        """Sample a (n, m) pair from the empirical KDE distribution."""
        while True:
            sample = np.round(self.kde.resample(1, seed=self.rng)).astype(int).flatten()
            n, m = sample
            if 1 <= n <= self.max_n and 1 <= m <= self.max_n:
                return n, m

    def _generate_permutation_vectors(self):
        """Generate permutation vectors of max_n length, padded/embedded."""
        perms = []

        identity = np.arange(self.max_n)
        perms.append(tuple(identity))

        for _ in range(self.k - 1):
            # n, m = self._sample_size()
            # perm_len = min(n, m)
            # perm = self.rng.permutation(perm_len)
            # vector = -1 * np.ones(self.max_n, dtype=int)
            # vector[:perm_len] = perm
            # perms.append(tuple(vector))
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


class AlphaPermutationLayer(nn.Module):
   
    def __init__(self, perm_pool: PermutationPool, perm_matrices: PermutationPool, embedding_dim: int, max_batch_size: int):
        super(AlphaPermutationLayer, self).__init__()
        self.perm_pool = perm_pool
        self.k = perm_pool.k
        self.temperature = 0.8
        self.perms = perm_matrices

        # self.alpha_logits = nn.Parameter(torch.randn(max_batch_size, perm_pool.k), requires_grad=True)

        dim = 128
        self.alpha_mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2 * embedding_dim, dim * 2),
            nn.ReLU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, self.k)
        )

    def get_alpha_weights(self):
        return torch.softmax(self.alpha_logits / self.temperature, dim=1)

        
    def forward(self, graph_repr_b1, graph_repr_b2):
        # B = graph_repr_b1.size(0)
        # alpha_logits = self.alpha_logits[:B]  # [B, k]
        # alphas = F.softmax(alpha_logits / self.temperature, dim=1)
        pair_repr = torch.cat([graph_repr_b1, graph_repr_b2], dim=1) # (B, 2D)
        alpha_logits = self.alpha_mlp(pair_repr) # (B, k)
        alphas = F.softmax(alpha_logits / self.temperature, dim=1) # (B, k)
        soft_assignments = torch.einsum('bk,kij->bij', alphas, self.perms)
        return soft_assignments, alphas


class LearnablePaddingAttention(nn.Module):

    def __init__(self, max_graph_size):
        super(LearnablePaddingAttention, self).__init__()
        self.max_graph_size = max_graph_size
        self.attention_logits = nn.Parameter(torch.randn(max_graph_size, max_graph_size))
    
    def forward(self, cost_matrices):
        attention_weights = torch.sigmoid(self.attention_logits)
        attention_weights = attention_weights.unsqueeze(0).to(cost_matrices.device)
        # weighted_cost = cost_matrices * attention_weights
        return cost_matrices * attention_weights