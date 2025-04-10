import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from random import sample
from itertools import permutations


class SoftGEDLoss(nn.Module):

    def __init__(self):
        super(SoftGEDLoss, self).__init__()
    
    def forward(self, cost_matrices, assignment_matrices):
        # return torch.einsum('bij,bij->', cost_matrices, assignment_matrices) # element-wise multiplication, followed by a summation
        return torch.sum(cost_matrices * assignment_matrices, dim=(1, 2)) # (B,)


class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        return F.relu(pos_dist - neg_dist + self.margin).mean()


class PermutationPool:

    def __init__(self, n, k, seed: int = 42):
        np.random.seed(seed)
        self.perm_vectors = self._generate_permutation_vectors(n, k)
        self.n = n
        self.k = k
    
    def _generate_permutation_vectors(self, n, k):
        perms = []
        for i in range(k):
            perms.append(tuple(np.random.permutation(n)))
        return torch.tensor(perms, dtype=torch.int8)
    
    def get_vectors(self):
        return self.perm_vectors

    def get_matrix_batch(self):
        return torch.nn.functional.one_hot(self.perm_vectors, num_classes=self.n)


class AlphaPermutationLayer(nn.Module):
   
    def __init__(self, batch_size, k_plus_one):
        super(PermutationMatrix, self).__init__()
        self.alpha_weights = nn.Parameter(torch.randn(batch_size, k_plus_one))
    
    def generate_permutation_matrices(self, batch_size, N, k_plus_one):
        """
        Generates permutation matrices based on Carathéodory's Theorem.
        """
        perm_matrices = torch.zeros((batch_size, k_plus_one, N, N))
        for b in range(batch_size): # iterate over square cost matrices
            for i in range(k_plus_one):
                perm = torch.randperm(N)
                perm_matrices[b, i, torch.arange(N), perm] = 1.0
        return perm_matrices # shape: (B, k+1, N, N)
    
    def forward(self, perm_matrices, temperature=0.1):
        # Learnable alpha weights (B, k+1)
        alphas = torch.nn.functional.softmax(
            self.alpha_weights / temperature, 
            dim=-1
        ).to(perm_matrices.device)
        return torch.einsum('bk,bkij->bij', alphas, perm_matrices)