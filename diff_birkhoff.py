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
        self.n = n
        self.k = k
        self.perm_vectors = self._generate_permutation_vectors(n, k)
    
    def _generate_permutation_vectors(self, n, k):
        perms = []
        for i in range(int(k)):
            perms.append(tuple(np.random.permutation(n)))
        return torch.tensor(perms, dtype=torch.long)
    
    def get_vectors(self):
        return self.perm_vectors

    def get_matrix_batch(self):
        return torch.nn.functional.one_hot(self.perm_vectors, num_classes=self.n).float()


class AlphaPermutationLayer(nn.Module):
   
    def __init__(self, perm_pool: PermutationPool):
        super(AlphaPermutationLayer, self).__init__()
        self.perm_pool = perm_pool
        self.alpha_weights = nn.Parameter(torch.randn(perm_pool.k))
        
    def forward(self, temperature=0.1):
        perms = self.perm_pool.get_matrix_batch().to(self.alpha_weights.device)
        alphas = torch.softmax(self.alpha_weights / temperature, dim=0)
        return torch.einsum('k,kij->ij', alphas, perms)