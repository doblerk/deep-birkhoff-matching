
import torch
import torch.nn as nn

from random import sample
from itertools import permutations


def compute_cost_matrix(representations1, representations2):
    return torch.cdist(representations1, representations2, p=2)


def pad_cost_matrix(cost_matrices):
    B, N, M = cost_matrices.shape
    max_size = max(N, M)
    padded_cost_matrices = torch.ones((B, max_size, max_size), device=cost_matrices.device)
    padded_cost_matrices[:, :N, :M] = cost_matrices
    return padded_cost_matrices


def generate_permutation_matrices(N):
    """
    Generate all possible N x N permutation matrices.
    """
    perms = list(itertools.permutations(range(N)))
    perm_matrices = torch.zeros(len(perms), N, N)

    for i, perm in enumerate(perms):
        for j, p in enumerate(perm):
            perm_matrices[i, j, p] = 1
    return perm_matrices  # Shape: (N!, N, N)


class ContrastiveLoss(nn.Module):

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, cost_matrix, assignment_matrix):
        return torch.einsum('bij,bij->', cost_matrix, assignment_matrix) # element-wise multiplication, followed by a summation


class PermutationMatrix(nn.Module):
   
    def __init__(self):
        super(PermutationMatrix, self).__init__()
    
    def generate_permutation_matrices(self, batch_size, N, k_plus_one):
        """
        Generates permutation matrices based on Carathéodory's Theorem.
        """
        perms = list(permutations(range(N)))
        sampled_perms = sample(perms, min(k_plus_one, len(perms)))
        perm_matrices = torch.zeros(len(sampled_perms), N, N)

        for i, perm in enumerate(sampled_perms):
            for j, p in enumerate(perm):
                perm_matrices[i, j, p] = 1
        
        return perm_matrices # shape: (k+1, N, N)

  

    
  
