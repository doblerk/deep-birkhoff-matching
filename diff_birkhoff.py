
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


class ContrastiveLoss(nn.Module):

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, cost_matrices, assignment_matrices):
        return torch.einsum('bij,bij->', cost_matrices, assignment_matrices) # element-wise multiplication, followed by a summation


class PermutationMatrix(nn.Module):
   
    def __init__(self):
        super(PermutationMatrix, self).__init__()
    
    def generate_permutation_matrices(self, batch_size, N, k_plus_one):
        """
        Generates permutation matrices based on Carathéodory's Theorem.
        """
        perms = list(permutations(range(N)))
        sampled_perm_matrices = torch.zeros((batch_size, k_plus_one, N, N))
        for b in range(batch_size): # iterate over square cost matrices
            sampled_perms = sample(perms, min(k_plus_one, len(perms)))
            for i, perm in enumerate(sampled_perms):
                for j, p in enumerate(perm):
                    sampled_perm_matrices[b, i, j, p] = 1.0
        return sampled_perm_matrices # shape: (B, k+1, N, N)
    
    def forward(self, perm_matrices, temperature=0.1):
        B, k_plus_one, N, N = perm_matrices.shape
        # Learnable alpha weights (B, k+1)
        alphas = torch.Parameter(
            torch.nn.functional.softmax(
                torch.randn(B, k_plus_one, device=perm_matrices.device) / temperature,
                dim=1,
            )
        )
        soft_assignment_matrices = torch.einsum('bk,bkij->bij', alphas, perm_matrices)
        return soft_assignment_matrices



  

    
  
