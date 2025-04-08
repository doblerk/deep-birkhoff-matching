
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # return torch.einsum('bij,bij->', cost_matrices, assignment_matrices) # element-wise multiplication, followed by a summation
        return torch.sum(cost_matrices * assignment_matrices, dim=(1, 2)) # (B,)


class TripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        return F.relu(pos_dist - neg_dist + self.margin).mean()


class PermutationMatrix(nn.Module):
   
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



  

    
  
