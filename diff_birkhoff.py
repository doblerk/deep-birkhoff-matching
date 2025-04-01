
import torch
import torch.nn as nn


def compute_cost_matrix(representations1, representations2):
  cost_matrix = torch.cdist(representations1, representations2, p=2)
  # should we make it square? padding?
  return cost_matrix

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
