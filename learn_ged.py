import torch
import numpy as np
import torch.nn as nn

from itertools import permutations

from torch.utils.data import Dataset
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset


https://arxiv.org/pdf/2304.02458
https://www.pragmatic.ml/sparse-sinkhorn-attention/


def calc_cost_matrix(emb1, emb2, max_size):
  n1, n2 = emb1.size(0), emb2.size(0)
  cost_matrix = torch.cdist(emb1, emb2, p=2)
  square_cost_matrix = torch.ones((max_size, max_size), device=cost_matrix.device)
  square_cost_matrix[:n1, :n2] = cost_matrix
  return square_cost_matrix

def process_batch(node_emb1, mask1, node_emb2, mask2):
  batch_size = node_emb1.shape[0]

  # determine max graph size in the batch
  max_nodes1 = mask1.sum(dim=1).max().item()  # Max valid nodes in batch1
  max_nodes2 = mask2.sum(dim=1).max().item()  # Max valid nodes in batch2
  max_size = max(max_nodes1, max_nodes2)  # Uniform padding size

  # initialize a tensor to hold all cost matrices
  cost_matrices = torch.zeros((batch_size, max_size, max_size), device=node_emb1.device)

  # Compute cost matrices for each graph pair
  for i in range(batch_size):
      emb1 = node_emb1[i][mask1[i]]  # Extract valid nodes for graph i in batch1
      emb2 = node_emb2[i][mask2[i]]  # Extract valid nodes for graph i in batch2
      cost_matrices[i] = calc_cost_matrix(emb1, emb2, max_size)  # Pad to max_size

  return cost_matrices  # Shape: [batch_size, max_size, max_size]


class PermutationMatrix(nn.Module):

  def __init__(self, num_nodes):
    super(PermutationMatrix, self).__init__()

  def generate_permutation_matrices(self, num_nodes, k, batch_size):
    permutation_matrices = []
    while len(permutation_matrices) < k:
      m = np.eye(num_nodes)
      np.random.shuffle(m)
      if not any(np.array_equal(m, existing) for existing in permutation_matrices):
        permutation_matrices.append(torch.from_numpy(m).float())
    permutation_matrices = torch.stack(permutation_matrices)
    expanded_permutation_matrices = permutation_matrices.unsqueeze(0)
    permutation_matrices_batch = expanded_permutation_matrices.repeat(batch_size, 1, 1, 1) # could generate new random PMs
    return permutation_matrices_batch

    # return torch.stack(permutation_matrices)

  def forward(self, num_nodes, batch_size):
    k = num_nodes + 1  # Carathéodory's theorem
    alphas = nn.Parameter(torch.softmax(torch.randn(batch_size, k), dim=1), requires_grad=True)
    permutation_matrices = self.generate_permutation_matrices(num_nodes, k, batch_size)
    assignment_matrices = torch.einsum('bk,bkij->bij', alphas, permutation_matrices)
    return assignment_matrices








