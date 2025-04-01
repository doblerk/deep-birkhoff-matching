
import torch
import torch.nn as nn


def compute_cost_matrix(representations1, representations2):
  cost_matrix = torch.cdist(representations1, representations2, p=2)
  # should we make it square? padding?
  return cost_matrix



class ContrastiveLoss(nn.Module):

  def __init__(self):
    super(ContrastiveLoss, self).__init__()

  def forward(self, cost_matrix, assignment_matrix):
    return torch.einsum('bij,bij->', cost_matrix, assignment_matrix) # element-wise multiplication, followed by a summation
