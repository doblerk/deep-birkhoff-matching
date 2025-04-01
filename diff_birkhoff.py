
import torch
import torch.nn as nn






class ContrastiveLoss(nn.Module):

  def __init__(self):
    super(ContrastiveLoss, self).__init__()

  def forward(self, cost_matrix, assignment_matrix):
    return torch.einsum('bij,bij->', cost_matrix, assignment_matrix) # element-wise multiplication, followed by a summation
