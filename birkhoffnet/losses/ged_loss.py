import torch
import torch.nn as nn
import torch.nn.functional as F


class GEDLoss(nn.Module):

    def __init__(self):
        super(GEDLoss, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, cost_matrices, assignment_matrices):
        return torch.sum(cost_matrices * assignment_matrices, dim=(1, 2)) * self.scale