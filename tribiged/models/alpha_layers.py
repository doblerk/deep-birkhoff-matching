import torch
import torch.nn as nn
import torch.nn.functional as F

from tribiged.utils.permutation import PermutationPool


class AlphaPermutationLayer(nn.Module):
   
    def __init__(self, perm_matrices: torch.Tensor, k: int, embedding_dim: int):
        super(AlphaPermutationLayer, self).__init__()
        self.perm_matrices = perm_matrices
        self.k = k
        self.temperature = 1.0
        
        dim = embedding_dim * 2
        self.alpha_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.4),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, self.k)
        )

    def get_alpha_weights(self):
        return torch.softmax(self.alpha_logits / self.temperature, dim=1)
        
    def forward(self, graph_repr_b1, graph_repr_b2):
        pair_repr = torch.cat([graph_repr_b1, graph_repr_b2], dim=1)
        alpha_logits = self.alpha_mlp(pair_repr)
        alphas = F.softmax(alpha_logits / self.temperature, dim=1)
        soft_assignments = torch.einsum('bk,kij->bij', alphas, self.perm_matrices)
        return soft_assignments, alphas