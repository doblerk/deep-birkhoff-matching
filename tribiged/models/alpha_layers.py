import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from tribiged.utils.permutation import PermutationPool


class AlphaPermutationLayer(nn.Module):
   
    def __init__(self, perm_matrices: torch.Tensor, k: int, embedding_dim: int, window_history: int = 20):
        super(AlphaPermutationLayer, self).__init__()
        self.perm_matrices = perm_matrices
        self.k = k
        self.temperature = 1.0
        
        dim = embedding_dim * 2
        self.alpha_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.4),
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, self.k)
        )

        self.window_history = window_history
        self.register_buffer("alpha_history", torch.zeros(self.k))

    def get_alpha_weights(self, alpha_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(alpha_logits / self.temperature, dim=1)
    
    def update_alpha_history(self, alphas: torch.Tensor):
        self.alpha_history += alphas.detach()
    
    def clear_alpha_history(self):
        self.alpha_history.zero_()
    
    def rank_alpha(self, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_weights = self.alpha_history / self.window_history
        sorted_idx = torch.argsort(mean_weights) # ascending
        worst_idx = sorted_idx[:k]
        best_idx = sorted_idx[-k:]
        return worst_idx, best_idx
        
    def forward(self, graph_repr_b1: torch.Tensor, graph_repr_b2: torch.Tensor):
        pair_repr = torch.cat([graph_repr_b1, graph_repr_b2], dim=1)
        alpha_logits = self.alpha_mlp(pair_repr)
        alphas = F.softmax(alpha_logits / self.temperature, dim=1)
        soft_assignments = torch.einsum('bk,kij->bij', alphas, self.perm_matrices)
        return soft_assignments, alphas