import torch
import torch.nn.functional as F

from torch.nn import Linear


class AttentionPooling(torch.nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.proj = Linear(d, hidden)
        self.attn = Linear(hidden, 1)
    
    def forward(self, Z):
        scores = self.attn(torch.tanh(self.proj(Z))).squeeze(-1) # (B, N)
        alphas = F.softmax(scores, dim=-1) # (B, N)
        h = torch.sum(alphas.unsqueeze(-1) * Z, dim=1) # (B, d)
        return h, alphas