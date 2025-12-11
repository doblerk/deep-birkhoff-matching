import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch


class DensePooling(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_dim * n_layers, hidden_dim * n_layers),
            nn.BatchNorm1d(hidden_dim * n_layers),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim * n_layers, hidden_dim)
        )
    
    def forward(self, Z):
        return self.dense_layers(Z)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        self.proj = nn.Linear(input_dim, hidden_dim)

        self.attn_heads = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_heads)
        ])
    
    def forward(self, Z, batch):
        """
        Z: (N, d)
        batch: (N,)
        """
        Z_dense, mask = to_dense_batch(Z, batch) # (B, N_max, d), (B, N_max)
        H = self.proj(Z_dense) # (B, N_max, hidden_dim)
        pooled_outputs = []
        
        for k in range(self.num_heads):
            scores = (H * self.attn_heads[k]).sum(dim=-1)

            scores = scores.masked_fill(~mask, float('-inf'))
            weights = F.softmax(scores, dim=1)
            
            h = torch.bmm(weights.unsqueeze(1), Z_dense).squeeze(1)
            pooled_outputs.append(h)

        if self.concat:
            return torch.cat(pooled_outputs, dim=-1) # (B, num_heads * d)
        else:
            return torch.mean(torch.stack(pooled_outputs, dim=0), dim=0) # (B, d)