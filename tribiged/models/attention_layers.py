import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=1, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        self.proj = nn.Linear(input_dim, hidden_dim)

        self.attn_heads = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_heads)
        ])
    
    def forward(self, Z):
        """
        Z: (B, N, d)
        """
        H = self.proj(Z) # (B, N, hidden_dim)
        pooled_outputs = []

        for k in range(self.num_heads):
            scores = torch.matmul(H, self.attn_heads[k]) # (B, N)
            weights = F.softmax(scores, dim=1) # (B, N)

            h = torch.sum(weights.unsqueeze(1) * Z, dim=1)
            pooled_outputs.append(h)

        if self.concat:
            return torch.cat(pooled_outputs, dim=-1) # (B, num_heads * d)
        else:
            return torch.mean(torch.stack(pooled_outputs, dim=0), dim=0) # (B, d)