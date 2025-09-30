import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Alpha Generator Models
# --------------------------

class AlphaMLP(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.ReLU(inplace=True),
            nn.LayerNorm(input_dim * 4),
            nn.Dropout(0.4),
            nn.Linear(input_dim * 4, input_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * 4, k)
        )
    
    def forward(self, g1, g2):
        pair_repr = torch.cat([g1, g2], dim=-1)
        return self.mlp(pair_repr)


class AlphaBilinear(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        # One bilinear weight matrix per permutation
        self.bilinear = nn.ModuleList([
            nn.Bilinear(input_dim, input_dim, 1, bias=False)
            for _ in range(k)
        ])
    
    def forward(self, g1, g2):
        scores = [b(g1, g2) for b in self.bilinear]
        return torch.cat(scores, dim=-1)


class AlphaCrossAttention(nn.Module):
    def __init__(self, input_dim, k, num_heads=4, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, k)
        )
    
    def forward(self, g1, g2):
        g1 = g1.unsqueeze(1)
        g2 = g2.unsqueeze(1)

        # Cross-attention: let g1 qury g2
        attn_out, _ = self.attn(query=g1, key=g2, value=g2)

        # Pool across sequence dimension
        pooled = attn_out.mean(dim=1)

        # Project to logits
        return self.fc(pooled)


# --------------------------
# General Permutation Layer
# --------------------------

class AlphaPermutationLayer(nn.Module):
   
    def __init__(self, perm_matrices: torch.Tensor, model: nn.Module, temperature: float = 1.0):
        """
        Args:
            perm_matrices: tensor of fixed permutation matrices (k, n, n)
            model: alpha generator model that outputs logits (B, k)
        """
        super().__init__()
        self.perm_matrices = perm_matrices
        self.k = perm_matrices.size(0)
        self.temperature = temperature
        self.model = model
        self.freeze_counter = 10

    def get_alpha_weights(self, alpha_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(alpha_logits / self.temperature, dim=1)
    
    def freeze_module(self):
        for p in self.alpha_mlp.parameters():
            p.requires_grad = False
    
    def unfreeze_module(self):
        for p in self.alpha_mlp.parameters():
            p.requires_grad = True
    
    def update_counter(self):
        self.freeze_counter -= 1
    
    def clear_counter(self):
        self.freeze_counter = 10
    
    def get_freeze_counter(self):
        return self.freeze_counter
    
    def entropy_loss(self, alphas: torch.Tensor) -> torch.Tensor:
        entropy = - (alphas * torch.log(alphas + 1e-8)).sum(dim=1)
        return entropy.mean()

    def forward(self, g1: torch.Tensor, g2: torch.Tensor):
        alpha_logits = self.model(g1, g2)
        alphas = self.get_alpha_weights(alpha_logits)
        soft_assignments = torch.einsum('bk,kij->bij', alphas, self.perm_matrices)
        return soft_assignments, alphas