import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Alpha Generator Models
# --------------------------

class AlphaMLP(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(input_dim * 2, input_dim * 4),
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(input_dim * 4),
        #     nn.Dropout(0.4),
        #     nn.Linear(input_dim * 4, input_dim * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(input_dim * 4, k)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.ReLU(inplace=True),
            nn.LayerNorm(input_dim * 4),
            nn.Dropout(0.2),
            
            nn.Linear(input_dim * 4, input_dim * 4),
            nn.GELU(),
            nn.LayerNorm(input_dim * 4),
            nn.Dropout(0.2),

            nn.Linear(input_dim * 4, k)
        )
    
    def forward(self, g1, g2):
        pair_repr = torch.cat([g1, g2], dim=-1)
        # pair_repr = torch.abs(g1 - g2)
        return self.mlp(pair_repr)


class AlphaBilinear(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        # One bilinear weight matrix per permutation
        self.bilinear = nn.Parameter(torch.randn(k, input_dim, input_dim))
    
    def forward(self, g1, g2):
        scores = []
        for i in range(self.bilinear.shape[0]):
            W = self.bilinear[i]
            score = torch.sum((g1 @ W) * g2, dim=-1, keepdim=True)
            scores.append(score)
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

        # self.fc = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, k)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.ReLU(inplace=True),
            nn.LayerNorm(input_dim * 4),
            nn.Dropout(0.2),
            
            nn.Linear(input_dim * 4, input_dim * 4),
            nn.GELU(),
            nn.LayerNorm(input_dim * 4),
            nn.Dropout(0.2),

            nn.Linear(input_dim * 4, k)
        )
    
    def forward(self, g1, g2):
        g1 = g1.unsqueeze(1)
        g2 = g2.unsqueeze(1)

        # Cross-attention: let g1 query g2
        attn_out, _ = self.attn(query=g1, key=g2, value=g2)
        print(attn_out[0])
        attn_out = attn_out.squeeze(1)

        # Pool across sequence dimension
        # pooled = attn_out.mean(dim=1)
        
        # Concatenate g1 and attended g2
        combined = torch.cat([g1.squeeze(1), attn_out], dim=-1)

        # Project to logits
        return self.mlp(combined)


# --------------------------
# General Permutation Layer
# --------------------------

class AlphaPermutationLayer(nn.Module):
   
    def __init__(self, perm_matrices: torch.Tensor, model: nn.Module, temperature: float = 1.0, freeze_epochs: int = 2):
        """
        Args:
            perm_matrices: tensor of fixed permutation matrices (k, n, n)
            model: alpha generator model that outputs logits (B, k)
        """
        super().__init__()
        self.perm_matrices = perm_matrices
        self.k = perm_matrices.size(0)
        self.temperature = temperature #nn.Parameter(torch.ones(1))
        self.model = model
        self.freeze_epochs = freeze_epochs
        self.freeze_timer = 0
        self._frozen = False

    # @property
    # def temperature(self):
    #     # ensure temperature > 0
    #     return torch.exp(self.log_temp) + 1e-6

    def get_alpha_weights(self, alpha_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(alpha_logits / self.temperature, dim=1)
    
    def freeze_module(self):
        for p in self.model.parameters():
            p.requires_grad = False
        self._frozen = True
    
    def unfreeze_module(self):
        for p in self.model.parameters():
            p.requires_grad = True
        self._frozen = False
    
    def start_freeze_timer(self):
        self.freeze_timer = self.freeze_epochs
    
    def update_freeze_timer(self):
        if self.freeze_timer > 0:
            self.freeze_timer -= 1
    
    def reset_freeze_timer(self):
        self.freeze_counter = self.freeze_epochs
    
    def is_frozen(self):
        return self._frozen
    
    def entropy_loss(self, alphas: torch.Tensor) -> torch.Tensor:
        entropy = - (alphas * torch.log(alphas + 1e-8)).sum(dim=1)
        return entropy.mean()

    def forward(self, g1: torch.Tensor, g2: torch.Tensor):
        alpha_logits = self.model(g1, g2)
        alphas = self.get_alpha_weights(alpha_logits)
        soft_assignments = torch.einsum('bk,kij->bij', alphas, self.perm_matrices)
        return soft_assignments, alphas