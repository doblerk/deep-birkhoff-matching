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
   
    def __init__(
            self, 
            perm_vectors: torch.Tensor, 
            model: nn.Module, 
            temperature: float = 1.0, 
            freeze_epochs: int = 2,
            entropy_weight: float = 0.02
    ):
        """
        Args:
            perm_matrices: tensor of fixed permutation matrices (k, n, n)
            model: alpha generator model that outputs logits (B, k)
        """
        super().__init__()

        self.register_buffer("perm_vectors", perm_vectors.clone())

        self.k, self.n = perm_vectors.shape

        self.temperature = temperature #nn.Parameter(torch.ones(1))
        self.model = model
        
        self.freeze_epochs = freeze_epochs
        self.freeze_timer = 0
        self._frozen = False

        self.entropy_weight = entropy_weight

    # @property
    # def temperature(self):
    #     # ensure temperature > 0
    #     return torch.exp(self.log_temp) + 1e-6

    def get_alpha_weights(self, alpha_logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(alpha_logits / self.temperature, dim=1)
    
    def _set_requires_grad(self, flag: bool):
        for p in self.model.parameters():
            p.requires_grad_(flag)
    
    def freeze_module(self):
        if not self._frozen:
            self._set_requires_grad(False)
            self._frozen = True
            self.freeze_timer = self.freeze_epochs + 1
            print("Freezing: ", self.freeze_timer)
    
    def unfreeze_module(self):
        if self._frozen:
            self._set_requires_grad(True)
            self._frozen = False
    
    def update_freeze_timer(self):
        if not self._frozen:
            return
        print("Updating...")
        self.freeze_timer -= 1
        print(self.freeze_timer)
        if self.freeze_timer <= 0:
            print("Unfreezing...")
            self.unfreeze_module()

    def set_permutations(self, new_perm_vectors: torch.Tensor):
        self.perm_vectors.copy_(new_perm_vectors.to(self.perm_vectors.device))
    
    def get_entropy(self, alphas: torch.Tensor) -> torch.Tensor:
        entropy = -(alphas * alphas.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return entropy
    
    def mse_loss(self, input, target, use_entropy=False, alphas=None, epoch=None, entropy_weight=None):
        loss = F.mse_loss(input, target, reduction="mean")
        if use_entropy and alphas is not None:
            entropy = self.get_entropy(alphas)
            weight = entropy_weight if entropy_weight is not None else self.entropy_weight
            lambda_ent = weight * torch.exp(torch.tensor(-epoch / 50))
            return loss - lambda_ent * entropy
        return loss

    def forward(self, g1: torch.Tensor, g2: torch.Tensor):
        alpha_logits = self.model(g1, g2)
        alphas = self.get_alpha_weights(alpha_logits)

        # ----------------------------
        # Build soft permutation
        # S[b,i,j] = k ∑ ​α[b,k] ⋅ 1[j=permk​[i]]
        # ----------------------------
        
        B = g1.shape[0]

        soft = torch.zeros(
            B, self.n, self.n, 
            device=alphas.device,
            dtype=alphas.dtype,
        )

        # perm = self.perm_vectors.T   # (n,k)

        rows = torch.arange(self.n, device=alphas.device)

        # soft = torch.zeros(
        #     B, self.n, self.n,
        #     device=alphas.device,
        #     dtype=alphas.dtype,
        # )

        # cols = perm.unsqueeze(0).expand(B, -1, -1)       # (B,n,k)
        # weights = alphas.unsqueeze(1).expand(-1, self.n, -1)  # (B,n,k)

        # soft.scatter_add_(2, cols, weights)

        for k_idx in range(self.k):
            cols = self.perm_vectors[k_idx]
            soft[:, rows, cols] += alphas[:, k_idx].unsqueeze(-1)

        # debug
        # matrices = torch.zeros((self.k, 10, 10), device=alphas.device, dtype=torch.float32)
        # matrices.scatter_(2, self.perm_vectors.unsqueeze(-1), 1.0)
        # ref = torch.einsum('bk,kij->bij', alphas, matrices)
        # print(torch.allclose(ref, soft, atol=1e-6))

        return soft, alphas
        # soft_assignments = torch.einsum('bk,kij->bij', alphas, self.perm_matrices)
        # entropy = -(alphas * (alphas + 1e-8).log()).sum(dim=-1).mean()
        # return soft_assignments, alphas, entropy