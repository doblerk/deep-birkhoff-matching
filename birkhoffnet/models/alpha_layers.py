import math
import logging
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
        #     nn.Linear(input_dim * 2, input_dim * 2),
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(input_dim * 2),
        #     nn.Dropout(0.2),
            
        #     nn.Linear(input_dim * 2, input_dim * 2),
        #     nn.GELU(),
        #     nn.LayerNorm(input_dim * 2),
        #     nn.Dropout(0.2),

        #     nn.Linear(input_dim * 2, k)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(4 * input_dim, 2 * input_dim),
            nn.GELU(),
            nn.LayerNorm(2 * input_dim),
            nn.Dropout(0.2),
            nn.Linear(2 * input_dim, k)
        )
    
    def forward(self, g1, g2):
        # pair_repr = torch.cat([g1, g2], dim=-1)
        # return self.mlp(pair_repr)
        pair_repr = torch.cat([
            g1,
            g2,
            torch.abs(g1 - g2),
            g1 * g2
        ], dim=-1)
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
            min_temp: float = 0.5,
            max_temp: float = 3.0,
            entropy_weight: float = 0.02
    ):
        """
        Args:
            perm_matrices: tensor of fixed permutation matrices (k, n, n)
            model: alpha generator model that outputs logits (B, k)
        """
        super().__init__()

        # Permutations
        self.register_buffer("perm_vectors", perm_vectors.clone())
        self.k, self.n = perm_vectors.shape

        # Alpha generator
        self.model = model

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.min_temp = min_temp
        self.max_temp = max_temp

        # Entropy weight
        self.entropy_weight = entropy_weight
    
    # --------------------------------------------------
    # Temperature
    # --------------------------------------------------
    def get_temperature(self):
        return self.min_temp + (self.max_temp - self.min_temp) * torch.sigmoid(self.log_temperature) # shifted sigmoid to max_temp

    # --------------------------------------------------
    # Alpha weights
    # --------------------------------------------------
    def get_alpha_weights(self, logits: torch.Tensor) -> torch.Tensor:

        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits / (logits.std(dim=1, keepdim=True) + 1e-8)
        
        T = self.get_temperature()
        
        alphas = F.softmax(logits / T, dim=1)
        
        return alphas

    # --------------------------------------------------
    # Regularization
    # --------------------------------------------------
    def get_entropy(self, alphas: torch.Tensor) -> torch.Tensor:
        entropy = -(alphas * alphas.clamp_min(1e-8).log()).sum(dim=1)
        return entropy.mean() / math.log(self.k)

    def get_kl_to_uniform(self, logits: torch.Tensor) -> torch.Tensor:
        alphas_raw = F.softmax(logits, dim=1)
        uniform = torch.full_like(alphas_raw, 1.0 / self.k)
        kl = (alphas_raw * (alphas_raw / uniform).clamp_min(1e-8).log()).sum(dim=-1)
        return kl.mean()
    
    def effective_k(self, alphas: torch.Tensor) -> torch.Tensor:
        entropy = self.get_entropy(alphas)
        return torch.exp(entropy * math.log(self.k))
    
    # --------------------------------------------------
    # Loss
    # --------------------------------------------------
    def loss_fn(self, pred, target, use_entropy=False, alphas=None, epoch=None):
        
        mae = F.l1_loss(pred, target, reduction="mean")
        mse = F.mse_loss(pred, target, reduction="mean")

        loss = mse

        entropy = None
        penalty = None
        
        if use_entropy and alphas is not None:

            entropy = self.get_entropy(alphas)

            penalty = self.entropy_weight * loss.detach() * entropy

            loss = loss - penalty

        # if alphas is not None:

        #     entropy_str = (
        #         f"{entropy.item():.3f}" if entropy is not None else "N/A"
        #     )

        #     penalty_str = (
        #         f"{penalty.item():.4f}" if penalty is not None else "0.0000"
        #     )


        #     penalty_ratio = (
        #         (penalty / (mse + 1e-8)).item() if penalty is not None else 0.0
        #     )

        #     logging.info(
        #         f"[Epoch {epoch}] "
        #         f"Train MAE: {mae.item():.4f} | "
        #         f"Train MSE: {mse.item():.4f} | "
        #         f"Entropy: {entropy_str} | "
        #         f"T: {self.get_temperature():.2f} | "
        #         f"Eff_k: {self.effective_k(alphas):.2f} | "
        #         f"Penalty: {penalty_str} | "
        #         f"Penalty/Loss: {penalty_ratio:.4f}"
        #     )
        
        return loss
    
    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def _set_requires_grad(self, flag: bool):
        for p in self.model.parameters():
            p.requires_grad_(flag)

    def set_permutations(self, new_perm_vectors: torch.Tensor):
        self.perm_vectors.copy_(new_perm_vectors.to(self.perm_vectors.device))

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, g1: torch.Tensor, g2: torch.Tensor):
        logits = self.model(g1, g2)
        alphas = self.get_alpha_weights(logits)

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

        rows = torch.arange(self.n, device=alphas.device)

        for k_idx in range(self.k):
            cols = self.perm_vectors[k_idx]
            soft[:, rows, cols] += alphas[:, k_idx].unsqueeze(-1)

        return soft, alphas