import math
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
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim * 2, input_dim * 4),
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(input_dim * 4),
        #     nn.Dropout(0.2),
            
        #     nn.Linear(input_dim * 4, input_dim * 4),
        #     nn.GELU(),
        #     nn.LayerNorm(input_dim * 4),
        #     nn.Dropout(0.2),

        #     nn.Linear(input_dim * 4, k)
        # )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 4, input_dim * 8),
            nn.ReLU(inplace=True),
            nn.LayerNorm(input_dim * 8),
            nn.Dropout(0.4),
            
            nn.Linear(input_dim * 8, input_dim * 8),
            nn.GELU(),
            nn.LayerNorm(input_dim * 8),
            nn.Dropout(0.4),

            nn.Linear(input_dim * 8, k)
        )
    
    def forward(self, g1, g2):
        # pair_repr = torch.cat([g1, g2], dim=-1)
        pair_repr = torch.cat([
            g1,
            g2,
            torch.abs(g1 - g2),
            g1 * g2
        ], dim=-1)
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
            n_epochs: int,
            entropy_weight: float = 0.01,
            k_min: float = 5.0,
            k_floor_weight: float = 0.1,
            min_temp: float = 0.5,
            max_temp: float = 5.0,
            mixing_eps: float = 0.02
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
        # self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.temperature = 1.0
        self.min_temp = min_temp
        self.max_temp = max_temp

        # Regularization
        self.entropy_weight = entropy_weight
        self.k_floor_weight = k_floor_weight
        self.k_min = int(self.k * 0.1) + 1 #k_min

        # Target entropy (normalized)
        self.target_entropy = math.log(self.k_min) / math.log(self.k)

        # Mixing safeguard
        self.mixing_eps = mixing_eps

        # Entropy annealing
        self.n_epochs = n_epochs
    
    # --------------------------------------------------
    # Temperature
    # --------------------------------------------------
    def get_temperature(self, epoch: int):
        progress = min(epoch / self.n_epochs, 1.0)
        return self.max_temp - (self.max_temp - self.min_temp) * progress
        # return self.min_temp + (self.max_temp - self.min_temp) * torch.sigmoid(self.log_temperature) # shifted sigmoid to max_temp

    # --------------------------------------------------
    # Alpha weights
    # --------------------------------------------------
    def get_alpha_weights(self, logits: torch.Tensor) -> torch.Tensor:
        # return F.softmax(logits / self.temperature, dim=1)

        # logits = logits - logits.mean(dim=1, keepdim=True)
        # logits = logits / (logits.std(dim=1, keepdim=True) + 1e-6)
        
        # T = self.get_temperature(epoch)
        
        alphas = F.softmax(logits / self.temperature, dim=1)
        
        if self.mixing_eps > 0:
            alphas = (1 - self.mixing_eps) * alphas + self.mixing_eps / self.k
        
        return alphas

    # --------------------------------------------------
    # Entropy
    # --------------------------------------------------
    def get_entropy(self, alphas: torch.Tensor) -> torch.Tensor:
        entropy = -(alphas * alphas.clamp_min(1e-8).log()).sum(dim=-1)
        return entropy.mean() / math.log(self.k)

    # def get_kl_to_uniform(self, logits: torch.Tensor) -> torch.Tensor:
    #     alphas_raw = F.softmax(logits, dim=1)
    #     uniform = torch.full_like(alphas_raw, 1.0 / self.k)
    #     kl = (alphas_raw * (alphas_raw / uniform).clamp_min(1e-8).log()).sum(dim=-1)
    #     return kl.mean()
    
    def effective_k(self, alphas: torch.Tensor) -> torch.Tensor:
        entropy = self.get_entropy(alphas)
        return torch.exp(entropy * math.log(self.k))
    
    # --------------------------------------------------
    # Loss
    # --------------------------------------------------
    def mse_loss(self, pred, target, use_entropy=False, alphas=None, epoch=None):
        
        mse = F.mse_loss(pred, target, reduction="mean")
        loss = mse
        
        if alphas is not None:
            entropy = self.get_entropy(alphas)
            eff_k = self.effective_k(alphas)

            # entropy_loss = (entropy - self.target_entropy) ** 2
            entropy_loss = F.relu(self.target_entropy - entropy)

            k_floor_loss = F.relu(self.k_min - eff_k)

            loss = loss \
                + self.entropy_weight * entropy_loss \
                + self.k_floor_weight * k_floor_loss

            if epoch % 50 == 0:
                print(
                    f"[Epoch {epoch}] "
                    f"MSE: {mse.item():.4f} | "
                    f"H: {entropy.item():.3f} | "
                    f"target_H: {self.target_entropy:.3f} | "
                    f"eff_k: {eff_k.item():.2f} | "
                    f"k_min: {self.k_min} | "
                    # f"T: {self.get_temperature(epoch):.2f}"
                )

        #     # progress = epoch / self.n_epochs
        #     # lambda_entropy = self.entropy_weight * (1 - progress)

        #     # loss = loss + self.entropy_weight * (entropy_loss + k_floor_loss)

        #     entropy_loss = ((entropy - self.target_entropy) ** 2) / (self.target_entropy + 1e-6)

        #     # optional annealing
        #     progress = epoch / self.n_epochs if epoch is not None else 0.0
        #     lambda_entropy = self.entropy_weight * (1 - progress)

        #     loss = loss + lambda_entropy * entropy_loss #* mse.detach()
        
        # if epoch is not None and epoch % 50 == 0:
        #     print(
        #         f"[Epoch {epoch}] "
        #         f"MSE: {mse.item():.4f} | "
        #         f"H: {entropy.item():.3f} | "
        #         f"eff_k: {eff_k.item():.1f} | "
        #         f"T: {self.get_temperature().item():.2f}"
        #     )
        
        # return loss

        # loss = F.mse_loss(pred, target, reduction="mean")
        
        # if use_entropy and alphas is not None:

        #     entropy = self.get_entropy(alphas)

        #     scaled_entropy = self.entropy_weight * entropy #* loss.detach()

        #     # # progress = min(epoch / self.n_epochs, 1.0)
        #     # # lambda_ent = self.start + (self.end - self.start) * progress
        #     if epoch % 50 == 0:
        #         # print(f"Epoch: {epoch + 1}: {loss.item():.4f} - {lambda_ent:.4f} x {entropy.item():.4f} -> fraction of total loss {(lambda_ent * entropy) / loss * 100:.4f}%")
        #         # print(f"Entropy: {entropy}")
        #         # print(f"Entropy to MSE ratio: {scaled_entropy / loss * 100}%")
        #         # print(f"Epoch: {epoch + 1}: {loss.item():.4f} - {self.entropy_weight:.4f} x {entropy.item():.4f} x {loss.item():.4f} -> fraction of total loss {scaled_entropy:.4f} / {loss:.4f} * 100 = {scaled_entropy / loss * 100:.4f}%")
        #         print(f"Epoch: {epoch + 1}: {loss.item():.4f} - {self.entropy_weight:.4f} x {entropy.item():.4f} -> fraction of total loss {scaled_entropy:.4f} / {loss:.4f} * 100 = {scaled_entropy / loss * 100:.4f}%")

        #     # return loss - lambda_ent * entropy
        #     return loss - scaled_entropy
        
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