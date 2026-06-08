import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch


class CostMatrixBuilder(nn.Module):
    def __init__(self, embedding_dim, max_graph_size, use_learned_sub=False):
        super().__init__()
        self.d = embedding_dim
        self.max_graph_size = max_graph_size
        self.use_learned_sub = use_learned_sub

        if use_learned_sub:
            # single bilinear matrix W for substitution (learned similarity)
            # Use a factorization of W to ensure it is symmetric and positive semidefinite
            r = embedding_dim
            self.L = nn.Parameter(torch.randn(self.d, r))
            self.sub_bias = nn.Parameter(torch.tensor(1.0))
        else:
            self.sub_L = None

        self.ins_mlp = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.GELU(),
            nn.LayerNorm(self.d),
            nn.Linear(self.d, 1),
        )
    
    def to_dense_node_embeddings(self, node_repr, batch_vec):
        """
        Converts tensor of node embeddings to batched graphs.

        Args:
            node_repr: (N, d)
            batch_vec: (N,)
        
        Returns:
            dense_repr: (B, N_max, d)
            mask: (B, N_max)
            counts: (B,)
        """
        dense_repr, mask = to_dense_batch(
            x=node_repr, 
            batch=batch_vec, 
            fill_value=0.0,
            max_num_nodes=self.max_graph_size
        )
        counts = mask.sum(dim=1)
        return dense_repr, mask, counts
    
    def substitution_cost(self, H1, H2, mask1, mask2):
        """
        Computes pairwise substitution costs between node embeddings.

        Args:
            H1: (B, N1, d)
            H2: (B, N2, d)
        
        Returns:
            C: (B, N, N)
        """
        mask1 = mask1.unsqueeze(2) # (B, N1, 1)
        mask2 = mask2.unsqueeze(1) # (B, 1, N2)
        mask = mask1 & mask2 # (B, N1, N2)

        if not self.use_learned_sub:
            # compute batched p-norm (default Euclidean)
            C = torch.cdist(H1, H2, p=2)
        else:
            H1 = F.normalize(H1, dim=-1)
            H2 = F.normalize(H2, dim=-1)
            # build W = L @ L^T (guaranteed to be PSD)
            W = self.L @ self.L.T
            W = W / (W.norm() + 1e-8)
            # weighted h1 = H1 @ W
            weighted_h1 = torch.einsum('bnd,dk->bnk', H1, W)
            s = torch.einsum('bnk,bmk->bnm', weighted_h1, H2)
            C = F.softplus(-s + self.sub_bias)

        C = C.masked_fill(~mask, 0.0)
        return C
    
    # def forward(self, node_repr_b1, graph_emb_b1, batch1, node_repr_b2, graph_emb_b2, batch2):
    def forward(self, H1, mask1, g1_emb, H2, mask2):
        """
        Builds the complete cost matrix.
            - Rectangular matrix if substitution costs only.
            - Square matrix if optional indel costs.

        Args:
            node_repr_b1: (N1_total, d)
            node_repr_b2: (N2_total, d)
            batch1: (N1_total,)
            batch2: (N2_total,)
        
        Returns:
            C_padded: (B, N_max, N_max) cost matrices
            mask1, mask2: (B, N_max) validity masks
        """

        _, N_max, _ = H1.shape

        subs = self.substitution_cost(H1, H2, mask1, mask2)
        C = subs.clone()

        updated_mask1 = mask1.clone()

        # counts1 = mask1.sum(dim=1)
        # counts2 = mask2.sum(dim=1)

        # g1_expand = g1_emb.unsqueeze(1).expand(-1, N_max, -1)

        # ins_input = torch.cat([H2, g1_expand], dim=-1)

        # eps = F.softplus(
        #     self.ins_mlp(ins_input)
        # ).squeeze(-1)

        # row_idx = torch.arange(N_max, device=C.device).unsqueeze(0)  # (1, N)

        # eps_row_mask_per_graph = (
        #     (row_idx >= counts1.unsqueeze(1)) &
        #     (row_idx < counts2.unsqueeze(1))
        # )

        # eps_row_mask = eps_row_mask_per_graph.unsqueeze(2) & mask2.unsqueeze(1)

        # eps_values = eps.unsqueeze(1).expand(-1, N_max, -1)

        # C = torch.where(eps_row_mask, eps_values, C)

        # updated_mask1 = mask1 | eps_row_mask_per_graph

        return C, updated_mask1, mask2