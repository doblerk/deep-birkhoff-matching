import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch


class CostMatrixBuilder(nn.Module):
    def __init__(self, embedding_dim, max_graph_size, use_learned_sub=False, model_indel=None, rank=None):
        super().__init__()
        self.d = embedding_dim
        self.max_graph_size = max_graph_size
        self.use_learned_sub = use_learned_sub
        self.model_indel = model_indel

        if use_learned_sub:
            # single bilinear matrix W for substitution (learned similarity)
            # Use a factorization of W to ensure it is symmetric and positive semidefinite
            r = rank or embedding_dim
            self.L = nn.Parameter(torch.randn(self.d, r))
            # self.sub_W = nn.Parameter(torch.randn(self.d, self.d))
            self.sub_bias = nn.Parameter(torch.tensor(1.0))
        else:
            self.sub_L = None
        
        self.eps_rows = nn.Parameter(torch.ones(max_graph_size))
    
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
            # build W = L @ L^T (guaranteed to be PSD)
            W = self.L @ self.L.T
            # weighted h1 = H1 @ W
            weighted_h1 = torch.einsum('bnd,dk->bnk', H1, W)
            s = torch.einsum('bnk,bmk->bnm', weighted_h1, H2)
            s = (s - s.mean()) / (s.std() + 1e-8)      
            # convert similarity to positive cost (higher similarity -> lower cost)
            C = F.softplus(-s + self.sub_bias) + 1e-8 # (B, N, N)
        
        C = C.masked_fill(~mask, 0.0)
        return C
    
    # def forward(self, node_repr_b1, graph_emb_b1, batch1, node_repr_b2, graph_emb_b2, batch2):
    def forward(self, H1, mask1, H2, mask2):
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
        # Convert variable-size graphs to dense padded tensors
        # H1, mask1, counts1 = self.to_dense_node_embeddings(node_repr_b1, batch1)
        # H2, mask2, counts2 = self.to_dense_node_embeddings(node_repr_b2, batch2)

        B, N_max, d = H1.shape

        # Compute substitution cost
        subs = self.substitution_cost(H1, H2, mask1, mask2)
        C = subs.clone()

        updated_mask1 = mask1.clone()

        # Compute indel costs
        # if self.model_indel is not None:
        #     _, N, _ = subs.shape
        #     # insertion: which nodes in G2 (columns) are likely to be inserted into G1
        #     c_ins = self.model_indel(H2, graph_emb_b1)

        #     # deletion: which nodes in G1 (rows) are likely to be deleted
        #     c_del = self.model_indel(H1, graph_emb_b2)

        #     # broadcast these indels
        #     c_ins_mat = c_ins.unsqueeze(1).expand(-1, N, -1) # (B, N, N)
        #     c_del_mat = c_del.unsqueeze(2).expand(-1, -1, N) # (B, N, N)

        #     # mask real nodes
        #     mask1_bool = mask1.bool()
        #     mask2_bool = mask2.bool()

        #     # Dummy-row mask: row is padded (not real) & col is real -> this cell should be insertion cost for that column
        #     dummy_row_mask = (~mask1_bool).unsqueeze(2) & mask2_bool.unsqueeze(1)   # (B, N, N)
        #     # Dummy-col mask: row is real & col is padded (not real) -> deletion cost for that row
        #     dummy_col_mask = mask1_bool.unsqueeze(2) & (~mask2_bool).unsqueeze(1)   # (B, N, N)
        #     # Dummy-dummy mask: both padded -> set zero
        #     dummy_dummy_mask = (~mask1_bool).unsqueeze(2) & (~mask2_bool).unsqueeze(1)

        #     C = subs.clone()
        #     # Where dummy_row_mask is True -> use c_ins_mat
        #     C = torch.where(dummy_row_mask, c_ins_mat, C)
        #     # Where dummy_col_mask is True -> use c_del_mat
        #     C = torch.where(dummy_col_mask, c_del_mat, C)
        #     # Where both dummy -> zero
        #     C = torch.where(dummy_dummy_mask, torch.zeros_like(C), C)
        # else:
        #     C = subs

        counts1 = mask1.sum(dim=1)
        counts2 = mask2.sum(dim=1)

        # Add learnable epsilon rows for insertion
        # eps_mat = torch.zeros_like(C)

        valid_mask = mask1.unsqueeze(2) & mask2.unsqueeze(1)
        mean_sub_cost = (subs * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        # scale learnable eps with batch statistics
        eps_rows_scaled = F.softplus(self.eps_rows) * mean_sub_cost.detach()

        row_idx = torch.arange(N_max, device=C.device).unsqueeze(0)  # (1, N)

        eps_row_mask_per_graph = (
            (row_idx >= counts1.unsqueeze(1)) &
            (row_idx < counts2.unsqueeze(1))
        )

        eps_row_mask = eps_row_mask_per_graph.unsqueeze(2) & mask2.unsqueeze(1)

        eps_values = eps_rows_scaled.unsqueeze(0).unsqueeze(2)
        eps_values = eps_values.expand(B, -1, N_max)

        C = torch.where(eps_row_mask, eps_values, C)

        updated_mask1 = mask1 | eps_row_mask_per_graph

        return C, updated_mask1, mask2

        # for b in range(B):
        #     n1 = counts1[b].item()
        #     n2 = counts2[b].item()
        #     if n2 > n1:
        #         # Rows n1..n2-1 are epsilon rows for insertion
        #         eps_row_idx = torch.arange(n1, n2, device=C.device)
        #         # Broadcast scalar eps cost for each column in G2
        #         eps_mat[b, eps_row_idx, :n2] = eps_rows_scaled[eps_row_idx].unsqueeze(1)
        #         # Update mask1 to incorporate epsilon rows
        #         updated_mask1[b, n1:n2] = 1
        
        # # Apply epsilon rows only where the row is not real
        # eps_row_mask = (~mask1).unsqueeze(2) & mask2.unsqueeze(1)
        # C = torch.where(eps_row_mask, eps_mat, C)

        # # Padding outside G2 nodes stays zero
        # dummy_col_mask = mask2.unsqueeze(1) == 0
        # C = torch.where(dummy_col_mask, torch.zeros_like(C), C)

        # # Dummy-dummy (padding rows and columns) = 0
        # dummy_dummy_mask = (~mask1).unsqueeze(2) & (~mask2).unsqueeze(1)
        # C = torch.where(dummy_dummy_mask, torch.zeros_like(C), C)

        # return C, mask1, mask2
        # return C, updated_mask1, mask2