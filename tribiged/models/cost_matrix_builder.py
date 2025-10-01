import torch
import torch.nn as nn
import torch.nn.functional as F


class CostMatrixBuilder(nn.Module):
    def __init__(self, embedding_dim, use_learned_sub=False, model_indel=None, diag_indel=False):
        super().__init__()
        self.d = embedding_dim
        self.use_learned_sub = use_learned_sub

        if use_learned_sub:
            # single bilinear matrix for substitution (learned similarity)
            self.sub_W = nn.Parameter(torch.randn(self.d, self.d))
            self.sub_bias = nn.Parameter(torch.zeros(1))
        else:
            self.sub_W = None
        
        self.model_indel = model_indel
    
    def substitution_cost(self, H1, H2):
        """
        Computes pairwise substitution costs between node embeddings.

        Args:
            H1: (B, N1, d)
            H2: (B, N2, d)
        
        Returns:
            subs: (B, N, N)
        """

        B, N, d = H1.shape
        if self.use_learned_sub:
            # bilinear similarity -> produce a cost, we want higher dissimilarity => maybe negative similarity
            # compute s_ij = h1_i^T W h2_j  => then transform to positive cost via softplus on -s
            # weighted H1: (B,N,d) @ (d,d) -> (B,N,d)
            weighted_h1 = torch.einsum('bnd,dk->bnk', H1, self.sub_W)
            # s_ij = (weighted_h1) dot h2_j  -> (B,N,N)
            s = torch.einsum('bnk,bmk->bnm', weighted_h1, H2)
            # convert similarity to cost (higher similarity -> lower cost)
            # e.g. cost = softplus(-s + bias)
            cost = F.softplus(-s + self.sub_bias)
            return cost  # (B,N,N)
        else:
            # Euclidean squared distance, vectorized:
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            h1_sq = (H1 ** 2).sum(dim=-1, keepdim=True) # (B,N,1)
            h2_sq = (H2 ** 2).sum(dim=-1, keepdim=True).transpose(1, 2) # (B,1,N)
            dot = torch.einsum('bnd,bmd->bnm', H1, H2)
            cost = h1_sq + h2_sq - 2.0 * dot
            # numerical safety
            cost = torch.clamp(cost, min=0.0)
            return cost
    
    def forward(self, H1, H2, graph_emb1, graph_emb2, mask1, mask2):
        """
        Builds the complete cost matrix.

        Args:
            H1: (B, N_max, d) node embeddings for graph1 (padded)
            H2: (B, N_max, d) node embeddings for graph2 (padded)
            graph_emb1: (B, d) graph-level embeddings for graph1 (usually smaller)
            graph_emb2: (B, d) graph-level embeddings for graph2
            mask1: (B, N_max) boolean mask True for real nodes in graph1
            mask2: (B, N_max) boolean mask True for real nodes in graph2
        
        Returns:
            C: (B, N_max, N_max) cost matrices with substitutions + indel costs
        """
        B, N, d = H1.shape

        # basic subs block for all rows/cols
        subs = self.substitution_cost(H1, H2)
        C = subs.clone()

        # compute indel costs (for nodes that are real in the other graph)
        # insertion: which nodes in G2 (columns) are likely to be inserted into G1
        c_ins = self.model_indel(H2, graph_emb1)

        # deletion: which nodes in G1 (rows) are likely to be deleted
        c_del = self.model_indel(H1, graph_emb2)

        # broadcast these indels
        c_ins_mat = c_ins.unsqueeze(1).expand(-1, N, -1) # (B, N, N)
        c_del_mat = c_del.unsqueeze(2).expand(-1, -1, N) # (B, N, N)

        # mask real nodes
        mask1_bool = mask1.bool()
        mask2_bool = mask2.bool()

        # Dummy-row mask: row is padded (not real) & col is real -> this cell should be insertion cost for that column
        dummy_row_mask = (~mask1_bool).unsqueeze(2) & mask2_bool.unsqueeze(1)   # (B, N, N)
        # Dummy-col mask: row is real & col is padded (not real) -> deletion cost for that row
        dummy_col_mask = mask1_bool.unsqueeze(2) & (~mask2_bool).unsqueeze(1)   # (B, N, N)
        # Dummy-dummy mask: both padded -> set zero
        dummy_dummy_mask = (~mask1_bool).unsqueeze(2) & (~mask2_bool).unsqueeze(1)

        # Apply: replace subs values by indel costs where appropriate
        # Where dummy_row_mask is True -> use c_ins_mat
        C = torch.where(dummy_row_mask, c_ins_mat, C)
        # Where dummy_col_mask is True -> use c_del_mat
        C = torch.where(dummy_col_mask, c_del_mat, C)
        # Where both dummy -> zero
        C = torch.where(dummy_dummy_mask, torch.zeros_like(C), C)

        return C  # (B, N, N)