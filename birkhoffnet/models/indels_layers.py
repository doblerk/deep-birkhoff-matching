import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear


class IndelBilinear(nn.Module):
    """
    Bilinear indel cost model: learns insertion/deletion costs based on
    node embeddings and opposite graph embedding.

    Cost ~ softplus(scale *  (nodeᵀ W graph + b))

    Args:
        dim: embedding dimension
        bias: whether to include bias term
        diagonal: if True, use diagonal bilinear form (fewer params)
    """
    def __init__(self, dim, bias=False, diagonal=False):
        super().__init__()
        self.diagonal = diagonal
        
        if diagonal:
            # parametrize W as vector (diagonal)
            self.W = nn.Parameter(torch.randn(dim))
        else:
            self.W = nn.Parameter(torch.randn(dim, dim))
        
        if bias:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('b', None)
        
        # learnable scaling to control magnitude
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, node_emb, graph_emb):
        """
        Computes indel cost.

        Args:
            node_emb: (B, N, d) node embeddings (candidate nodes to be inserted/deleted)
            graph_emb: (B, d) graph-level embeddings of the smaller graphs
        
        Returns:
            costs: (B, N) indel costs per node
        """
        if self.diagonal:
            # (B, N, d) * (d) -> (B, N, d) then dot with graph_emb
            weighted = node_emb * self.W
            # (B, N, d) dot (B, d) -> (B, N)
            costs = torch.einsum('bnd,bd->bn', weighted, graph_emb)
        else:
            # (d, d) matmul each node: use einsum for batch
            weighted_node = torch.einsum('bnd,dk->bnk', node_emb, self.W) # (B, N, d)
            # dot with graph embedding → (B, N)
            costs = torch.einsum('bnk,bd->bn', weighted_node, graph_emb) # (B, N)

        if self.b is not None:
            costs = costs + self.b
        
        # ensute positive costs
        costs = F.softplus(costs * self.scale)
        return costs


class IndelLowRankBilinear(nn.Module):
    def __init__(self, d, r=64):
        super().__init__()
        self.U = Linear(d, r, bias=False) # proj node
        self.V = Linear(d, r, bias=False) # proj graph
        self.out = Linear(r, 1)
    
    def forward(self, node_h, graph_g):
        """
        Computes indel cost.

        Args:
            node_h: (B, N, D) node embeddings of the larger graphs
            graph_g: (B, D) graph-level embeddings of the smaller graphs
        
        Returns:
            cost: (B, N) indel costs for each node w.r.t. the smaller graphs
        """
        node_r = self.U(node_h)
        graph_r = self.V(graph_g).unsqueeze(1)
        prod = node_r * graph_r
        scores = self.out(torch.tanh(prod)).squeeze(-1)
        cost = F.softplus(scores)
        return cost


class IndelCrossAttention(nn.Module):
    """
    Cross-attention + bilinear head for indel costs.
    Node to be inserted asks: `how well would I fit into graph G1?`
    Node to be deleted asks: `how redundant am I in G2?`
    """
    def __init__(self, dim, num_heads=4, bias=False, diagonal=False):
        super().__init__()
        self.fim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # multihead linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        # bilinear head to map (node, context) -> cost
        if diagonal:
            self.W = nn.Parameter(torch.randn(dim))
        else:
            self.W = nn.Parameter(torch.randn(dim, dim))
        if bias:
            self.b = nn.Paramter(torch.zeros(1))
        else:
            self.register_parameter('b', None)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def multihead_attention(self, Q, K, V):
        """
        Args:
            Q: (B, Nq, d)
            K,V: (B, Nk, d)
        
        Returns:
            context: (B, Nq, d)
        """
        B, Nq, d = Q.shape
        _, Nk, _ = K.shape

        # project
        Q = self.q_proj(Q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, Nq, d_h)
        K = self.k_proj(K).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, Nk, d_h)
        V = self.v_proj(V).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, Nk, d_h)

        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / (self.head_dim ** 0.5)   # (B, h, Nq, Nk)

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", attn_weights, V)  # (B, h, Nq, d_h)
        context = context.transpose(1, 2).reshape(B, Nq, d)  # (B, Nq, d)

        return self.out_proj(context)

    def forward(self, node_emb_g1, node_emb_g2):
        """
        Args:
            node_emb_g1: (B, N1, d) candidate nodes for indel.
            node_emb_g2: (B, N2, d) nodes of other graph.
        
        Returns:
            costs: (B, N1)
        """
        # cross-attend each node in node_emb against the other nodes
        context = self.multihead_attention(node_emb_g1, node_emb_g2, node_emb_g2)

        # bilinear scoring between node and context
        if self.W.ndim == 1:  # diagonal
            weighted = node_emb_g1 * self.W
            score = torch.einsum("bnd,bnd->bn", weighted, context)
        else:
            weighted = torch.einsum("bnd,dk->bnk", node_emb_g1, self.W)
            score = torch.einsum("bnk,bnk->bn", weighted, context)

        if self.b is not None:
            score = score + self.b
        
        cost = F.softplus(score * self.scale)
        return cost  # (B, N1)     