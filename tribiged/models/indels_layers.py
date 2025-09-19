import torch
import torch.nn.functional as F

from torch.nn import Linear


class BilinearIndel(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = Linear(d, d, bias=False)
        self.out = Linear(d, 1)
    
    def forward(self, node_h, graph_g):
        """
        Computes indel cost.

        Args:
            node_h: (B, N, D) node embeddings of the larger graphs
            graph_g: (B, D) graph-level embeddings of the smaller graphs
        
        Returns:
            cost: (B, N) indel costs for each node w.r.t. the smaller graphs
        """
        g_proj = self.W(graph_g)
        scores = torch.einsum("bnd,bd->bn", node_h, g_proj)
        scores = self.out(torch.tanh(scores.unsqueeze(-1))).squeeze(-1) # optional proj
        cost = F.softplus(scores)
        return cost


class LowRankBilinearIndel(torch.nn.Nodule):
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
