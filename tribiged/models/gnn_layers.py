"""
    Graph Isomorphism Network

    @References:
        - "How Powerful are Graph Neural Networks?" paper 
        - https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Sequential
from torch_geometric.nn import GINConv


class GINLayer(torch.nn.Module):
    """
    A class defining the Graph Isomorphism layer

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    """
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        
        self.conv = GINConv(
            Sequential(
                Linear(input_dim,
                       hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            )
        )

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)