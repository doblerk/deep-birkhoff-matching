"""
    Graph Isomorphism Network

    @References:
        - "How Powerful are Graph Neural Networks?" paper 
        - https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, Sequential
from torch_geometric.nn import GINConv, global_add_pool


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


class Model(torch.nn.Module):
    """
    A class defining the Graph Isomorphism Network

    Attributes
    ----------
    input_dim: int
        number of features per node
    hidden_dim: int
        number of channels
    n_layers: int
        number of hidden layers
    """
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()

        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(GINLayer(input_dim, hidden_dim))
                                
        for _ in range(n_layers - 1):
            self.conv_layers.append(GINLayer(hidden_dim, hidden_dim))

        self.input_proj = Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
    
    def freeze_params(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x, edge_index, batch):
        x_residual = self.input_proj(x) if self.input_proj is not None else x
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = x + x_residual
            x_residual = x
        return x, global_add_pool(x, batch)