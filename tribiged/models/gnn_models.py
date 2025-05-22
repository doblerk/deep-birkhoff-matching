import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, Sequential
from torch_geometric.nn import global_add_pool

from tribiged.models.gnn_layers import GINLayer


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
        
        self.dense_layers = Sequential(
            Linear(hidden_dim * n_layers, hidden_dim * n_layers),
            BatchNorm1d(hidden_dim * n_layers),
            ReLU(),
            Dropout(p=0.2),
            Linear(hidden_dim * n_layers, hidden_dim)
        )

        self.input_proj = Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
    
    def freeze_params(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x, edge_index, batch):
        x_residual = self.input_proj(x) if self.input_proj is not None else x

        node_embeddings = []
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = x + x_residual
            x_residual = x
            node_embeddings.append(x)
        
        graph_pooled = []
        for embeddings in node_embeddings:
            pooled = global_add_pool(embeddings, batch)
            graph_pooled.append(pooled)
        
        h = torch.cat(graph_pooled, dim=1)

        graph_embeddings = self.dense_layers(h)

        return node_embeddings[-1], graph_embeddings