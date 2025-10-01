import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool

from tribiged.models.gnn_layers import GINLayer
from tribiged.models.pooling_layers import AttentionPooling, DensePooling


class Model(nn.Module):
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
    def __init__(self, input_dim, hidden_dim, n_layers, use_attention=False, attn_concat=True, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.attn_concat = attn_concat
        self.num_heads = num_heads

        self.conv_layers = torch.nn.ModuleList()

        self.conv_layers.append(GINLayer(input_dim, hidden_dim))
                                
        for _ in range(n_layers - 1):
            self.conv_layers.append(GINLayer(hidden_dim, hidden_dim))

        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

        if self.use_attention:
            self.pooling = AttentionPooling(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                concat=attn_concat
            )
        else:
            self.pooling = DensePooling(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            )
    
    @property
    def output_dim(self):
        if self.attn_concat:
            return self.hidden_dim * self.num_heads
        else:
            return self.hidden_dim
    
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
        
        if self.use_attention:
            Z = node_embeddings[-1]
            graph_embeddings = self.pooling(Z, batch)
        else:
            graph_pooled = [global_add_pool(h, batch) for h in node_embeddings]
            h = torch.cat(graph_pooled, dim=1)
            graph_embeddings = self.pooling(h)

        return node_embeddings[-1], graph_embeddings