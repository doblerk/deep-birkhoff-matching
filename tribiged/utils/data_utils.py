import os

import numpy as np

import networkx as nx

import torch
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from itertools import combinations

from scipy.stats import spearmanr, kendalltau

from torch_geometric.data import Batch
from torch.utils.data import random_split, ConcatDataset
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset, GEDDataset

from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold




def compute_cost_matrices(node_repr_b1, counts1, node_repr_b2, counts2):
    # Split node embeddings by graph
    b1_node_splits = torch.split(node_repr_b1, counts1.tolist())
    b2_node_splits = torch.split(node_repr_b2, counts2.tolist())
    
    distance_matrices = [
        torch.cdist(b1, b2, p=2)
        for b1, b2 in zip(b1_node_splits, b2_node_splits)
    ]

    return distance_matrices


def pad_cost_matrices(cost_matrices, max_graph_size, pad_value=0.0):
    device = cost_matrices[0].device
    B = len(cost_matrices)
    padded_batch = torch.full(
        (B, max_graph_size, max_graph_size),
        pad_value,
        device=device,
        dtype=cost_matrices[0].dtype
    )

    for i, cost in enumerate(cost_matrices):
        n1, n2 = cost.shape
        padded_batch[i, :n1, :n2] = cost

    return padded_batch


def get_node_masks(batch, max_size, counts):
    """
    batch: torch_geometric.data.Batch object
    returns: [B, max_size] tensor, 1 for real nodes, 0 for padded
    """
    device = batch.batch.device
    idx = torch.arange(max_size, device=device).unsqueeze(0) # [1, B]
    masks = idx < counts.unsqueeze(1) # [B, max_size]
    return masks


def ged_matrix_to_dict(ged_matrix):
    return {
        (i, j): float(ged_matrix[i, j])
        for i in range(ged_matrix.shape[0])
        for j in range(i + 1, ged_matrix.shape[1])
    }


def get_sampled_cost_matrix_sizes(dataset, num_samples=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(dataset)
    sizes = []
    for _ in range(num_samples):
        i, j = rng.choice(range(n), 2)
        g1, g2 = dataset[i], dataset[j]
        small, large = sorted([g1.num_nodes, g2.num_nodes])
        sizes.append((small, large))
    return np.array(sizes)