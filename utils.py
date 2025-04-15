import numpy as np

import torch
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from itertools import combinations


def compute_cost_matrix(representations1, representations2):
    return torch.cdist(representations1, representations2, p=2)


def compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b2, batch2):
    distance_matrices = []
    num_graphs = batch1.batch.max().item() + 1
    for i in range(num_graphs):
        nodes1 = node_repr_b1[batch1.batch == i]
        nodes2 = node_repr_b2[batch2.batch == i]
        dist = torch.cdist(nodes1, nodes2, p=2)
        distance_matrices.append(dist)
    return distance_matrices


def pad_cost_matrices(cost_matrices, max_graph_size, pad_value=0.0):
    padded = []
    for cost in cost_matrices:
        n1, n2 = cost.shape
        pad_bottom = max_graph_size - n1
        pad_right = max_graph_size - n2
        padded_cost = F.pad(cost, (0, pad_right, 0, pad_bottom), value=pad_value)
        padded.append(padded_cost)
    return torch.stack(padded)


def generate_attention_masks(batch_size, batch1, batch2, max_graph_size):
    masks = torch.zeros(batch_size, max_graph_size, max_graph_size)
    num_nodes_b1 = np.bincount(batch1.batch.cpu().numpy())
    num_nodes_b2 = np.bincount(batch2.batch.cpu().numpy())
    for b in range(batch_size):
        real_size1 = num_nodes_b1[b]
        real_size2 = num_nodes_b2[b]
        masks[b, :real_size1, :real_size2] = 1
    return masks


def plot_attention(attention_weights, epoch, save_path="./res/attention_epoch_aware_test_{epoch}.png"):
    attn = torch.sigmoid(attention_weights).detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn, cmap="viridis", square=True, cbar=True)
    plt.title(f"Learned Attention Weights (Epoch {epoch})")
    plt.xlabel("Node i")
    plt.ylabel("Node j")
    plt.tight_layout()
    plt.savefig(save_path.format(epoch=epoch))
    plt.close()


def get_ged_labels(distance_matrix):
    num_graphs = len(distance_matrix)
    graph_pairs = combinations(range(num_graphs), r=2)
    ged_labels = {}
    for i, j in graph_pairs:
        ged_labels[(i,j)] = distance_matrix[i,j]
    return ged_labels


def get_cost_matrices_distr(dataset):
    sizes = []
    for i, j in combinations(range(len(dataset)), r=2):
        g1, g2 = dataset[i], dataset[j]
        if g1.num_nodes <= g2.num_nodes:
            sizes.append((g1.num_nodes, g2.num_nodes))
        else:
            sizes.append((g2.num_nodes, g1.num_nodes))
    sizes = np.array(sizes)
    return sizes


def visualize_node_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        data = data.to('cuda')
        node_embeddings, _ = model(data.x, data.edge_index, data.batch)
    
    # One label per graph (batch element)
    graph_labels = data.y.cpu().numpy()  # shape: (num_graphs,)
    node_graph_ids = data.batch.cpu().numpy()  # shape: (num_nodes,)

    # Choose colors (expand if you have more than 2 classes)
    c = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink']

    # Assign color per node based on the graph label
    colors = [c[graph_labels[graph_id]] for graph_id in node_graph_ids]

    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = reducer.fit_transform(node_embeddings.cpu().numpy())

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, s=10, alpha=0.7)
    plt.title("Node Embedding Clusters (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()