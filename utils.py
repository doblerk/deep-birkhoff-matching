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


def triplet_collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    anchor_batch = Batch.from_data_list(anchors)
    pos_batch = Batch.from_data_list(positives)
    neg_batch = Batch.from_data_list(negatives)
    return anchor_batch, pos_batch, neg_batch


def compute_cost_matrix(representations1, representations2):
    return torch.cdist(representations1, representations2, p=2)


def compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b2, batch2):
    num_graphs = batch1.batch.max().item() + 1

    # Precompute all distances (between all nodes)
    all_dists = torch.cdist(node_repr_b1, node_repr_b2, p=2)  # [N1, N2]

    # Build index lists for each graph
    b1_indices = [((batch1.batch == i).nonzero(as_tuple=True)[0]) for i in range(num_graphs)]
    b2_indices = [((batch2.batch == i).nonzero(as_tuple=True)[0]) for i in range(num_graphs)]
    
    # Gather relevant slices
    distance_matrices = [
        all_dists[idx1][:, idx2]
        for idx1, idx2 in zip(b1_indices, b2_indices)
    ]

    return distance_matrices


# def pad_cost_matrices(cost_matrices, max_graph_size, pad_value=0.0):
#     padded = []
#     for cost in cost_matrices:
#         n1, n2 = cost.shape
#         pad_bottom = max_graph_size - n1
#         pad_right = max_graph_size - n2
#         padded_cost = F.pad(cost, (0, pad_right, 0, pad_bottom), value=pad_value)
#         padded.append(padded_cost)
#     return torch.stack(padded)

def pad_cost_matrices(cost_matrices, max_graph_size, pad_value=0.0):
    padded = []
    for cost in cost_matrices:
        n1, n2 = cost.shape
        square_size = max(n1, n2)

        # First: make square (insertion/deletion)
        cost = F.pad(cost, (0, square_size - n2, 0, square_size - n1), value=pad_value)

        # Second: pad to common batch-wide size
        pad_bottom = max_graph_size - square_size
        pad_right = max_graph_size - square_size
        cost = F.pad(cost, (0, pad_right, 0, pad_bottom), value=0.0)

        padded.append(cost)

    return torch.stack(padded)  # shape: (B, max_graph_size, max_graph_size)


def get_masks(cost_matrices, max_graph_size):
    """
    Generate masks for square padding (to handle insertions/deletions)
    and batch padding (to pad to max_graph_size).
    """
    square_masks = []
    batch_masks = []

    for cost in cost_matrices:
        n1, n2 = cost.shape
        
        # Create the square mask (for insertions/deletions)
        square_mask = torch.ones_like(cost, dtype=torch.bool)
        
        # Mask out the padded rows and columns (square padding)
        if n1 < n2:  # Insertion in rows
            square_mask[n1:] = 0
        elif n2 < n1:  # Insertion in columns
            square_mask[:, n2:] = 0

        # Create the batch mask (for padding to max_graph_size)
        batch_mask = torch.ones_like(cost, dtype=torch.bool)
        batch_mask[n1:, :] = 0  # Padding in rows
        batch_mask[:, n2:] = 0  # Padding in columns

        square_masks.append(square_mask)
        batch_masks.append(batch_mask)

    return torch.stack(square_masks), torch.stack(batch_masks)

def get_node_masks(batch, max_size):
    """
    batch: torch_geometric.data.Batch object
    returns: [B, max_size] tensor, 1 for real nodes, 0 for padded
    """
    device = batch.batch.device
    counts = batch.batch.bincount()
    B = counts.size(0)

    masks = torch.zeros(B, max_size, device=device, dtype=torch.bool)
    for i, count in enumerate(counts):
        masks[i, :count] = True
    return masks


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


def plot_ged(test_preds, distance_matrix, test_data_indices, train_data_indices):
    test_pred_distance_matrix = [i[2] for i in test_preds]
    pred_distances_normalized = ((test_pred_distance_matrix - np.min(test_pred_distance_matrix)) / (np.max(test_pred_distance_matrix) - np.min(test_pred_distance_matrix))).squeeze()

    test_true_distance_matrix = distance_matrix[test_data_indices, :]
    test_true_distance_matrix = test_true_distance_matrix[:, train_data_indices]
    test_true_distance_matrix = test_true_distance_matrix.flatten(order='C').tolist()
    true_distances_normalized = ((test_true_distance_matrix - np.min(test_true_distance_matrix)) / (np.max(test_true_distance_matrix) - np.min(test_true_distance_matrix))).squeeze()
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(true_distances_normalized, pred_distances_normalized, s=26, alpha=0.75, edgecolor='lightgrey', c='grey', linewidth=0.5)
    ax.plot(true_distances_normalized, true_distances_normalized, linestyle='-', linewidth=1, color='black')
    ax.set_xlabel('BP-GED', fontsize=22)
    ax.set_ylabel('GNN-GED', fontsize=22)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_assignments(idx1, idx2, soft_assignment):
    # dataset = TUDataset(root='data', name='MUTAG')
    train_dataset = GEDDataset(root='data/datasets/IMDBMulti', name='IMDBMulti', train=True)
    test_dataset = GEDDataset(root='data/datasets/IMDBMulti', name='IMDBMulti', train=False)
    dataset = ConcatDataset([train_dataset, test_dataset])
    
    g1 = dataset[idx1]
    g2 = dataset[idx2]
    
    G1 = to_networkx(g1, to_undirected=True)
    G2 = to_networkx(g2, to_undirected=True)

    node_labels1 = g1.x.argmax(dim=1).numpy()  # tensor -> class indices
    node_labels2 = g2.x.argmax(dim=1).numpy()

    pos1 = nx.kamada_kawai_layout(G1)  # positions for nodes in G1
    pos2 = nx.kamada_kawai_layout(G2)  # positions for nodes in G2

    color_list = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 9)  # Total = 29

    # Optionally, offset pos2 so the two graphs don't overlap
    for key in pos2:
        pos2[key][0] += 3                                              

    fig, ax = plt.subplots(figsize=(10, 6))

    nx.draw(G1, pos=pos1, ax=ax, node_color=[color_list[label] for label in node_labels1], edge_color='gray', with_labels=True)
    nx.draw(G2, pos=pos2, ax=ax, node_color=[color_list[label] for label in node_labels2], edge_color='gray', with_labels=True)

    # Overlay soft assignments as weighted edges
    for i in range(len(G1.nodes)):
        for j in range(len(G2.nodes)):
            weight = soft_assignment[i, j]
            if weight >= 0.1:
                x_vals = [pos1[i][0], pos2[j][0]]
                y_vals = [pos1[i][1], pos2[j][1]]
                ax.plot(x_vals, y_vals, color='red', alpha=weight, linewidth=2*weight)

    plt.axis('off')
    # plt.savefig(f'./res/MUTAG/assignments_{idx1}_{idx2}.png', dpi=1000)
    plt.show()


def knn_classifier(distance_matrix, train_idx, test_idx, dataset_name):

    train_distance_matrix = distance_matrix[train_idx,:]
    train_distance_matrix = train_distance_matrix[:, train_idx]
    np.fill_diagonal(train_distance_matrix, 1000)

    test_distance_matrix = distance_matrix[test_idx,:]
    test_distance_matrix = test_distance_matrix[:,train_idx]

    dataset = TUDataset(root='data', name=dataset_name)
    
    train_labels = list(dataset[train_idx].y.numpy())
    test_labels = list(dataset[test_idx].y.numpy())

    scoring = 'f1_micro'

    ks = (3, 5, 7, 9, 11)
    best_k = None
    best_score = 0

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        scores = cross_val_score(knn, 
                                 train_distance_matrix,
                                 train_labels,
                                 cv=kf,
                                 scoring=scoring)
        
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    # retrain on the test data with the best k
    knn_test = KNeighborsClassifier(n_neighbors=best_k, metric='precomputed')
    
    knn_test.fit(train_distance_matrix, train_labels)

    predictions = knn_test.predict(test_distance_matrix)

    f1 = f1_score(test_labels, predictions, average='micro')

    with open(f'./res/{dataset_name}/classification_accuracy.txt', 'a') as file:
        file.write(f'The optimal number of K-nearest neighbors is {best_k} with a mean F1 score of {best_score}\n')
        file.write(f'F1 Score on new data with K={best_k}: {f1}\n')
        file.write('\n')


def get_ged_labels(distance_matrix):
    num_graphs = len(distance_matrix)
    graph_pairs = combinations(range(num_graphs), r=2)
    ged_labels = {}
    for i, j in graph_pairs:
        ged_labels[(i,j)] = distance_matrix[i,j]
    return ged_labels


def ged_matrix_to_dict(ged_matrix):
    return {
        (i, j): float(ged_matrix[i, j])
        for i in range(ged_matrix.shape[0])
        for j in range(i + 1, ged_matrix.shape[1])
    }


def get_cost_matrix_sizes(dataset):
    sizes = []
    for i, j in combinations(range(len(dataset)), r=2):
        g1, g2 = dataset[i], dataset[j]
        small, large = sorted([g1.num_nodes, g2.num_nodes])
        sizes.append((small, large))
    return np.array(sizes)


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


def plot_querry_vs_closest(query_idx, pred_geds, dataset, N=5):

    distances = pred_geds[query_idx, :]
    distances[query_idx] = np.inf

    sorted_idx = list(np.argsort(distances))[4:4+N]

    sorted_idx.insert(0, query_idx)

    gs = [dataset[i] for i in sorted_idx]
    Gs = [to_networkx(g, to_undirected=True) for g in gs]    
    node_labels = [g.x.argmax(dim=1).numpy() for g in gs]
    pos = [nx.kamada_kawai_layout(g) for g in Gs]

    a = 3
    for i in range(1, len(pos)):
        p = pos[i]
        for key in p:
            p[key][0] += a
        a += 3                                        

    fig, ax = plt.subplots(figsize=(14, 5))

    for i in range(len(sorted_idx)):
        nx.draw(Gs[i], pos=pos[i], ax=ax, node_color=node_labels[i], cmap=plt.cm.tab10, edge_color='gray', with_labels=True)
    
    plt.axis('off')
    plt.show()


def extract_upper_triangular(matrix):
    triu_indices = np.tril_indices(matrix.shape[0], k=1)
    return matrix[triu_indices]


def precision_at_k(pred_matrix, true_matrix, k=10):
    pred_flat = pred_matrix.flatten()
    true_flat = true_matrix.flatten()

    topk_pred_indices = np.argpartition(-pred_flat, k)[:k]
    topk_true_indices = np.argpartition(-true_flat, k)[:k]

    topk_pred_set = set(topk_pred_indices)
    topk_true_set = set(topk_true_indices)

    intersection_count = len(topk_pred_set & topk_true_set)

    return intersection_count / k


def compute_rank_correlations(pred_matrix, true_matrix, k=10):
    # Spearman's rho
    spearman_rho, _ = spearmanr(pred_matrix, true_matrix, axis=None)

    # Kendall's tau
    kendall_tau, _ = kendalltau(pred_matrix.flatten(), true_matrix.flatten())

    # Precision at K
    p_at_k = precision_at_k(pred_matrix, true_matrix, k)
    return spearman_rho, kendall_tau, p_at_k


def save_model(model, model_optimizer, name, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    model_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer': model_optimizer.state_dict(),
        'epoch': epoch + 1,
    }
    torch.save(model_dict, os.path.join(output_dir, f'{name}_checkpoint.pth'))

    print(f"[INFO] Models saved to {output_dir}")


def compute_entropy(alphas):
    entropy = -torch.sum(alphas * torch.log(alphas.clamp(min=1e-8)), dim=1)
    return entropy.mean().item()
