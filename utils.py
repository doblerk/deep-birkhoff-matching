import random

import numpy as np

import torch
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from itertools import combinations

from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset

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


def pad_cost_matrices(cost_matrices, max_graph_size, pad_value=0.0):
    padded = []
    for cost in cost_matrices:
        n1, n2 = cost.shape
        pad_bottom = max_graph_size - n1
        pad_right = max_graph_size - n2
        padded_cost = F.pad(cost, (0, pad_right, 0, pad_bottom), value=pad_value)
        padded.append(padded_cost)
    return torch.stack(padded)


def get_node_masks(batch, max_size):
    """
    batch: torch_geometric.data.Batch object
    returns: [B, max_size] tensor, 1 for real nodes, 0 for padded
    """
    device = batch.batch.device
    counts = batch.batch.bincount()
    B = counts.size(0)

    masks = torch.zeros(B, max_size, device=device)
    for i, count in enumerate(counts):
        masks[i, :count] = 1
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


def get_cost_matrix_sizes(dataset):
    sizes = []
    for i, j in combinations(range(len(dataset)), r=2):
        g1, g2 = dataset[i], dataset[j]
        small, large = sorted([g1.num_nodes, g2.num_nodes])
        sizes.append((small, large))
    return np.array(sizes)


def get_sampled_cost_matrix_sizes(dataset, num_samples=10000, seed=42):
    random.seed(seed)
    n = len(dataset)
    sizes = []
    for _ in range(num_samples):
        i, j = random.sample(range(n), 2)
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