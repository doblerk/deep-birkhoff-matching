import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import ConcatDataset
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import Constant


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


def plot_assignments(idx1, idx2, soft_assignment, dataset, output_dir):
    train_dataset = GEDDataset(root=f'data/datasets/{dataset}', name=dataset, train=True)
    test_dataset = GEDDataset(root=f'data/datasets/{dataset}', name=dataset, train=False)

    if 'x' not in train_dataset[0]:
        train_dataset.transform = Constant(value=1.0)
        test_dataset.transform = Constant(value=1.0)

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

    for key in pos2:
        pos2[key][0] += 3                                              

    fig, ax = plt.subplots(figsize=(10, 6))

    nx.draw(G1, pos=pos1, ax=ax, node_color=[color_list[label] for label in node_labels1], edge_color='gray', with_labels=True)
    nx.draw(G2, pos=pos2, ax=ax, node_color=[color_list[label] for label in node_labels2], edge_color='gray', with_labels=True)

    # Overlay soft assignments as weighted edges
    for i in range(len(G1.nodes)):
        for j in range(len(G2.nodes)):
            weight = soft_assignment[i, j]
            x_vals = [pos1[i][0], pos2[j][0]]
            y_vals = [pos1[i][1], pos2[j][1]]
            ax.plot(x_vals, y_vals, color='red', alpha=weight, linewidth=2*weight)

    plt.axis('off')
    plt.savefig(f'{output_dir}/assignments_{idx1}_{idx2}.png', dpi=600)


def plot_assignments_and_alphas(idx1, idx2, soft_assignment, alphas, dataset, output_dir):
    train_dataset = GEDDataset(root=f'data/datasets/{dataset}', name=dataset, train=True)
    test_dataset = GEDDataset(root=f'data/datasets/{dataset}', name=dataset, train=False)

    if 'x' not in train_dataset[0]:
        train_dataset.transform = Constant(value=1.0)
        test_dataset.transform = Constant(value=1.0)
    
    dataset = ConcatDataset([train_dataset, test_dataset])

    g1 = dataset[idx1]
    g2 = dataset[idx2]

    G1 = to_networkx(g1, to_undirected=True)
    G2 = to_networkx(g2, to_undirected=True)

    node_labels1 = g1.x.argmax(dim=1).numpy()
    node_labels2 = g2.x.argmax(dim=1).numpy()

    pos1 = nx.kamada_kawai_layout(G1)
    pos2 = nx.kamada_kawai_layout(G2)

    for key in pos2:
        pos2[key][0] += 3

    color_list = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 9)

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

    # --- Graph + Soft Assignments ---
    nx.draw(G1, pos=pos1, ax=ax1, node_color=[color_list[l] for l in node_labels1],
            edge_color='gray', with_labels=False)
    nx.draw(G2, pos=pos2, ax=ax1, node_color=[color_list[l] for l in node_labels2],
            edge_color='gray', with_labels=False)

    for i in range(len(G1.nodes)):
        for j in range(len(G2.nodes)):
            weight = soft_assignment[i, j]
            x_vals = [pos1[i][0], pos2[j][0]]
            y_vals = [pos1[i][1], pos2[j][1]]
            ax1.plot(x_vals, y_vals, color='red', alpha=float(weight), linewidth=2 * float(weight))

    ax1.set_title("Soft Assignments")
    ax1.axis('off')

    # --- Alpha Distribution ---
    ax2.bar(range(len(alphas)), alphas)
    ax2.set_title("Alpha Distribution")
    ax2.set_xlabel("Alpha - Permutation Matrix Index")
    ax2.set_xticks(range(0, len(alphas), 2), range(0, len(alphas), 2), rotation=90)
    ax2.set_ylabel("Alpha Weight")
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_assignments_{idx1}_{idx2}.png', dpi=800)