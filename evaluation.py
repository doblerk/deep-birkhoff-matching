import torch
import numpy as np
from time import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from torch.utils.data import Dataset, random_split, ConcatDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, GEDDataset
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import Constant

from tribiged.datasets.siamese_dataset import SiameseDataset
from tribiged.datasets.triplet_dataset import TripletDataset
from tribiged.models.gnn_models import Model
from tribiged.losses.triplet_loss import TripletLoss
from tribiged.losses.ged_loss import GEDLoss
from tribiged.utils.permutation import PermutationPool
from tribiged.models.alpha_layers import AlphaPermutationLayer, AlphaMLP, AlphaBilinear
from tribiged.utils.train_utils import AlphaTracker
from tribiged.utils.diagnostics import accumulate_epoch_stats, \
                                       batched_diagnostics, \
                                       plot_history
from tribiged.utils.data_utils import ged_matrix_to_dict, \
                                      compute_cost_matrices, \
                                      pad_cost_matrices, \
                                      get_node_masks


class CustomGraphPairDataset(Dataset):
    def __init__(self, base_graph, other_graphs, norm_ged_matrix, base_idx, other_indices):
        self.base_graph = base_graph
        self.other_graphs = other_graphs
        self.norm_ged_matrix = norm_ged_matrix
        self.base_idx = base_idx
        self.other_indices = other_indices

    def __len__(self):
        return len(self.other_graphs)

    def __getitem__(self, idx):
        g1 = self.base_graph
        g2 = self.other_graphs[idx]
        i, j = self.base_idx, self.other_indices[idx]
        norm_ged = self.norm_ged_matrix[i, j]
        return g1, g2, norm_ged


def plot_assignments_and_alphas(idx1, idx2, soft_assignment, alphas):
    # Load dataset
    train_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=True)
    test_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=False)

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
        pos2[key][0] += 3  # Offset second graph

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
            # if weight >= 0.1:
            x_vals = [pos1[i][0], pos2[j][0]]
            y_vals = [pos1[i][1], pos2[j][1]]
            ax1.plot(x_vals, y_vals, color='red', alpha=float(weight), linewidth=2 * float(weight))

    ax1.set_title("Soft Assignments")
    ax1.axis('off')

    # --- Alpha Distribution ---
    ax2.bar(range(1, len(alphas) + 1), alphas)
    ax2.set_title("Alpha Distribution")
    ax2.set_xlabel("Alpha - Permutation Matrix Index")
    ax2.set_xticks(range(1, len(alphas) + 1, 2), range(1, len(alphas) + 1, 2), rotation=90)
    ax2.set_ylabel("Alpha Weight")
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout()
    # plt.savefig(f'./res/AIDS/combined_assignments_{idx1}_{idx2}.png', dpi=800)
    plt.show()


def main():

    # Load dataset
    train_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=True)
    test_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=False)

    print(train_dataset.ged[20, 607], ' ', train_dataset.ged[20, 562], ' ', train_dataset.ged[20, 611])

    if 'x' not in train_dataset[0]:
        train_dataset.transform = Constant(value=1.0)
        test_dataset.transform = Constant(value=1.0)

    dataset = ConcatDataset([train_dataset, test_dataset])

    num_features = train_dataset.num_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    embedding_dim = 64
    encoder = Model(num_features, embedding_dim, 1, use_attention=False, attn_concat=False).to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)

    max_graph_size = max([g.num_nodes for g in dataset])
    k = (max_graph_size - 1) ** 2 + 1 # upper (theoretical) bound
    k = 21
    
    perm_pool = PermutationPool(max_n=max_graph_size, k=k)
    perm_matrices = perm_pool.get_matrix_batch().to(device)

    model = AlphaMLP(encoder.output_dim, k)
    alpha_layer = AlphaPermutationLayer(perm_matrices, model).to(device)

    criterion = criterion = GEDLoss().to(device)

    ged_optimizer = torch.optim.Adam(
        list(alpha_layer.parameters()) + list(criterion.parameters()),
        lr=1e-3, 
        weight_decay=1e-5
    )

    checkpoint_encoder = torch.load('res/AIDS/checkpoint_encoder_entropy.pth', map_location=device)
    encoder.load_state_dict(checkpoint_encoder['encoder'])
    encoder_optimizer.load_state_dict(checkpoint_encoder['optimizer'])

    encoder = encoder.to(device)

    checkpoint_ged = torch.load('res/AIDS/checkpoint_ged_entropy.pth', map_location=device)
    alpha_layer.load_state_dict(checkpoint_ged['alpha_layer'])
    ged_optimizer.load_state_dict(checkpoint_ged['optimizer'])
    criterion.load_state_dict(checkpoint_ged['criterion'])

    alpha_layer = alpha_layer.to(device)
    criterion = criterion.to(device)

    encoder.eval()
    alpha_layer.eval()
    criterion.eval()

    # Select graphs
    indices = [20, 607, 562, 611] # G0, G1 similar, G2 less similar, G3 dissimilar
    # indices = [20, 560, 570, 604]
    # indices = [6, 7, 13, 17]
    # indices = [10, 1230, 1, 1]
    selected_graphs = [dataset[i] for i in indices]

    G0 = selected_graphs[0]
    others = selected_graphs[1:]

    data = CustomGraphPairDataset(
        G0,
        others,
        train_dataset.ged,
        indices[0],
        indices[1:]
    )

    loader = DataLoader(data, batch_size=3, collate_fn=lambda batch: (
        Batch.from_data_list([x[0] for x in batch]),  # b1
        Batch.from_data_list([x[1] for x in batch]),  # b2
        torch.tensor([x[2] for x in batch])           # norm_geds
    ))

    with torch.no_grad():
        for batch in loader:

                batch1, batch2, _ = batch
                batch1, batch2 = batch1.to(device), batch2.to(device)

                n_nodes_1 = batch1.batch.bincount()
                n_nodes_2 = batch2.batch.bincount()

                normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

                node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
                node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

                cost_matrices = compute_cost_matrices(node_repr_b1, n_nodes_1, node_repr_b2, n_nodes_2)
                padded_cost_matrices = pad_cost_matrices(cost_matrices, max_graph_size)

                soft_assignments, alphas = alpha_layer(graph_repr_b1, graph_repr_b2)

                row_masks = get_node_masks(batch1, max_graph_size, n_nodes_1)
                col_masks = get_node_masks(batch2, max_graph_size, n_nodes_2)

                assignment_mask = row_masks.unsqueeze(2) * col_masks.unsqueeze(1)

                soft_assignments = soft_assignments * assignment_mask

                row_sums = soft_assignments.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                soft_assignments = soft_assignments / row_sums
                
                predicted_ged = criterion(padded_cost_matrices, soft_assignments)
                print(predicted_ged)
                # normalized_predicted_ged = torch.exp(- predicted_ged / normalization_factor)
                # print(normalized_predicted_ged)

    plot_assignments_and_alphas(20, 607, soft_assignments[0], alphas[0].cpu().numpy())
    # plot_assignments_and_alphas(20, 562, soft_assignments[1], alphas[1].cpu().numpy())
    plot_assignments_and_alphas(20, 611, soft_assignments[2], alphas[2].cpu().numpy())
    
    # plot_assignments_and_alphas(10, 1230, soft_assignments[0], alphas[0].cpu().numpy())


if __name__ == '__main__':
    main()