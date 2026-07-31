import torch
import argparse
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
from torch_geometric.transforms import NormalizeFeatures, Constant

from birkhoffnet.datasets.siamese_dataset import SiameseDataset
from birkhoffnet.datasets.triplet_dataset import TripletDataset
from birkhoffnet.models.gnn_models import Model
from birkhoffnet.losses.triplet_loss import TripletLoss
from birkhoffnet.losses.ged_loss import GEDLoss
from birkhoffnet.utils.permutation import PermutationPool
from birkhoffnet.models.alpha_layers import AlphaPermutationLayer, AlphaMLP, AlphaBilinear, AlphaCrossAttention
# from birkhoffnet.utils.train_utils import AlphaTracker
from birkhoffnet.models.cost_matrix_builder import CostMatrixBuilder
from birkhoffnet.utils.diagnostics import accumulate_epoch_stats, \
                                       batched_diagnostics, \
                                       plot_history
from birkhoffnet.utils.data_utils import ged_matrix_to_dict, \
                                      compute_cost_matrices, \
                                      pad_cost_matrices, \
                                      get_node_masks
from birkhoffnet.utils.model_utils import ModelFactory
from birkhoffnet.utils.config import load_data



class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        g.graph_id = idx
        return g

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


# def plot_assignments_and_alphas(idx1, idx2, soft_assignment, alphas):
#     # Load dataset
#     train_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=True)
#     test_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=False)

#     if 'x' not in train_dataset[0]:
#         train_dataset.transform = Constant(value=1.0)
#         test_dataset.transform = Constant(value=1.0)
    
#     dataset = ConcatDataset([train_dataset, test_dataset])

#     g1 = dataset[idx1]
#     g2 = dataset[idx2]

#     G1 = to_networkx(g1, to_undirected=True)
#     G2 = to_networkx(g2, to_undirected=True)

#     node_labels1 = g1.x.argmax(dim=1).numpy()
#     node_labels2 = g2.x.argmax(dim=1).numpy()

#     pos1 = nx.kamada_kawai_layout(G1)
#     pos2 = nx.kamada_kawai_layout(G2)

#     for key in pos2:
#         pos2[key][0] += 3  # Offset second graph

#     color_list = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 9)

#     # Create the figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

#     # --- Graph + Soft Assignments ---
#     nx.draw(G1, pos=pos1, ax=ax1, node_color=[color_list[l] for l in node_labels1],
#             edge_color='gray', with_labels=False)
#     nx.draw(G2, pos=pos2, ax=ax1, node_color=[color_list[l] for l in node_labels2],
#             edge_color='gray', with_labels=False)

#     for i in range(len(G1.nodes)):
#         for j in range(len(G2.nodes)):
#             weight = soft_assignment[i, j]
#             # if weight >= 0.1:
#             x_vals = [pos1[i][0], pos2[j][0]]
#             y_vals = [pos1[i][1], pos2[j][1]]
#             ax1.plot(x_vals, y_vals, color='red', alpha=float(weight), linewidth=2 * float(weight))

#     ax1.set_title("Soft Assignments")
#     ax1.axis('off')

#     # --- Alpha Distribution ---
#     ax2.bar(range(1, len(alphas) + 1), alphas)
#     ax2.set_title("Alpha Distribution")
#     ax2.set_xlabel("Alpha - Permutation Matrix Index")
#     ax2.set_xticks(range(1, len(alphas) + 1, 2), range(1, len(alphas) + 1, 2), rotation=90)
#     ax2.set_ylabel("Alpha Weight")
#     ax2.set_ylim(0.0, 1.0)

#     plt.tight_layout()
#     # plt.savefig(f'./res/AIDS/combined_assignments_{idx1}_{idx2}_unnormalized.png', dpi=800)
#     plt.show()


def plot_assignments_and_alphas(
        dataset,
        idx1,
        idx2,
        soft_assignment,
        alphas,
        threshold=0.15,
):

    g1 = dataset[idx1]
    g2 = dataset[idx2]

    G1 = to_networkx(g1, to_undirected=True)
    G2 = to_networkx(g2, to_undirected=True)

    real_n1 = len(G1.nodes)
    real_n2 = len(G2.nodes)

    node_labels1 = g1.x.argmax(dim=1).cpu().numpy()
    node_labels2 = g2.x.argmax(dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    pos1 = nx.kamada_kawai_layout(G1)
    pos2 = nx.kamada_kawai_layout(G2)

    # normalize layouts
    for p in (pos1, pos2):
        xs = np.array([v[0] for v in p.values()])
        ys = np.array([v[1] for v in p.values()])

        for k in p:
            p[k][0] = (p[k][0] - xs.mean()) / xs.std()
            p[k][1] = (p[k][1] - ys.mean()) / ys.std()

    # move G2 to the right
    gap = 4.5
    for k in pos2:
        pos2[k][0] += gap

    # ------------------------------------------------------------------
    # Dummy nodes
    # ------------------------------------------------------------------

    n_dummy = real_n2 - real_n1

    if n_dummy > 0:

        x_dummy = max(v[0] for v in pos1.values()) + 0.55

        ymin = min(v[1] for v in pos1.values())
        ymax = max(v[1] for v in pos1.values())

        ys = np.linspace(ymin, ymax, n_dummy)

        dummy_pos = {}

        for idx, node in enumerate(range(real_n1, real_n2)):
            pos1[node] = np.array([x_dummy, ys[idx]])
            dummy_pos[node] = pos1[node]

    else:
        dummy_pos = {}

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    palette = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 9)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [2.4, 1]},
    )

    # ------------------------------------------------------------------
    # Draw graph edges first
    # ------------------------------------------------------------------

    nx.draw_networkx_edges(
        G1,
        pos1,
        ax=ax1,
        edge_color="0.75",
        width=1.0,
    )

    nx.draw_networkx_edges(
        G2,
        pos2,
        ax=ax1,
        edge_color="0.75",
        width=1.0,
    )

    # ------------------------------------------------------------------
    # Soft assignments
    # ------------------------------------------------------------------

    soft_assignment = soft_assignment.cpu()

    for i in range(real_n2):

        for j in range(real_n2):

            w = float(soft_assignment[i, j])

            if w < threshold:
                continue

            ax1.plot(
                [pos1[i][0], pos2[j][0]],
                [pos1[i][1], pos2[j][1]],
                color="crimson",
                alpha=min(1.0, 0.3 + w),
                linewidth=2 * w,
                zorder=1,
            )

    # ------------------------------------------------------------------
    # Real nodes
    # ------------------------------------------------------------------

    nx.draw_networkx_nodes(
        G1,
        pos1,
        nodelist=range(real_n1),
        node_color=[palette[c] for c in node_labels1],
        node_size=320,
        edgecolors="black",
        linewidths=0.7,
        ax=ax1,
    )

    nx.draw_networkx_nodes(
        G2,
        pos2,
        node_color=[palette[c] for c in node_labels2],
        node_size=320,
        edgecolors="black",
        linewidths=0.7,
        ax=ax1,
    )

    # ------------------------------------------------------------------
    # Dummy nodes
    # ------------------------------------------------------------------

    if len(dummy_pos):

        ax1.scatter(
            [p[0] for p in dummy_pos.values()],
            [p[1] for p in dummy_pos.values()],
            marker="X",
            s=160,
            color="dimgray",
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label="Dummy nodes",
        )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    g1_x = [pos1[i][0] for i in range(real_n1)]
    g2_x = [pos2[i][0] for i in range(real_n2)]

    y_text = max(
        max(pos1[i][1] for i in range(real_n1)),
        max(pos2[i][1] for i in range(real_n2))
    ) + 0.5

    ax1.text(
        np.mean(g1_x),
        y_text,
        r"$G_1$",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    ax1.text(
        np.mean(g2_x),
        y_text,
        r"$G_2$",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    # if n_dummy > 0:
    #     ax1.text(
    #         x_dummy,
    #         y_text,
    #         "Dummy",
    #         ha="center",
    #         fontsize=12,
    #     )

    # if len(dummy_pos):
    #     ax1.text(
    #         x_dummy,
    #         1.8,
    #         "Dummy",
    #         fontsize=12,
    #         ha="center",
    #     )

    # ax1.text(
    #     np.mean([v[0] for v in pos2.values()]),
    #     1.8,
    #     r"$G_2$",
    #     fontsize=14,
    #     ha="center",
    #     fontweight="bold",
    # )

    ax1.axis("off")

    # ------------------------------------------------------------------
    # Alpha histogram
    # ------------------------------------------------------------------

    ax2.bar(
        np.arange(len(alphas)),
        alphas,
        color="steelblue",
        width=0.8,
    )

    ax2.set_xlabel("Permutation index")
    ax2.set_ylabel(r"$\alpha$")
    ax2.set_xticks(
        np.arange(0, len(alphas), 5),
        labels=np.arange(1, len(alphas) + 1, 5)
    )
    ax2.set_ylim(0, 1.0)

    sns.despine(ax=ax2)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"res/journal/PROTEINS_full/assignment_{idx1}_{idx2}.pdf", dpi=400)
    plt.close()


# def plot_assignments_and_alphas(
#         dataset,
#         idx1, 
#         idx2, 
#         soft_assignment, 
#         alphas
#     ):

#     g1 = dataset[idx1]
#     g2 = dataset[idx2]

#     G1 = to_networkx(g1, to_undirected=True)
#     G2 = to_networkx(g2, to_undirected=True)

#     real_n1 = len(G1.nodes)
#     real_n2 = len(G2.nodes)

#     node_labels1 = g1.x.argmax(dim=1).numpy()
#     node_labels2 = g2.x.argmax(dim=1).numpy()

#     pos1 = nx.kamada_kawai_layout(G1)
#     pos2 = nx.kamada_kawai_layout(G2)

#     # for i in range(real_n1, real_n2):
#     #     pos1[i] = [-1.0, i - real_n1]

#     x_dummy = max(p[0] for p in pos1.values()) + 0.35

#     ys = np.linspace(
#         min(y for _, y in pos1.values()),
#         max(y for _, y in pos1.values()),
#         real_n2 - real_n1,
#     )

#     for k, i in enumerate(range(real_n1, real_n2)):
#         pos1[i] = (x_dummy, ys[k])

#     for key in pos2:
#         pos2[key][0] += 3  # Offset second graph

#     color_list = sns.color_palette("tab20", 20) + sns.color_palette("Set3", 9)

#     # Create the figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(
#         1, 2, figsize=(14, 6), 
#         gridspec_kw={'width_ratios': [2, 1]}
#     )

#     # --- Graph + Soft Assignments ---
#     nx.draw(
#         G1,
#         pos=pos1, 
#         ax=ax1, 
#         node_color=[color_list[l] for l in node_labels1],
#         edge_color='gray', 
#         with_labels=False
#     )

#     nx.draw(
#         G2, 
#         pos=pos2, 
#         ax=ax1, 
#         node_color=[color_list[l] for l in node_labels2],
#         edge_color='gray', 
#         with_labels=False
#     )

#     ax1.scatter(
#         [pos1[i][0] for i in range(real_n1, real_n2)],
#         [pos1[i][1] for i in range(real_n1, real_n2)],
#         color='black',
#         marker='x',
#         s=100,
#         label='Dummy G1'
#     )

#     soft_assignment = soft_assignment.cpu()

#     for i in range(len(G2.nodes)):
#         for j in range(len(G2.nodes)):
            
#             weight = float(soft_assignment[i, j])
            
#             x_vals = [pos1[i][0], pos2[j][0]]
#             y_vals = [pos1[i][1], pos2[j][1]]
            
#             ax1.plot(
#                 x_vals, 
#                 y_vals, 
#                 color='red', 
#                 alpha=weight, 
#                 linewidth=2 * weight
#             )

#     # ax1.set_title(f"Soft Assignments (G{idx1} ↔ G{idx2})")
#     ax1.axis('off')

#     # --- Alpha Distribution ---
#     # ax2.bar(range(len(alphas)), alphas)
#     ax2.bar(np.arange(len(alphas)), alphas, width=0.8)

#     # ax2.set_title("Permutation Weights (alphas)")
#     ax2.set_xlabel("Permutation index")
#     ax2.set_ylabel("Weight")
#     ax2.set_ylim(0.0, 1.0)

#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f"res/journal/PROTEINS_full/assignment_{idx1}_{idx2}.pdf", dpi=400)
#     plt.close()


# def main():

#     # Load dataset
#     train_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=True)
#     test_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=False)

#     # print(train_dataset.ged[20, 607], ' ', train_dataset.ged[20, 562], ' ', train_dataset.ged[20, 611])
#     print(torch.exp(-train_dataset.norm_ged[20, 607]).item(), ' ', torch.exp(-train_dataset.norm_ged[20, 562]).item(), ' ', torch.exp(-train_dataset.norm_ged[20, 611]).item())

#     if 'x' not in train_dataset[0]:
#         train_dataset.transform = Constant(value=1.0)
#         test_dataset.transform = Constant(value=1.0)

#     dataset = ConcatDataset([train_dataset, test_dataset])

#     num_features = train_dataset.num_features

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load models
#     embedding_dim = 64
#     encoder = Model(num_features, embedding_dim, 1, use_attention=False, attn_concat=False).to(device)
#     encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-5)

#     max_graph_size = max([g.num_nodes for g in dataset])
#     k = (max_graph_size - 1) ** 2 + 1 # upper (theoretical) bound
#     k = 21
    
#     perm_pool = PermutationPool(max_n=max_graph_size, k=k)
#     perm_matrices = perm_pool.get_matrix_batch().to(device)

#     model = AlphaMLP(encoder.output_dim, k)
#     # model = AlphaBilinear(encoder.output_dim, k)
#     # model = AlphaCrossAttention(encoder.output_dim, k)
#     alpha_layer = AlphaPermutationLayer(perm_matrices, model).to(device)

#     cost_builder = CostMatrixBuilder(embedding_dim, max_graph_size, use_learned_sub=False)

#     criterion = criterion = GEDLoss().to(device)

#     ged_optimizer = torch.optim.AdamW(
#         list(alpha_layer.parameters()) + list(cost_builder.parameters()) + list(criterion.parameters()),
#         lr=1e-3,
#         weight_decay=1e-5
#     )

#     checkpoint_encoder = torch.load('res/debug/checkpoint_encoder_debug.pth', map_location=device)
#     encoder.load_state_dict(checkpoint_encoder['encoder'])
#     encoder_optimizer.load_state_dict(checkpoint_encoder['optimizer'])

#     encoder = encoder.to(device)

#     checkpoint_ged = torch.load('res/debug/checkpoint_ged_debug.pth', map_location=device)
#     alpha_layer.load_state_dict(checkpoint_ged['alpha_layer'])
#     ged_optimizer.load_state_dict(checkpoint_ged['optimizer'])
#     criterion.load_state_dict(checkpoint_ged['criterion'])

#     alpha_layer = alpha_layer.to(device)
#     cost_builder = cost_builder.to(device)
#     criterion = criterion.to(device)

#     encoder.eval()
#     alpha_layer.eval()
#     criterion.eval()

#     # Select graphs
#     indices = [20, 607, 562, 611] # G0, G1 similar, G2 less similar, G3 dissimilar
#     # indices = [20, 560, 570, 604]
#     # indices = [6, 7, 13, 17]
#     # indices = [10, 1230, 1, 1]
#     selected_graphs = [dataset[i] for i in indices]

#     G0 = selected_graphs[0]
#     others = selected_graphs[1:]

#     data = CustomGraphPairDataset(
#         G0,
#         others,
#         train_dataset.ged,
#         indices[0],
#         indices[1:]
#     )

#     loader = DataLoader(data, batch_size=3, collate_fn=lambda batch: (
#         Batch.from_data_list([x[0] for x in batch]),  # b1
#         Batch.from_data_list([x[1] for x in batch]),  # b2
#         torch.tensor([x[2] for x in batch])           # geds
#     ))

#     with torch.no_grad():
#         for batch in loader:

#                 batch1, batch2, _ = batch
#                 batch1, batch2 = batch1.to(device), batch2.to(device)

#                 n_nodes_1 = batch1.batch.bincount()
#                 n_nodes_2 = batch2.batch.bincount()

#                 normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

#                 node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
#                 node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

#                 cost_matrices, masks1, masks2 = cost_builder(node_repr_b1, graph_repr_b1, batch1.batch, node_repr_b2, graph_repr_b2, batch2.batch)

#                 soft_assignments, alphas = alpha_layer(graph_repr_b1, graph_repr_b2)

#                 assignment_masks = masks1.unsqueeze(2) * masks2.unsqueeze(1)
#                 soft_assignments = soft_assignments * assignment_masks

#                 row_sums = soft_assignments.sum(dim=-1, keepdim=True).clamp(min=1e-8)
#                 soft_assignments = soft_assignments / row_sums
                
#                 predicted_ged = criterion(cost_matrices, soft_assignments)
#                 # print(predicted_ged)
#                 normalized_predicted_ged = torch.exp(- predicted_ged / normalization_factor)
#                 print(normalized_predicted_ged)

#     plot_assignments_and_alphas(20, 607, soft_assignments[0], alphas[0].cpu().numpy())
#     plot_assignments_and_alphas(20, 562, soft_assignments[1], alphas[1].cpu().numpy())
#     plot_assignments_and_alphas(20, 611, soft_assignments[2], alphas[2].cpu().numpy())
    
#     # plot_assignments_and_alphas(10, 1230, soft_assignments[0], alphas[0].cpu().numpy())


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Path to parameters file')
    return parser

def main(args):

    # config, metadata, ged_data = load_data(args.params)
    config, metadata, ged_data, valid_idx, _, _, _ = load_data(args.params)

    device = torch.device(config.device)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    use_attrs = True
    transform = NormalizeFeatures() if use_attrs else None

    dataset_full = TUDataset(
        root=config.dataset_dir,
        name=config.dataset,
        use_node_attr=use_attrs,
        transform=transform
    )

    # --------------------------------------------------
    # 2. Metadata filtering
    # --------------------------------------------------

    valid_indices = valid_idx.tolist()

    dataset = [dataset_full[i] for i in valid_indices]

    # --------------------------------------------------
    # 3. Load GED matrices
    # --------------------------------------------------

    norm_ged_matrix = ged_data["norm_ged_matrix"]
    node_counts = ged_data["node_counts"]

    max_nodes = int(torch.max(node_counts).item())

    # --------------------------------------------------
    # 4. Build filtered dataset
    # --------------------------------------------------

    # filtered_dataset = [dataset[i] for i in valid_indices]

    # if filtered_dataset[0].x is None:
    #     for g in filtered_dataset:
    #         g.x = torch.ones((g.num_nodes, 1))

    # num_features = filtered_dataset[0].num_node_features

    # --------------------------------------------------
    # 5. Initialize models
    # --------------------------------------------------

    components = ModelFactory.initialize(
        num_features=dataset_full.num_features,
        max_graph_size=max_nodes,
        config=config
    )

    encoder = components.modules.encoder
    alpha_layer = components.modules.alpha_layer
    cost_builder = components.modules.cost_builder

    criterion = GEDLoss().to(config.device)

    # --------------------------------------------------
    # 6. Load checkpoints
    # --------------------------------------------------

    ckpt_encoder_path = f"{config.output_dir}/ckpt_encoder.pth"
    ckpt_encoder = torch.load(ckpt_encoder_path, map_location=device)

    encoder.load_state_dict(ckpt_encoder["encoder"])

    ckpt_ged_path = f"{config.output_dir}/ckpt_ged.pth"
    ckpt_ged = torch.load(ckpt_ged_path, map_location=device)

    alpha_layer.load_state_dict(ckpt_ged["alpha_layer"])
    cost_builder.load_state_dict(ckpt_ged["cost_builder"])
    criterion.load_state_dict(ckpt_ged["criterion"])

    encoder.eval()
    alpha_layer.eval()
    cost_builder.eval()
    criterion.eval()

    # --------------------------------------------------
    # 7. Select graphs
    # --------------------------------------------------

    # indices = [1, 4, 8, 2, 0, 3]
    # indices = [2, 2, 4, 6, 7, 9]
    # indices = [2, 2, 56, 349, 380]
    # indices = [2, 2, 80, 267, 349, 380]
    indices = [2, 2, 56, 267]

    # print(
    #     norm_ged_matrix[1, 4].item(),
    #     norm_ged_matrix[1, 8].item(),
    #     norm_ged_matrix[1, 2].item(),
    #     norm_ged_matrix[1, 0].item(),
    #     norm_ged_matrix[1, 3].item()
    # )

    selected_graphs = [dataset[i] for i in indices]

    G0 = selected_graphs[0]
    others = selected_graphs[1:]

    graph_dataset = GraphDataset(selected_graphs)

    data = CustomGraphPairDataset(
        G0,
        others,
        norm_ged_matrix,
        indices[0],
        indices[1:]
    )

    graph_loader = DataLoader(graph_dataset, batch_size=len(dataset), shuffle=False)

    loader = DataLoader(
        data,
        batch_size=3,
        collate_fn=lambda batch: (
            Batch.from_data_list([x[0] for x in batch]),
            Batch.from_data_list([x[1] for x in batch]),
            torch.tensor([x[2] for x in batch])
        )
    )

    # --------------------------------------------------
    # 8. Forward pass
    # --------------------------------------------------

    with torch.no_grad():

        all_node = []
        all_graph = []

        max_nodes = int(metadata["node_stats"]["upper_bound"])
        node_dim = None

        for batch in graph_loader:

            batch = batch.to(device)

            node_repr, graph_repr = encoder(
                batch.x, batch.edge_index, batch.batch
            )

            node_splits = batch.batch.bincount().tolist()
            node_split = torch.split(node_repr, node_splits)

            for i, gid in enumerate(batch.graph_id.tolist()):
                n = node_split[i].detach().cpu()

                # max_nodes = max(max_nodes, n.size(0))
                node_dim = n.size(1)

                all_node.append((gid, n))
                all_graph.append((gid, graph_repr[i].detach().cpu()))

        num_graphs = max(g for g, _ in all_node) + 1

        node_cache = torch.zeros((num_graphs, max_nodes, node_dim), device=device)
        mask_cache = torch.zeros((num_graphs, max_nodes), dtype=torch.bool, device=device)
        graph_cache = torch.zeros((num_graphs, graph_repr.size(1)), device=device)

        for (gid, n) in all_node:
            node_cache[gid, :n.size(0)] = n
            mask_cache[gid, :n.size(0)] = True

        for (gid, g) in all_graph:
            graph_cache[gid] = g

        for graphs_a, graphs_b, target_similarity in loader:

            graphs_a = graphs_a.to(config.device)
            graphs_b = graphs_b.to(config.device)
            target_similarity = target_similarity.to(config.device)

            graph_ids_a = graphs_a.graph_id
            graph_ids_b = graphs_b.graph_id

            node_emb_a = node_cache[graph_ids_a]
            node_mask_a = mask_cache[graph_ids_a]
            graph_emb_a = graph_cache[graph_ids_a]

            node_emb_b = node_cache[graph_ids_b]
            node_mask_b = mask_cache[graph_ids_b]
            graph_emb_b = graph_cache[graph_ids_b]

            num_nodes_a = node_mask_a.sum(dim=1)
            num_nodes_b = node_mask_b.sum(dim=1)

            avg_num_nodes = 0.5 * (num_nodes_a + num_nodes_b)

            cost_matrix, node_mask_a, node_mask_b = cost_builder(
                node_emb_a,
                node_mask_a,
                graph_emb_a,
                node_emb_b,
                node_mask_b,
            )

            assignment_matrix, alphas = alpha_layer(
                graph_emb_a,
                graph_emb_b,
            )

            assignment_mask = node_mask_a.unsqueeze(2) * node_mask_b.unsqueeze(1)

            assignment_matrix = assignment_matrix * assignment_mask

            row_sums = assignment_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            assignment_matrix = assignment_matrix / row_sums

            col_sums = assignment_matrix.sum(dim=-2, keepdim=True).clamp(min=1e-8)
            assignment_matrix = assignment_matrix / col_sums

            pred_ged = criterion(
                cost_matrix,
                assignment_matrix,
            )

            pred_similarity = torch.exp(
                -pred_ged / avg_num_nodes
            )

            print(pred_similarity)

    # --------------------------------------------------
    # 9. Visualization
    # --------------------------------------------------

    dataset_full = TUDataset(
        root=config.dataset_dir,
        name=config.dataset,
    )

    dataset = [dataset_full[i] for i in valid_indices]

    plot_assignments_and_alphas(dataset, indices[0], indices[1], assignment_matrix[0].cpu(), alphas[0].cpu().numpy())
    plot_assignments_and_alphas(dataset, indices[0], indices[2], assignment_matrix[1].cpu(), alphas[1].cpu().numpy())
    plot_assignments_and_alphas(dataset, indices[0], indices[3], assignment_matrix[2].cpu(), alphas[2].cpu().numpy())
    # plot_assignments_and_alphas(dataset, indices[0], indices[4], assignment_matrix[3].cpu(), alphas[3].cpu().numpy())
    # plot_assignments_and_alphas(dataset, indices[0], indices[5], assignment_matrix[4].cpu(), alphas[4].cpu().numpy())


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)