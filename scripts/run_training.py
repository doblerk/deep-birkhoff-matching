import argparse

import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import random_split, ConcatDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import Constant

from birkhoffnet.datasets.siamese_dataset import SiameseDataset
from birkhoffnet.datasets.triplet_dataset import TripletDataset
from birkhoffnet.models.gnn_models import Model
from birkhoffnet.losses.triplet_loss import TripletLoss
from birkhoffnet.losses.ged_loss import GEDLoss
from birkhoffnet.utils.permutation import PermutationPool
from birkhoffnet.models.indels_layers import IndelBilinear
from birkhoffnet.models.alpha_layers import AlphaPermutationLayer, AlphaMLP, AlphaBilinear, AlphaCrossAttention
from birkhoffnet.utils.train_utils import AlphaTracker
from birkhoffnet.models.cost_matrix_builder import CostMatrixBuilder
from birkhoffnet.utils.diagnostics import accumulate_epoch_stats, \
                                       batched_diagnostics, \
                                       plot_history
from birkhoffnet.utils.data_utils import ged_matrix_to_dict, \
                                      compute_cost_matrices, \
                                      pad_cost_matrices, \
                                      get_node_masks


def train_triplet_network(loader, encoder, optimizer, device, args, epochs=2001):
    encoder.train()
    # optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-6) # added AdamW
    criterion = TripletLoss(margin=0.8)
    
    for epoch in range(epochs):

        total_loss = 0
        total_samples = 0

        for triplet in loader:

            anchor_graphs, pos_graphs, neg_graphs = triplet

            a_batch = anchor_graphs.to(device)
            p_batch = pos_graphs.to(device)
            n_batch = neg_graphs.to(device)

            optimizer.zero_grad()

            _, a_graph_emb = encoder(a_batch.x, a_batch.edge_index, a_batch.batch)
            _, p_graph_emb = encoder(p_batch.x, p_batch.edge_index, p_batch.batch)
            _, n_graph_emb = encoder(n_batch.x, n_batch.edge_index, n_batch.batch)

            loss = criterion(a_graph_emb, p_graph_emb, n_graph_emb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * anchor_graphs.i.size(0)
            total_samples += anchor_graphs.i.size(0)

        average_loss = total_loss / total_samples
        
        if epoch % 10 == 0:
            print(f"[Triplet Stage] Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}")
    
    torch.save({
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, f'{args.output_dir}/checkpoint_encoder_debug.pth')

    return encoder


def train_siamese_network(train_loader, val_loader, test_loader, encoder, alpha_layer, alpha_tracker, perm_pool, cost_builder, criterion, device, args, epochs=201):
    optimizer = torch.optim.AdamW(
        list(alpha_layer.parameters()) + list(cost_builder.parameters()) + list(criterion.parameters()),
        lr=1e-3,
        weight_decay=1e-6
    ) # Added AdamW

    history = {k: [] for k in ["spectral_gap", "top_singular_value", "mean_singular_value", "mean_entropy"]}
    # val_losses = []

    for epoch in range(epochs):

        train_ged(train_loader, encoder, alpha_layer, alpha_tracker, perm_pool, cost_builder, criterion, optimizer, device, history=history)

        if epoch % 10 == 0:
            average_val_loss = eval_ged(val_loader, encoder, alpha_layer, cost_builder, criterion, device)
            # val_losses.append(average_val_loss)
            print(f"[GED] Epoch {epoch+1}/{epochs} - Val MSE: {average_val_loss:.4f} - RMSE: {np.sqrt(average_val_loss):.1f} - Scale: {criterion.scale.item():.4f}")
    # np.save('res/AIDS/val_losses_bl_21_second.npy', np.array(val_losses))
    average_test_loss = eval_ged(test_loader, encoder, alpha_layer, cost_builder, criterion, device)
    print(f"[GED] Final Epoch - Test MSE: {average_test_loss:.4f} - RMSE: {np.sqrt(average_test_loss):.1f} - Scale: {criterion.scale.item():.4f}")

    # plot_history(history)

    # torch.save({
    #     'alpha_layer': alpha_layer.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'criterion': criterion.state_dict(),
    # }, f'{args.output_dir}/checkpoint_ged_debug.pth')


def train_ged(train_loader, encoder, alpha_layer, alpha_tracker, perm_pool, cost_builder, criterion, optimizer, device, history=None):
    alpha_layer.train()
    criterion.train()

    for batch in train_loader:

        batch1, batch2, ged_labels = batch

        batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

        n_nodes_1 = batch1.batch.bincount()
        n_nodes_2 = batch2.batch.bincount()

        normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

        optimizer.zero_grad()

        with torch.no_grad():
            node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
            node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

        cost_matrices, masks1, masks2 = cost_builder(node_repr_b1, graph_repr_b1, batch1.batch, node_repr_b2, graph_repr_b2, batch2.batch)

        soft_assignments, alphas, entropy = alpha_layer(graph_repr_b1, graph_repr_b2)
        alpha_tracker.collect(alphas)
        
        assignment_masks = masks1.unsqueeze(2) * masks2.unsqueeze(1)
        soft_assignments = soft_assignments * assignment_masks

        row_sums = soft_assignments.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_assignments = soft_assignments / row_sums

        # if history is not None:
        #     stats = batched_diagnostics(soft_assignments.detach().cpu().numpy())
        #     history["spectral_gap"].append(stats["spectral_gap"].mean())
        #     history["top_singular_value"].append(stats["top_singular_value"].mean())
        #     history["mean_singular_value"].append(stats["mean_singular_value"].mean())
        #     history["mean_entropy"].append(stats["mean_entropy"].mean())

        predicted_ged = criterion(cost_matrices, soft_assignments)
        # normalized_predicted_ged = predicted_ged / normalization_factor
        normalized_predicted_ged = torch.exp(- predicted_ged / normalization_factor)

        loss = F.mse_loss(normalized_predicted_ged, ged_labels, reduction='mean') - 0.01 * entropy 
        # λ_ent = λ0 * exp(-epoch / τ)
        # λ0 ≈ 0.01 – 0.05
        # τ ≈ 50–100 epochs

        loss.backward()
        optimizer.step()

    sorted_idx, scores = alpha_tracker.update()
    if sorted_idx is not None:
        print(sorted_idx)
        # perm_pool.mate_permutations(sorted_idx, k=2)

        # freeze weights after pruning
        # alpha_layer.freeze_module()
        # alpha_layer.start_freeze_timer()
    
    # if alpha_layer.is_frozen():
    #     alpha_layer.update_freeze_timer()

    #     if alpha_layer.freeze_timer == 0:
    #         alpha_layer.unfreeze_module()
    #         alpha_layer.reset_freeze_timer()


@torch.no_grad()
def eval_ged(loader, encoder, alpha_layer, cost_builder, criterion, device):
    alpha_layer.eval()
    cost_builder.eval()
    criterion.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        batch1, batch2, ged_labels = batch
        batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

        n_nodes_1 = batch1.batch.bincount()
        n_nodes_2 = batch2.batch.bincount()

        normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

        node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
        node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

        cost_matrices, masks1, masks2 = cost_builder(node_repr_b1, graph_repr_b1, batch1.batch, node_repr_b2, graph_repr_b2, batch2.batch)

        soft_assignments, _, _ = alpha_layer(graph_repr_b1, graph_repr_b2)
        
        assignment_masks = masks1.unsqueeze(2) * masks2.unsqueeze(1)
        soft_assignments = soft_assignments * assignment_masks

        row_sums = soft_assignments.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_assignments = soft_assignments / row_sums

        predicted_ged = criterion(cost_matrices, soft_assignments)
        # normalized_predicted_ged = predicted_ged / normalization_factor
        normalized_predicted_ged = torch.exp(- predicted_ged / normalization_factor)

        loss = F.mse_loss(normalized_predicted_ged, ged_labels, reduction='mean')

        total_loss += loss.item() * ged_labels.size(0)
        total_samples += ged_labels.size(0)

    return total_loss / total_samples


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--k', type=int, default=21, help='Number of generated permutation matrices')
    return parser


def main(args):

    train_dataset = GEDDataset(root=f'data/datasets/{args.dataset}', name=args.dataset, train=True)
    test_dataset = GEDDataset(root=f'data/datasets/{args.dataset}', name=args.dataset, train=False)

    if 'x' not in train_dataset[0]:
        train_dataset.transform = Constant(value=1.0)
        test_dataset.transform = Constant(value=1.0)
    
    dataset = ConcatDataset([train_dataset, test_dataset])

    num_features = train_dataset.num_features

    ged_matrix, norm_ged_matrix = train_dataset.ged, train_dataset.norm_ged
    norm_ged_matrix = torch.exp(-norm_ged_matrix) # -> range (0, 1] ?

    train_size = int(0.75 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
    train_dataset_indices, val_dataset_indices = sorted(train_dataset.indices), sorted(val_dataset.indices)

    ged_dict = ged_matrix_to_dict(norm_ged_matrix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    p = int(len(train_dataset) * 0.4) # before 0.25

    triplet_train = TripletDataset(dataset, train_dataset_indices, ged_dict, k=p)
    triplet_loader = DataLoader(triplet_train, batch_size=len(triplet_train), shuffle=True, num_workers=0)

    siamese_train = SiameseDataset(dataset, norm_ged_matrix, pair_mode='train', train_indices=train_dataset_indices)
    siamese_val = SiameseDataset(dataset, norm_ged_matrix, pair_mode='val', train_indices=train_dataset_indices, val_indices=val_dataset_indices)
    siamese_test = SiameseDataset(dataset, norm_ged_matrix, pair_mode='test', train_indices=train_dataset_indices, test_indices=test_dataset.i)

    siamese_train_loader = DataLoader(siamese_train, batch_size=len(siamese_train), shuffle=True, num_workers=0)
    siamese_val_loader = DataLoader(siamese_val, batch_size=len(siamese_val), shuffle=False, num_workers=0)
    siamese_test_loader = DataLoader(siamese_test, batch_size=64 * 192, shuffle=False, num_workers=10)

    # if loading ckpt:
    ckpt = torch.load(f'{args.output_dir}/checkpoint_encoder_debug.pth', map_location=device)

    embedding_dim = 64
    encoder = Model(num_features, embedding_dim, 1, use_attention=False, attn_concat=False).to(device)
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-6) # added AdamW

    encoder.load_state_dict(ckpt["encoder"])

    encoder.eval()

    # encoder = train_triplet_network(triplet_loader, encoder, encoder_optimizer, device, args)
    encoder.freeze_params(encoder)

    max_graph_size = max([g.num_nodes for g in dataset])
    k = args.k

    perm_pool = PermutationPool(max_n=max_graph_size, k=k)
    perm_matrices = perm_pool.get_matrix_batch().to(device)

    model = AlphaMLP(encoder.output_dim, k)
    # model = AlphaCrossAttention(encoder.output_dim, k)
    alpha_layer = AlphaPermutationLayer(perm_matrices, model).to(device)
    alpha_tracker = AlphaTracker(k, warmup=10, window=5)
    model_indel = IndelBilinear(encoder.output_dim, bias=True)

    cost_builder = CostMatrixBuilder(
        embedding_dim=embedding_dim,
        max_graph_size=max_graph_size,
        use_learned_sub=False,
        model_indel=None,
        rank=None
    ).to(device)

    criterion = GEDLoss().to(device)

    train_siamese_network(siamese_train_loader, siamese_val_loader, siamese_test_loader, encoder, alpha_layer, alpha_tracker, perm_pool, cost_builder, criterion, device, args)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)