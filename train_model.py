import torch
import numpy as np

import torch.nn.functional as F

from torch.utils.data import random_split
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from utils import get_ged_labels, \
                  compute_cost_matrix, \
                  compute_graphwise_node_distances, \
                  pad_cost_matrices, \
                  generate_attention_masks, \
                  get_cost_matrices_distr, \
                  visualize_node_embeddings, plot_attention

from model import Model

from data_aug import TripletDataset, SiameseDataset

from diff_birkhoff import PermutationPool, \
                          AlphaPermutationLayer, \
                          LearnablePaddingAttention, \
                          SoftGEDLoss, \
                          TripletLoss

import matplotlib.pyplot as plt

@torch.no_grad()
def evaluate_ged(loader, encoder, alpha_layer, device, max_graph_size):
    encoder.eval()
    alpha_layer.eval()
    
    attention_layer = LearnablePaddingAttention(max_graph_size).to(device)
    attention_layer.eval()

    criterion = SoftGEDLoss()

    all_preds = []
    all_labels = []

    for batch in loader:

        batch1, batch2, ged_labels = batch
        batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

        n_nodes_1 = batch1.batch.bincount()
        n_nodes_2 = batch2.batch.bincount()

        normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

        node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
        node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

        cost_matrices = compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b2, batch2)
        padded_cost_matrices = pad_cost_matrices(cost_matrices, max_graph_size)

        B, _, _ = padded_cost_matrices.shape
        masks = generate_attention_masks(B, batch1, batch2, max_graph_size).to(device)
        masked_cost_matrices = attention_layer(padded_cost_matrices, masks)

        # soft_assignment = alpha_layer()
        # soft_assignments = soft_assignment.unsqueeze(0).repeat(B, 1, 1)
        soft_assignments, alpha_weights = alpha_layer(graph_repr_b1, graph_repr_b2)

        predicted_ged = criterion(masked_cost_matrices, soft_assignments)
        normalized_predicted_ged = predicted_ged / normalization_factor

        all_preds.append(normalized_predicted_ged.cpu())
        all_labels.append(ged_labels.cpu())



    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)

    print(f"[Test] RMSE: {rmse:.4f}")


def train_triplet_encoder(loader, encoder, device, epochs=11):
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    critertion = TripletLoss(margin=0.2)

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

            loss = critertion(a_graph_emb, p_graph_emb, n_graph_emb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * anchor_graphs.y.size(0)
            total_samples += anchor_graphs.y.size(0)

        average_loss = total_loss / total_samples
        
        if epoch % 10 == 0:
            print(f"[Triplet Stage] Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}")
    
    return encoder


def train_ged(loader, encoder, alpha_layer, device, max_graph_size, epochs=11):
    encoder.eval()
    alpha_layer.train()
    
    attention_layer = LearnablePaddingAttention(max_graph_size).to(device)
    attention_layer.train()

    optimizer = torch.optim.Adam(
        list(alpha_layer.parameters()) + list(attention_layer.parameters()), 
        lr=1e-3, 
        weight_decay=1e-5
    )
    
    criterion = SoftGEDLoss()

    lambda_self = 0.2

    for epoch in range(epochs):

        total_loss = 0
        total_samples = 0

        for batch in loader:

            batch1, batch2, ged_labels = batch

            batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

            n_nodes_1 = batch1.batch.bincount()
            n_nodes_2 = batch2.batch.bincount()

            normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

            optimizer.zero_grad()

            with torch.no_grad():
                node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)    # [n1, D]
                node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)    # [n2, D]
            
            # cost_matrices = compute_cost_matrix(dense_b1, dense_b2)      # [B, N1, N2]
            cost_matrices = compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b2, batch2)

            # padded_cost = pad_cost_matrix(cost_matrices, max_graph_size) # [B, maxN, maxN]
            padded_cost_matrices = pad_cost_matrices(cost_matrices, max_graph_size)

            # Get the mask for padding (1 for valid, 0 for padded)
            B, _, _ = padded_cost_matrices.shape
            masks = generate_attention_masks(B, batch1, batch2, max_graph_size).to(device)
            
            # Apply learnable attention to focus on the real parts of the cost matrix
            masked_cost_matrices = attention_layer(padded_cost_matrices, masks)

            # Soft assignment via learnable alpha-weighted permutation matrices
            soft_assignments, alphas = alpha_layer(graph_repr_b1, graph_repr_b2)

            # Repeat assignment matrix across batch
            # soft_assignments = soft_assignment.unsqueeze(0).repeat(B, 1, 1)

            predicted_ged = criterion(masked_cost_matrices, soft_assignments) # (B,)
            normalized_predicted_ged = predicted_ged / normalization_factor

            # Enforce GED(g, g) = 0
            # with torch.no_grad():
            #     self_cost_marices = compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b1, batch1)
            
            # padded_self_cost_matrices = pad_cost_matrices(self_cost_marices, max_graph_size)
            # self_masks = generate_attention_masks(B, batch1, batch1, max_graph_size).to(device)
            # masked_self_cost_matrices = attention_layer(padded_self_cost_matrices, self_masks)

            # Compute predicted GED(g, g)
            # ged_self = criterion(masked_self_cost_matrices, soft_assignments)  # (B,)
            # normalized_ged_self = ged_self / normalization_factor

            loss = F.mse_loss(normalized_predicted_ged, ged_labels, reduction='mean') \
                #  + F.mse_loss(normalized_ged_self, torch.zeros_like(ged_self), reduction='mean') * lambda_self

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch1.y.size(0)
            total_samples += batch1.y.size(0)
        
        average_loss = total_loss / total_samples
        
        if epoch % 10 == 0:
            print(f"[GED] Epoch {epoch+1}/{epochs} - MSE: {average_loss:.4f} - RMSE: {np.sqrt(average_loss):.1f}")


def main():

    distance_matrix = np.load('./data/distances_alpha_-1.0.npy')

    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset from TUDataset
    dataset = TUDataset(root='data', name='MUTAG')
    ged_labels = get_ged_labels(distance_matrix)
    sizes = get_cost_matrices_distr(dataset)

    # Train/Test split
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    test_len = total_len - train_len

    generator = torch.Generator().manual_seed(42)
    train_data, test_data = random_split(dataset, [train_len, test_len], generator=generator)

    # Prepare DataLoader
    triplet_train = TripletDataset(dataset, train_data.indices, ged_labels)
    siamese_train = SiameseDataset(dataset, train_data.indices, ged_labels)
    siamese_test = SiameseDataset(dataset, test_data.indices, ged_labels)

    triplet_loader = DataLoader(triplet_train, batch_size=64, shuffle=True)
    siamese_loader = DataLoader(siamese_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(siamese_test, batch_size=64, shuffle=False)

    # Model
    encoder = Model(dataset.num_features, 64, 3).to(device)

    encoder = train_triplet_encoder(triplet_loader, encoder, device)
    encoder.freeze_params(encoder) # you should not only freeze, but checkpoint the model as well

    max_graph_size = max([g.num_nodes for g in dataset])
    k = (max_graph_size - 1) ** 2 + 1 # upper (theoretical) bound
    k = 50

    perm_pool = PermutationPool(max_n=max_graph_size, k=k, size_data=sizes)

    alpha_layer = AlphaPermutationLayer(128, perm_pool).to(device)

    train_ged(siamese_loader, encoder, alpha_layer, device, max_graph_size)

    # evaluate_ged(test_loader, encoder, alpha_layer, device, max_graph_size)


if __name__ == '__main__':
    main()