import torch
import numpy as np

import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from utils import get_ged_labels, \
                  compute_cost_matrix, \
                  pad_cost_matrix, \
                  visualize_node_embeddings, \
                  generate_permutation_bank

from model import Model

from data_aug import GraphPairDataset, TripletGraphDataset

from diff_birkhoff import compute_cost_matrix, \
                          pad_cost_matrix, \
                          PermutationMatrix, \
                          SoftGEDLoss, \
                          TripletLoss


@torch.no_grad()
def test(test_loader, device, model, criterion):
    model.eval()
    history = []
    for batch in test_loader:
        batch1, batch2, ged_labels = batch
        batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)
        
        node_representations_b1 = model(batch1.x, batch1.edge_index)
        node_representations_b2 = model(batch2.x, batch2.edge_index)

        dense_b1, mask1 = to_dense_batch(node_representations_b1, batch1.batch)
        dense_b2, mask2 = to_dense_batch(node_representations_b2, batch2.batch)

        cost_matrices = compute_cost_matrix(dense_b1, dense_b2)

        padded_cost_matrices = pad_cost_matrix(cost_matrices)

        B, N, N = padded_cost_matrices.shape
        k_plus_one = N + 1

        pm = PermutationMatrix()
        perm_matrices = pm.generate_permutation_matrices(B, N, k_plus_one).to(device)

        soft_assignment_matrices = pm(perm_matrices)

        predicted_ged = criterion(padded_cost_matrices, soft_assignment_matrices)

        true_ged = torch.sum(ged_labels)
            
        loss = torch.nn.functional.mse_loss(predicted_ged, true_ged)

        history.append(loss)

    return history


def train_triplet_encoder(loader, encoder, device, epochs=101):
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=0.0001)
    critertion = TripletLoss(margin=0.2)

    for epoch in range(epochs):

        total_loss = 0

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

        average_loss = total_loss / len(loader.dataset)
        print(f"[Triplet Stage] Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}")
    
    return encoder


def train_ged_supervised(loader, encoder, device, epochs=101):
    encoder.eval()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=0.0001)
    criterion = SoftGEDLoss()

    for epoch in range(epochs):

        total_loss = 0

        for batch in loader:

            batch1, batch2, ged_labels = batch

            batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                node_repr_b1, _ = encoder(batch1.x, batch1.edge_index, batch1.batch)
                node_repr_b2, _ = encoder(batch2.x, batch2.edge_index, batch2.batch)

            dense_b1, _ = to_dense_batch(node_repr_b1, batch1.batch) # [B, N1, D]
            dense_b2, _ = to_dense_batch(node_repr_b2, batch2.batch) # [B, N2, D]
            
            cost_matrices = compute_cost_matrix(dense_b1, dense_b2)  # [B, N, N]
            padded_cost = pad_cost_matrix(cost_matrices)    # [B, N, N]

            B, N, _ = padded_cost.shape
            k_plus_one = N + 1 # Carathéodory theorem

            pm = PermutationMatrix(B, k_plus_one).to(device)
            perms = pm.generate_permutation_matrices(B, N, k_plus_one).to(device)
            soft_assignments = pm(perms)

            predicted_ged = criterion(padded_cost, soft_assignments) # (B,)
            
            loss = F.mse_loss(predicted_ged, ged_labels, reduction='mean')

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch1.y.size(0)
        
        average_loss = total_loss / len(loader.dataset)
        print(f"[GED] Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}")


def main():

    distance_matrix = np.load('./data/distances_alpha_-1.0.npy')

    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset from TUDataset
    dataset = TUDataset(root='data', name='MUTAG')
    ged_labels = get_ged_labels(distance_matrix)

    # Prepare DataLoader
    triplet_dataset = TripletGraphDataset(dataset, ged_labels)
    graph_pair_dataset = GraphPairDataset(dataset, ged_labels)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    triplet_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)
    graph_pair_loader = DataLoader(graph_pair_dataset, batch_size=32, shuffle=True)

    # Model
    encoder = Model(dataset.num_features, 64, 3).to(device)

    encoder = train_triplet_encoder(triplet_loader, encoder, device)
    encoder.freeze_params(encoder) # you should not only freeze, but checkpoint the model as well

    max_graph_size = max([g.num_node for g in dataset])
    k_plus_one = max_graph_size ** 2 + 1 # lower bound (theoretical anchor)

    perm_vectors = generate_permutation_bank(max_graph_size, k_plus_one)

    # train_ged_supervised(graph_pair_loader, encoder, device)


if __name__ == '__main__':
    main()