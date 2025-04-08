import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from utils import get_ged_labels

from model import Model

from data_aug import GraphPairDataset, TripletGraphDataset

from diff_birkhoff import compute_cost_matrix, \
                          pad_cost_matrix, \
                          PermutationMatrix, \
                          ContrastiveLoss, \
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


def train_model(dataloader, device, optimizer, model, criterion):
     for epoch in range(1):
          train(dataloader, device, optimizer, model, criterion)


def train_triplet_encoder(loader, encoder, device, epochs=1):
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=0.0001)
    critertion = TripletLoss(margin=1.0)

    for epoch in range(epochs):
        
        total_loss = 0

        for triplet in loader:

            # anchor_info, positive_info, negative_info = triplet

            # graphs = [anchor_info[0], positive_info[0], negative_info[0]]
            # node_idxs = [anchor_info[1], positive_info[1], negative_info[1]]

            # batch_graphs = [loader.dataset.graphs[i.tolist()] for i in graphs]
            # batch = Batch.from_data_list(batch_graphs).to(device) # bug here

            anchor_graphs, pos_graphs, neg_graphs = triplet

            a_batch = Batch.from_data_list(anchor_graphs).to(device)
            p_batch = Batch.from_data_list(pos_graphs).to(device)
            n_batch = Batch.from_data_list(neg_graphs).to(device)

            optimizer.zero_grad()

            # _, graph_embeddings = encoder(batch.x, batch.edge_index, batch.batch)

            _, a_graph_emb = encoder(a_batch.x, a_batch.edge_index, a_batch.batch)
            _, p_graph_emb = encoder(p_batch.x, p_batch.edge_index, p_batch.batch)
            _, n_graph_emb = encoder(n_batch.x, n_batch.edge_index, n_batch.batch)

        #     node_embs, _ = encoder(batch.x, batch.edge_index, batch.batch)

        #     anchor_node = node_embs[node_idxs[0]]
        #     pos_node = node_embs[node_idxs[1]]
        #     neg_node = node_embs[node_idxs[2]]
            loss = critertion(a_graph_emb, p_graph_emb, n_graph_emb) # ensure this is done between each pair batch-wise

        #     loss = critertion(anchor_node, pos_node, neg_node)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Triplet Stage] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    
    return encoder


def train_ged_supervised(loader, encoder, device, epochs=1):
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=0.0001)
    criterion = ContrastiveLoss()

    for epoch in epochs:

        for batch in loader:
            batch1, batch2, ged_labels = batch
            batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                node_representations_b1, _ = encoder(batch1.x, batch1.edge_index, batch1.batch)
                node_representations_b2, _ = encoder(batch2.x, batch2.edge_index, batch2.batch)

            dense_b1, _ = to_dense_batch(node_representations_b1, batch1.batch)
            dense_b2, _ = to_dense_batch(node_representations_b2, batch2.batch)
            
            cost_matrices = compute_cost_matrix(dense_b1, dense_b2)
            padded_cost_matrices = pad_cost_matrix(cost_matrices)

            B, N, _ = padded_cost_matrices.shape
            k_plus_one = N + 1

            pm = PermutationMatrix(B, k_plus_one).to(device)
            perm_matrices = pm.generate_permutation_matrices(B, N, k_plus_one).to(device)

            soft_assignment_matrices = pm(perm_matrices)

            predicted_ged = criterion(padded_cost_matrices, soft_assignment_matrices) # (B,)
            
            loss = torch.nn.functional.mse_loss(predicted_ged, ged_labels, reduction='mean')

            loss.backward()
            optimizer.step()
        
        print(f"[GED] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


def main():

    distance_matrix = np.load('./data/distances_alpha_-1.0.npy')

    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset from TUDataset
    dataset = TUDataset(root='data', name='MUTAG')
    ged_labels = get_ged_labels(distance_matrix)

    # Prepare DataLoader
    triplet_dataset = TripletGraphDataset(dataset, ged_labels, num_triplets=10)
    graph_pair_dataset = GraphPairDataset(dataset, ged_labels)

    triplet_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)
    graph_pair_loader = DataLoader(graph_pair_dataset, batch_size=32, shuffle=True)

    # Model, optimizer, and loss
    encoder = Model(dataset.num_features, 64, 3).to(device)

    encoder = train_triplet_encoder(triplet_loader, encoder, device)
    # encoder.freeze_params()

    # train_ged_supervised(graph_pair_loader, encoder, device)


if __name__ == '__main__':
    main()