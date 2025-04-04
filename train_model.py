import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from utils import get_ged_labels

from model import Model

from data_aug import GraphPairDataset

from diff_birkhoff import compute_cost_matrix, pad_cost_matrix, ContrastiveLoss




def main():

    distance_matrix = np.load('./data/distances_alpha_-1.0.npy')

    # Set device to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset from TUDataset
    dataset = TUDataset(root='data', name='MUTAG')
    ged_labels = get_ged_labels(distance_matrix)

    # Prepare DataLoader
    graph_pair_dataset = GraphPairDataset(dataset, ged_labels)

    dataloader = DataLoader(graph_pair_dataset, batch_size=32, shuffle=True)

    # Model, optimizer, and loss
    model = Model(dataset.num_features, 64, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    criterion = ContrastiveLoss()

    for epoch in range(1):
        model.train()
        for batch in dataloader:
            batch1, batch2, ged_labels = batch
            batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

            optimizer.zero_grad()

            node_representations1 = model(batch1.x, batch1.edge_index)
            node_representations2 = model(batch2.x, batch2.edge_index)

            representations1, mask1 = to_dense_batch(node_representations1, batch1.batch)
            representations2, mask2 = to_dense_batch(node_representations2, batch2.batch)

            cost_matrices = compute_cost_matrix(representations1, representations2)
            padded_cost_matrices = pad_cost_matrix(cost_matrices)

            # TODO: create a predefined set of permutation matrices
            #       -> our problem is bounded: for 3x3 matrix, there are 6 possibles permutation matrices
            #       -> should we create a set of permutation matrices and vary this number as a hyperparameter?

            # TODO: run the procedure to learn alpha weights of the Birkhoff polytope
            #       -> use a limited number of permutation matrices (Carathéodory's theorem)
            
            # TODO: compute soft assignment matrices
            #       -> torch.einsum('bk,bkij->bij', alphas, permutation_matrices)

            loss = criterion(cost_matrices, soft_assignment_matrices)
            loss.backward()
            optimizer.step()

            cost_matrices = process_batch(node_emb1, mask1, node_emb2, mask2)
            num_nodes, batch_size = cost_matrices.shape[1], cost_matrices.shape[0]
            permutation_matrices = PermutationMatrix(num_nodes).to(device)
            assignment_matrices = permutation_matrices(num_nodes, batch_size)

            loss = criterion(cost_matrices, assignment_matrices)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")









if __name__ == '__main__':
    main()