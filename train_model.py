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

from diff_birkhoff import compute_cost_matrix, \
                          pad_cost_matrix, \
                          PermutationMatrix, \
                          ContrastiveLoss




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

            B, N, N = padded_cost_matrices.shape
            k_plus_one = N + 1

            pm = PermutationMatrix()
            perm_matrices = pm.generate_permutation_matrices(B, N, k_plus_one).to(device)

            soft_assignment_matrices = pm(perm_matrices)

            predicted_ged = criterion(cost_matrices, soft_assignment_matrices)
            true_ged = get_true_ged()
            
            loss = torch.nn.functional.mse_loss(predicted_ged, true_ged)

            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()