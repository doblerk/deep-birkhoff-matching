import argparse

import torch
import numpy as np
import torch.nn.functional as F

from time import time

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
from birkhoffnet.models.alpha_layers import AlphaPermutationLayer
from birkhoffnet.utils.data_utils import ged_matrix_to_dict, \
                                      compute_cost_matrices, \
                                      pad_cost_matrices, \
                                      get_node_masks


@torch.no_grad()
def infer_ged(loader, encoder, alpha_layer, criterion, device, max_graph_size, num_graphs):
    encoder.eval()
    alpha_layer.eval()
    criterion.eval()

    distance_matrix = torch.zeros((num_graphs, num_graphs), dtype=torch.float32, device=device)
    
    t0 = time()

    for batch in loader:

        batch1, batch2, _, idx1, idx2 = batch
        batch1, batch2, idx1, idx2 = batch1.to(device), batch2.to(device), idx1.to(device), idx2.to(device)

        n_nodes_1 = batch1.batch.bincount()
        n_nodes_2 = batch2.batch.bincount()

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

        distance_matrix[idx1, idx2] = predicted_ged
    
    t1 = time()
    runtime = t1 - t0
    print('Runtime: ', runtime)

    distance_matrix = torch.maximum(distance_matrix, distance_matrix.T)

    return distance_matrix.cpu().numpy()


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
    norm_ged_matrix = train_dataset.norm_ged

    train_size = int(0.75 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
    train_dataset_indices, val_dataset_indices = sorted(train_dataset.indices), sorted(val_dataset.indices)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding_dim = 64
    encoder = Model(num_features, embedding_dim, 1)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)

    max_graph_size = max([g.num_nodes for g in dataset])
    k = args.k
    
    perm_pool = PermutationPool(max_n=max_graph_size, k=k)
    perm_matrices = perm_pool.get_matrix_batch().to(device)

    alpha_layer = AlphaPermutationLayer(perm_matrices, k, embedding_dim).to(device)

    criterion = GEDLoss()

    ged_optimizer = torch.optim.Adam(
        list(alpha_layer.parameters()) + list(criterion.parameters()),
        lr=1e-3, 
        weight_decay=1e-5
    )

    checkpoint_encoder = torch.load(f'{args.output_dir}/checkpoint_encoder.pth', map_location=device)
    encoder.load_state_dict(checkpoint_encoder['encoder'])
    encoder_optimizer.load_state_dict(checkpoint_encoder['optimizer'])

    encoder = encoder.to(device)

    checkpoint_ged = torch.load(f'{args.output_dir}/checkpoint_ged.pth', map_location=device)
    alpha_layer.load_state_dict(checkpoint_ged['alpha_layer'])
    ged_optimizer.load_state_dict(checkpoint_ged['optimizer'])
    criterion.load_state_dict(checkpoint_ged['criterion'])

    alpha_layer = alpha_layer.to(device)
    criterion = criterion.to(device)

    encoder.eval()
    alpha_layer.eval()
    criterion.eval()

    siamese_all = SiameseDataset(dataset, norm_ged_matrix, pair_mode='all', train_indices=train_dataset_indices, test_indices=test_dataset.i)
    siamese_all_loader = DataLoader(siamese_all, batch_size=64 * 384, shuffle=False, num_workers=10)

    pred_geds = infer_ged(siamese_all_loader, encoder, alpha_layer, criterion, device, max_graph_size, len(dataset))

    with open(f'{args.output_dir}/pred_geds.npy', 'wb') as file:
        np.save(file, pred_geds)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)