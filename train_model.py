import torch
import numpy as np
from time import time
import torch.nn.functional as F

from torch.utils.data import random_split, ConcatDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, GEDDataset

from utils import get_ged_labels, \
                  triplet_collate_fn, \
                  compute_cost_matrix, \
                  compute_graphwise_node_distances, \
                  pad_cost_matrices, \
                  get_node_masks, \
                  generate_attention_masks, \
                  get_cost_matrix_sizes, \
                  get_sampled_cost_matrix_sizes, \
                  ged_matrix_to_dict, \
                  compute_rank_correlations, \
                  plot_querry_vs_closest, \
                  save_model, \
                  compute_entropy, \
                  visualize_node_embeddings, plot_attention, plot_ged, knn_classifier, plot_assignments

from model import Model

from data_aug import TripletDataset, TripletNoLabelDataset, SiameseDataset, SiameseNoLabelDataset#, SiameseTestDataset, SiameseEvalDataset

from diff_birkhoff import PermutationPool, \
                          AlphaPermutationLayer, \
                          LearnablePaddingAttention, \
                          SoftGEDLoss, \
                          TripletLoss


def train_triplet_network(loader, encoder, device, epochs=501):
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = TripletLoss(margin=0.2)
    
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

            # total_loss += loss.item() * anchor_graphs.y.size(0)
            # total_samples += anchor_graphs.y.size(0)
            total_loss += loss.item() * anchor_graphs.i.size(0)
            total_samples += anchor_graphs.i.size(0)

        average_loss = total_loss / total_samples
        
        if epoch % 10 == 0:
            print(f"[Triplet Stage] Epoch {epoch+1}/{epochs} - Loss: {average_loss:.4f}")
    
    torch.save({
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, './res/AIDS/checkpoint_encoder.pth')

    return encoder


@torch.no_grad()
def extract_ged(loader, encoder, alpha_layer, criterion, device, max_graph_size, num_graphs):
    encoder.eval()
    alpha_layer.eval()
    criterion.eval()

    distance_matrix = torch.zeros((num_graphs, num_graphs), dtype=torch.float32, device=device)
    
    t0 = time()

    for batch in loader:

        batch1, batch2, ged_labels, idx1, idx2 = batch
        batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

        node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
        node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

        cost_matrices = compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b2, batch2)
        padded_cost_matrices = pad_cost_matrices(cost_matrices, max_graph_size)

        soft_assignments, alphas = alpha_layer(graph_repr_b1, graph_repr_b2)

        row_masks = get_node_masks(batch1, max_graph_size).to(soft_assignments.device)
        col_masks = get_node_masks(batch2, max_graph_size).to(soft_assignments.device)

        assignment_mask = row_masks.unsqueeze(2) * col_masks.unsqueeze(1)

        soft_assignments = soft_assignments * assignment_mask

        row_sums = soft_assignments.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_assignments = soft_assignments / row_sums
        
        predicted_ged = criterion(padded_cost_matrices, soft_assignments)

        distance_matrix[idx1, idx2] = predicted_ged
    
    t1 = time()
    runtime = t1 - t0

    distance_matrix = torch.maximum(distance_matrix, distance_matrix.T)

    return distance_matrix.cpu().numpy(), runtime


def train_ged(train_loader, encoder, alpha_layer, criterion, optimizer, device, max_graph_size):
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
            node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)    # [n1, D]
            node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)    # [n2, D]
        
        cost_matrices = compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b2, batch2)

        padded_cost_matrices = pad_cost_matrices(cost_matrices, max_graph_size)

        # Soft assignment via learnable alpha-weighted permutation matrices
        soft_assignments, alphas = alpha_layer(graph_repr_b1, graph_repr_b2)
        
        # Mask padded regions
        row_masks = get_node_masks(batch1, max_graph_size).to(soft_assignments.device)
        col_masks = get_node_masks(batch2, max_graph_size).to(soft_assignments.device)

        # (B, maxN, 1) * (B, 1, maxN) -> broadcast to (B, maxN, maxN)
        assignment_mask = row_masks.unsqueeze(2) * col_masks.unsqueeze(1)

        soft_assignments = soft_assignments * assignment_mask

        predicted_ged = criterion(padded_cost_matrices, soft_assignments) # (B,)
        normalized_predicted_ged = predicted_ged / normalization_factor

        loss = F.mse_loss(normalized_predicted_ged, ged_labels, reduction='mean')

        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_ged(val_loader, encoder, alpha_layer, criterion, device, max_graph_size):
    alpha_layer.eval()
    criterion.eval()

    val_loss = 0
    val_samples = 0

    for batch in val_loader:

        batch1, batch2, ged_labels = batch
        batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

        n_nodes_1 = batch1.batch.bincount()
        n_nodes_2 = batch2.batch.bincount()

        normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

        node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
        node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

        cost_matrices = compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b2, batch2)

        padded_cost_matrices = pad_cost_matrices(cost_matrices, max_graph_size)

        soft_assignments, alphas = alpha_layer(graph_repr_b1, graph_repr_b2)

        row_masks = get_node_masks(batch1, max_graph_size).to(soft_assignments.device)
        col_masks = get_node_masks(batch2, max_graph_size).to(soft_assignments.device)

        assignment_mask = row_masks.unsqueeze(2) * col_masks.unsqueeze(1)

        soft_assignments = soft_assignments * assignment_mask

        predicted_ged = criterion(padded_cost_matrices, soft_assignments)
        normalized_predicted_ged = predicted_ged / normalization_factor

        loss = F.mse_loss(normalized_predicted_ged, ged_labels, reduction='mean')

        val_loss += loss.item() * ged_labels.size(0)
        val_samples += ged_labels.size(0)
    
    average_val_loss = val_loss / val_samples

    return average_val_loss


@torch.no_grad()
def test_ged(test_loader, encoder, alpha_layer, criterion, device, max_graph_size):
    alpha_layer.eval()
    criterion.eval()

    test_loss = 0
    test_samples = 0

    for batch in test_loader:

        batch1, batch2, ged_labels = batch
        batch1, batch2, ged_labels = batch1.to(device), batch2.to(device), ged_labels.to(device)

        n_nodes_1 = batch1.batch.bincount()
        n_nodes_2 = batch2.batch.bincount()

        normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

        node_repr_b1, graph_repr_b1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
        node_repr_b2, graph_repr_b2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

        cost_matrices = compute_graphwise_node_distances(node_repr_b1, batch1, node_repr_b2, batch2)
        padded_cost_matrices = pad_cost_matrices(cost_matrices, max_graph_size)

        soft_assignments, alphas = alpha_layer(graph_repr_b1, graph_repr_b2)

        row_masks = get_node_masks(batch1, max_graph_size).to(soft_assignments.device)
        col_masks = get_node_masks(batch2, max_graph_size).to(soft_assignments.device)

        assignment_mask = row_masks.unsqueeze(2) * col_masks.unsqueeze(1)

        soft_assignments = soft_assignments * assignment_mask

        predicted_ged = criterion(padded_cost_matrices, soft_assignments)
        normalized_predicted_ged = predicted_ged / normalization_factor

        loss = F.mse_loss(normalized_predicted_ged, ged_labels, reduction='mean')

        test_loss += loss.item() * ged_labels.size(0)
        test_samples += ged_labels.size(0)

    average_test_loss = test_loss / test_samples

    return average_test_loss


def train_siamese_network(train_loader, val_loader, test_loader, encoder, alpha_layer, criterion, device, max_graph_size, epochs=1001):
    encoder.eval()

    optimizer = torch.optim.Adam(
        list(alpha_layer.parameters()) + list(criterion.parameters()),
        lr=1e-3, 
        weight_decay=1e-5
    )
    
    for epoch in range(epochs):

        train_ged(train_loader, encoder, alpha_layer, criterion, optimizer, device, max_graph_size)

        if epoch % 1 == 0:
            average_val_loss = eval_ged(val_loader, encoder, alpha_layer, criterion, device, max_graph_size)
            print(f"[GED] Epoch {epoch+1}/{epochs} - Val MSE: {average_val_loss:.4f} - RMSE: {np.sqrt(average_val_loss):.1f} - Scale: {criterion.scale.item():.4f}")

    average_test_loss = test_ged(test_loader, encoder, alpha_layer,criterion, device, max_graph_size)
    print(f"[GED] Final Epoch - Test MSE: {average_test_loss:.4f} - RMSE: {np.sqrt(average_test_loss):.1f} - Scale: {criterion.scale.item():.4f}")

    torch.save({
        'alpha_layer': alpha_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
    }, './res/AIDS/checkpoint_ged.pth')


def main():

    train_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=True)
    test_dataset = GEDDataset(root='data/datasets/AIDS700nef', name='AIDS700nef', train=False)

    dataset = ConcatDataset([train_dataset, test_dataset])

    num_features = train_dataset.num_features

    ged_matrix, norm_ged_matrix = train_dataset.ged, train_dataset.norm_ged

    train_size = int(0.75 * len(train_dataset)) # 420
    val_size = len(train_dataset) - train_size # 140

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
    train_dataset_indices, val_dataset_indices = sorted(train_dataset.indices), sorted(val_dataset.indices)

    ged_dict = ged_matrix_to_dict(ged_matrix)
    # norm_ged_dict = ged_matrix_to_dict(norm_ged_matrix)

    # norm = np.zeros_like(ged_matrix)
    # for i in range(norm.shape[0]):
    #     for j in range(i+1, norm.shape[1]):
    #         norm[i,j] = (0.5 * (dataset[i].num_nodes + dataset[j].num_nodes))
    # norm += norm.T
    # np.fill_diagonal(norm, val=1.0)

    # distance_matrix = np.load(f'/home/dobleraemon/Documents/PhD/compute-ged/results/AIDS/distances_alpha_-1.0.npy')

    # norm_distance_matrix = distance_matrix / norm

    # sub_distance_matrix = torch.tensor(norm_distance_matrix[np.ix_(test_dataset.i, train_dataset_indices)].flatten())
    # sub_distance_ged = norm_ged_matrix[np.ix_(test_dataset.i, train_dataset_indices)].flatten()

    # print(F.mse_loss(sub_distance_matrix, sub_distance_ged, reduction='mean'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sizes = get_sampled_cost_matrix_sizes(dataset)

    triplet_train = TripletNoLabelDataset(dataset, train_dataset_indices, ged_dict, k=100)
    triplet_loader = DataLoader(triplet_train, batch_size=64 * 12, shuffle=True, num_workers=2)

    siamese_train = SiameseNoLabelDataset(dataset, norm_ged_matrix, pair_mode='train', train_indices=train_dataset_indices)
    siamese_val = SiameseNoLabelDataset(dataset, norm_ged_matrix, pair_mode='val', train_indices=train_dataset_indices, val_indices=val_dataset_indices)
    siamese_test = SiameseNoLabelDataset(dataset, norm_ged_matrix, pair_mode='test', train_indices=train_dataset_indices, test_indices=test_dataset.i)
    siamese_all = SiameseNoLabelDataset(dataset, norm_ged_matrix, pair_mode='all', train_indices=train_dataset_indices, test_indices=test_dataset.i)

    siamese_train_loader = DataLoader(siamese_train, batch_size=64 * 4, shuffle=True, num_workers=2)
    siamese_val_loader = DataLoader(siamese_val, batch_size=64 * 4, shuffle=False, num_workers=2)
    siamese_test_loader = DataLoader(siamese_test, batch_size=64 * 12, shuffle=False, num_workers=2)
    siamese_all_loader = DataLoader(siamese_all, batch_size=64 * 32, shuffle=False, num_workers=2)

    embedding_dim = 64
    encoder = Model(num_features, embedding_dim, 3).to(device)

    encoder = train_triplet_network(triplet_loader, encoder, device)
    encoder.freeze_params(encoder) # you should not only freeze, but checkpoint the model as well

    max_graph_size = max([g.num_nodes for g in dataset])
    k = (max_graph_size - 1) ** 2 + 1 # upper (theoretical) bound
    k = 21

    perm_pool = PermutationPool(max_n=max_graph_size, k=k)
    perm_matrices = perm_pool.get_matrix_batch().to(device)

    alpha_layer = AlphaPermutationLayer(perm_pool, perm_matrices, embedding_dim, 64 * 4).to(device)

    criterion = SoftGEDLoss().to(device)

    train_siamese_network(siamese_train_loader, siamese_val_loader, siamese_test_loader, encoder, alpha_layer, criterion, device, max_graph_size)

    pred_geds, runtime = extract_ged(siamese_all_loader, encoder, alpha_layer, criterion, device, max_graph_size, len(dataset))

    print('Runtime: ', runtime)

    with open('./res/AIDS/pred_geds.npy', 'wb') as file:
        np.save(file, pred_geds)
    
    pred_matrix = pred_geds[560:, :560]
    # pred_matrix = distance_matrix[560:, :560]
    true_matrix = ged_matrix[560:, :560]

    rho, tau, p_at_k = compute_rank_correlations(pred_matrix, true_matrix, k=10)

    print(rho)
    print(tau)
    print(p_at_k)



if __name__ == '__main__':
    main()