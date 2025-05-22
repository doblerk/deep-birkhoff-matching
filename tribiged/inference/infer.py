import time
import torch

from tribiged.utils.data_utils import compute_cost_matrices, \
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