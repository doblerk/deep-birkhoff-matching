import torch
import numpy as np
import torch.nn as nn

from itertools import permutations

from torch.utils.data import Dataset
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset


"""
  THIS SCRIPT CONTAINS PREVIOUSLY WRITTEN VERSIONS OF THE CODE.
"""



# def calc_cost_matrix(emb1, emb2, max_size):
#   n1, n2 = emb1.size(0), emb2.size(0)
#   cost_matrix = torch.cdist(emb1, emb2, p=2)
#   square_cost_matrix = torch.ones((max_size, max_size), device=cost_matrix.device)
#   square_cost_matrix[:n1, :n2] = cost_matrix
#   return square_cost_matrix

def process_batch(node_emb1, mask1, node_emb2, mask2):
  batch_size = node_emb1.shape[0]

  # determine max graph size in the batch
  max_nodes1 = mask1.sum(dim=1).max().item()  # Max valid nodes in batch1
  max_nodes2 = mask2.sum(dim=1).max().item()  # Max valid nodes in batch2
  max_size = max(max_nodes1, max_nodes2)  # Uniform padding size

  # initialize a tensor to hold all cost matrices
  cost_matrices = torch.zeros((batch_size, max_size, max_size), device=node_emb1.device)

  # Compute cost matrices for each graph pair
  for i in range(batch_size):
      emb1 = node_emb1[i][mask1[i]]  # Extract valid nodes for graph i in batch1
      emb2 = node_emb2[i][mask2[i]]  # Extract valid nodes for graph i in batch2
      cost_matrices[i] = calc_cost_matrix(emb1, emb2, max_size)  # Pad to max_size

  return cost_matrices  # Shape: [batch_size, max_size, max_size]


class PermutationMatrix(nn.Module):

  def __init__(self, num_nodes):
    super(PermutationMatrix, self).__init__()

  def generate_permutation_matrices(self, num_nodes, k, batch_size):
    permutation_matrices = []
    while len(permutation_matrices) < k:
      m = np.eye(num_nodes)
      np.random.shuffle(m)
      if not any(np.array_equal(m, existing) for existing in permutation_matrices):
        permutation_matrices.append(torch.from_numpy(m).float())
    permutation_matrices = torch.stack(permutation_matrices)
    expanded_permutation_matrices = permutation_matrices.unsqueeze(0)
    permutation_matrices_batch = expanded_permutation_matrices.repeat(batch_size, 1, 1, 1) # could generate new random PMs
    return permutation_matrices_batch

    # return torch.stack(permutation_matrices)

  def forward(self, num_nodes, batch_size):
    k = num_nodes + 1  # Carathéodory's theorem
    alphas = nn.Parameter(torch.softmax(torch.randn(batch_size, k), dim=1), requires_grad=True)
    permutation_matrices = self.generate_permutation_matrices(num_nodes, k, batch_size)
    assignment_matrices = torch.einsum('bk,bkij->bij', alphas, permutation_matrices)
    return assignment_matrices


def train_ged(loader, encoder, alpha_layer, device, max_graph_size, epochs=501):
    encoder.eval()
    alpha_layer.train()
    
    # attention_layer = LearnablePaddingAttention(max_graph_size).to(device)
    # attention_layer.train()

    optimizer = torch.optim.Adam(
        list(alpha_layer.parameters()), # + list(attention_layer.parameters()), 
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
            # B, _, _ = padded_cost_matrices.shape
            # masks = generate_attention_masks(B, batch1, batch2, max_graph_size).to(device)
            
            # # Apply learnable attention to focus on the real parts of the cost matrix
            # masked_cost_matrices = attention_layer(padded_cost_matrices, masks)

            # Soft assignment via learnable alpha-weighted permutation matrices
            soft_assignments, alphas = alpha_layer(graph_repr_b1, graph_repr_b2)

            row_masks = get_node_masks(batch1, max_graph_size).to(soft_assignments.device)
            col_masks = get_node_masks(batch2, max_graph_size).to(soft_assignments.device)

            # (B, maxN, 1) * (B, 1, maxN) -> broadcast to (B, maxN, maxN)
            assignment_mask = row_masks.unsqueeze(2) * col_masks.unsqueeze(1)

            soft_assignments = soft_assignments * assignment_mask

            # soft_assignments = soft_assignments / (soft_assignments.sum(dim=-1, keepdim=True) + 1e-8)

            # print(soft_assignments.sum(dim=-1))

            # Repeat assignment matrix across batch
            # soft_assignments = soft_assignment.unsqueeze(0).repeat(B, 1, 1)

            predicted_ged = criterion(padded_cost_matrices, soft_assignments) # (B,)
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

            loss = F.mse_loss(normalized_predicted_ged, ged_labels, reduction='mean') #\
                #  + F.mse_loss(normalized_ged_self, torch.zeros_like(ged_self), reduction='mean') * lambda_self

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch1.y.size(0)
            total_samples += batch1.y.size(0)
        
        average_loss = total_loss / total_samples
        
        if epoch % 10 == 0:
            print(f"[GED] Epoch {epoch+1}/{epochs} - MSE: {average_loss:.4f} - RMSE: {np.sqrt(average_loss):.1f}")


# class AlphaPermutationLayer(nn.Module):
   
#     def __init__(self, input_dim, perm_pool: PermutationPool):
#         super(AlphaPermutationLayer, self).__init__()
#         self.perm_pool = perm_pool
#         self.temperature = 1.0
#         self.k = perm_pool.k

#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.k)
#         )
        
#     def forward(self, z1, z2):
#         perms = self.perm_pool.get_matrix_batch().to(z1.device) # (k, N, N)

#         h = torch.cat([z1, z2], dim=1)
#         alpha_logits = self.mlp(h) # (B, k)
#         alpha = F.softmax(alpha_logits / self.temperature, dim=1)

#         print(alpha)

#         soft_assignments = torch.einsum('bk,kij->bij', alpha, perms)
#         return soft_assignments, alpha


# perms = self.perm_pool.get_matrix_batch().to(self.alpha_logits.device)
# alphas = torch.softmax(self.alpha_logits[:batch_size] / self.temperature, dim=1)
# return torch.einsum('bk,kij->bij', alphas, perms)