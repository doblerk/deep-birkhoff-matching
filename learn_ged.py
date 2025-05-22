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


class AlphaPermutationLayer(nn.Module):
   
    def __init__(self, perm_pool: PermutationPool, embedding_dim: int, max_batch_size: int):
        super(AlphaPermutationLayer, self).__init__()
        self.perm_pool = perm_pool
        self.k = perm_pool.k
        self.temperature = 0.8

        # dim = 128
        # self.alpha_mlp = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(2 * embedding_dim, dim * 2),
        #     nn.ReLU(),
        #     nn.LayerNorm(dim * 2),
        #     nn.Dropout(0.2),
        #     nn.Linear(dim * 2, 2 * embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * embedding_dim, self.k)
        # )

        self.alpha_logits = nn.Parameter(torch.randn(max_batch_size, perm_pool.k), requires_grad=True)

    def get_alpha_weights(self):
        return torch.softmax(self.alpha_logits / self.temperature, dim=1)
        
    def forward(self, graph_repr_b1, graph_repr_b2):
        # pair_repr = torch.cat([graph_repr_b1, graph_repr_b2], dim=1) # (B, 2D)
        # alpha_logits = self.alpha_mlp(pair_repr) # (B, k)
        # alphas = F.softmax(alpha_logits / self.temperature, dim=1) # (B, k)
        B = graph_repr_b1.size(0)
        alpha_logits = self.alpha_logits[:B]  # [B, k]
        alphas = F.softmax(alpha_logits / self.temperature, dim=1)
        perms = self.perm_pool.get_matrix_batch().to(graph_repr_b1.device) # (B, maxN, maxN)
        soft_assignments = torch.einsum('bk,kij->bij', alphas, perms)
        return soft_assignments, alphas



#####

knn_classifier(pred_geds, train_data_indices, test_data_indices, dataset_name)

with open(f'./res/{dataset.name}/rmse_loss.txt', 'a') as file:
    file.write(f'Test RMSE: {rmse}\n')

with open(f'./res/{dataset.name}/runtimes.txt', 'a') as file:
    file.write(f'Runtime computation {runtime:.4f} seconds\n')


if idx == 28:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    sns.heatmap(padded_cost_matrices[61].detach().cpu().numpy(), annot=True, ax=axs[0])
    sns.heatmap(soft_assignments[61].detach().cpu().numpy(), annot=True, ax=axs[1])
    plt.tight_layout()
    plt.savefig('./res/MUTAG/graph20_vs_graph628.png', dpi=600, format='png')
    plot_assignments(20, 628, soft_assignments[61].detach().cpu().numpy())


import matplotlib.pyplot as plt
n = 5
fig, axs = plt.subplots(n, 2, figsize=(14,16))
for i in range(n):
    axs[i][0].imshow(padded_cost_matrices[i].detach().cpu().numpy())
    axs[i][1].imshow(soft_assignments[i].detach().cpu().numpy())
plt.tight_layout()
plt.show()
plot_assignments(19, 20, soft_assignments[22].detach().cpu().numpy())


if idx == 28:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(soft_assignments[61].detach().cpu().numpy(), annot=True)
    plt.show()


tud_dataset = TUDataset(root='data', name='AIDS')

tud_dataset_filtered = []

for i in range(len(tud_dataset)):
    g = tud_dataset[i]
    if g.x.shape[0] <= 10:
        tud_dataset_filtered.append(g)

print(len(tud_dataset_filtered))
print(ged_matrix[20, 628])

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(14, 8))
axs[0].imshow(dataset[20].x.numpy())
axs[1].imshow(dataset[628].x.numpy())
plt.show()

for i, batch in enumerate(siamese_all_loader):
    b1, b2, ged, idx1, idx2 = batch
    indices = list(zip(idx1, idx2))
    if (20, 628) in indices:
        print(i, ' ', indices.index((20, 628)))
        break

pred_matrix = np.load('./res/AIDS/test.npy')
print(pred_matrix[19, 20])
print(ged_matrix[19, 20])

print(train_dataset[19].x)
print(train_dataset[20].x)
with open('res/AIDS/test.npy', 'wb') as f:
    np.save(f, pred_geds)

pred_matrix = pred_matrix[:560, :560]
pred_matrix = pred_matrix.flatten().tolist()
pred_matrix_norm = ((pred_matrix - np.min(pred_matrix)) / (np.max(pred_matrix) - np.min(pred_matrix))).squeeze()

true_matrix = norm_ged_matrix[:560, :560]
true_matrix = true_matrix.flatten().tolist()
true_matrix_norm = ((true_matrix - np.min(true_matrix)) / (np.max(true_matrix) - np.min(true_matrix))).squeeze()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(true_matrix_norm, pred_matrix_norm, s=26, alpha=0.75, edgecolor='lightgrey', c='grey', linewidth=0.5)
ax.plot(true_matrix_norm, true_matrix_norm, linestyle='-', linewidth=1, color='black')
ax.set_xlabel('BP-GED', fontsize=22)
ax.set_ylabel('GNN-GED', fontsize=22)
ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.show()


"""
pearman’s Rank Correlation Coefficient
Kendall’s Rank Correlation Coefficient
Precision at peak
https://github.com/Sangs3112/SimGNN/blob/master/README_en.md
https://github.com/yunshengb/SimGNN
https://www.nature.com/articles/s44335-025-00026-4
"""

class PermutationPool:
    def __init__(self, max_n, k, seed: int = 42):
        """
        Args:
            max_n (int): Maximum graph size (i.e., full matrix size: max_n x max_n)
            k (int): Number of permutation matrices to generate
            size_data (np.ndarray): Array of shape (N, 2) containing historical (n, m) size pairs
            seed (int): RNG seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.max_n = max_n
        self.k = k
        # self.size_data = size_data
        # self.kde = gaussian_kde(size_data.T)
        self.perm_vectors = self._generate_permutation_vectors()
    
    def _sample_size(self):
        """Sample a (n, m) pair from the empirical KDE distribution."""
        while True:
            sample = np.round(self.kde.resample(1, seed=self.rng)).astype(int).flatten()
            n, m = sample
            if 1 <= n <= self.max_n and 1 <= m <= self.max_n:
                return n, m

    def _generate_permutation_vectors(self):
        """Generate permutation vectors of max_n length, padded/embedded."""
        perms = []

        identity = np.arange(self.max_n)
        perms.append(tuple(identity))

        for _ in range(self.k - 1):
            # n, m = self._sample_size()
            # perm_len = min(n, m)
            # perm = self.rng.permutation(perm_len)
            # vector = -1 * np.ones(self.max_n, dtype=int)
            # vector[:perm_len] = perm
            # perms.append(tuple(vector))
            perms.append(tuple(self.rng.permutation(self.max_n)))

        return torch.tensor(perms, dtype=torch.long)

class LearnablePaddingAttention(nn.Module):

    def __init__(self, max_graph_size):
        super(LearnablePaddingAttention, self).__init__()
        self.max_graph_size = max_graph_size
        self.attention_logits = nn.Parameter(torch.randn(max_graph_size, max_graph_size))
    
    def forward(self, cost_matrices):
        attention_weights = torch.sigmoid(self.attention_logits)
        attention_weights = attention_weights.unsqueeze(0).to(cost_matrices.device)
        # weighted_cost = cost_matrices * attention_weights
        return cost_matrices * attention_weights


class TripletDataset(Dataset):

    def __init__(self, graphs, indices, ged_labels):
        super(TripletDataset, self).__init__()
        self.graphs = graphs
        self.indices = indices
        self.ged_labels = ged_labels
        self.labels = [g.y.item() for g in graphs]
    
    def __len__(self):
        return len(self.indices)

    def _get_ged(self, i, j):
        return self.ged_labels.get((i, j), self.ged_labels.get((j, i), 0.0))
  
    def __getitem__(self, idx):
        anchor_idx = self.indices[idx]
        anchor_graph = self.graphs[anchor_idx]
        anchor_label = self.labels[anchor_idx]

        same_class = []
        diff_class = []
        for j in self.indices:
            if j == anchor_idx:
                continue
            elif self.labels[j] == anchor_label:
                same_class.append((j, self._get_ged(anchor_idx, j)))
            else:
                diff_class.append((j, self._get_ged(anchor_idx, j)))

        # Hard positive = same class, max GED
        pos_graph_idx, _ = max(same_class, key=lambda x: x[1])
        # Hard negative = different class, min GED
        neg_graph_idx, _ = min(diff_class, key=lambda x: x[1])

        pos_graph = self.graphs[pos_graph_idx]
        neg_graph = self.graphs[neg_graph_idx]

        return anchor_graph, pos_graph, neg_graph


class SiameseDataset(Dataset):

    def __init__(
            self,
            graphs,
            ged_labels,
            pair_mode='train', # 'train', 'cross', 'all'
            train_indices=None,
            test_indices=None,    
    ):
        super(SiameseDataset, self).__init__()
        self.graphs = graphs
        self.ged_labels = ged_labels
        self.pair_mode = pair_mode

        if pair_mode == 'train':
            assert train_indices is not None
            self.train_indices = train_indices
            self.pairs = None
        
        elif pair_mode == 'test':
            assert test_indices is not None
            self.pairs = list(combinations(test_indices, r=2))
        
        elif pair_mode == 'all':
            assert train_indices is not None and test_indices is not None
            total_indices = sorted(train_indices + test_indices)
            self.pairs = list(combinations(total_indices, r=2))
        
        else:
            raise ValueError(f'Unknown pair_mode: {pair_mode}')
    
    def _get_ged(self, i, j):
        return self.ged_labels.get((i, j), self.ged_labels.get((j, i), 0.0))
    
    def __len__(self):
        if self.pair_mode == 'train':
            return len(self.train_indices)
        else:
            return len(self.pairs)

    def __getitem__(self, idx):
        if self.pair_mode == 'train':
            idx1 = self.train_indices[idx]
            idx2 = int(np.random.choice(self.train_indices))
        else:
            idx1, idx2 = self.pairs[idx]
        
        g1, g2 = self.graphs[idx1], self.graphs[idx2]

        # Order graphs consistently
        if g1.num_nodes > g2.num_nodes:
            g1, g2 = g2, g1
        
        ged = self._get_ged(idx1, idx2)
        normalized_ged =  ged / (0.5 * (g1.num_nodes + g2.num_nodes))

        if self.pair_mode == 'all':
            return g1, g2, torch.tensor(normalized_ged, dtype=torch.float), idx1, idx2
        else:
            return g1, g2, torch.tensor(normalized_ged, dtype=torch.float)