import torch
import numpy as np
import torch.nn as nn
from itertools import permutations
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset




https://arxiv.org/pdf/2304.02458
https://www.pragmatic.ml/sparse-sinkhorn-attention/


class GraphPairDataset(Dataset):

  def __init__(self, dataset):
    super(GraphPairDataset, self).__init__()
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    graph1 = self.dataset[idx]

    # idx2 = np.random.randint(0, len(self.dataset) - 1)
    idx2 = np.random.choice([i for i in range(len(self.dataset)) if i != idx])
    graph2 = self.dataset[idx2]

    return graph1, graph2
    


class Model(nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim):
    super(Model, self).__init__()
    self.conv1 = GCNConv(input_dim, hidden_dim)
    self.conv2 = GCNConv(hidden_dim, hidden_dim)
    self.conv3 = GCNConv(hidden_dim, output_dim)

  def forward(self, x, edge_index, batch):
    x = self.conv1(x, edge_index)
    x = torch.relu(x)
    x = self.conv2(x, edge_index)
    x = torch.relu(x)
    x = self.conv3(x, edge_index)
    # x, _ = to_dense_batch(x, batch)
    return x

def calc_cost_matrix(emb1, emb2, max_size):
  n1, n2 = emb1.size(0), emb2.size(0)
  cost_matrix = torch.cdist(emb1, emb2, p=2)
  square_cost_matrix = torch.ones((max_size, max_size), device=cost_matrix.device)
  square_cost_matrix[:n1, :n2] = cost_matrix
  return square_cost_matrix

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


class ContrastiveLoss(nn.Module):

  def __init__(self):
    super(ContrastiveLoss, self).__init__()

  def forward(self, cost_matrix, assignment_matrix):
    return torch.einsum('bij,bij->', cost_matrix, assignment_matrix) # element-wise multiplication, followed by a summation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TUDataset(root='data', name='MUTAG')

# Prepare DataLoader
graph_pair_dataset = GraphPairDataset(dataset)
dataloader = DataLoader(graph_pair_dataset, batch_size=64, shuffle=True)

# Model, optimizer, and loss
model = Model(dataset.num_features, 64, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
criterion = ContrastiveLoss()

for epoch in range(101):
    model.train()
    for batch in dataloader:
        batch1, batch2 = batch
        batch1, batch2 = batch1.to(device), batch2.to(device)

        optimizer.zero_grad()

        embeddings1 = model(batch1.x, batch1.edge_index, batch1.batch)
        embeddings2 = model(batch2.x, batch2.edge_index, batch2.batch)

        node_emb1, mask1 = to_dense_batch(embeddings1, batch1.batch)
        node_emb2, mask2 = to_dense_batch(embeddings2, batch2.batch)

        cost_matrices = process_batch(node_emb1, mask1, node_emb2, mask2)
        num_nodes, batch_size = cost_matrices.shape[1], cost_matrices.shape[0]
        permutation_matrices = PermutationMatrix(num_nodes).to(device)
        assignment_matrices = permutation_matrices(num_nodes, batch_size)

        loss = criterion(cost_matrices, assignment_matrices)
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
      print(f"Epoch {epoch}, Loss: {loss.item()}")

