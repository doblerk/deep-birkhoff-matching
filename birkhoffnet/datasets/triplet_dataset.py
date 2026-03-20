import random

import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):

    def __init__(self, graphs, indices, ged_matrix, k_pos, k_neg):
        
        super(TripletDataset, self).__init__()
        
        self.graphs = graphs
        self.indices = indices
        # self.ged_dict = ged_dict
        self.ged_matrix = ged_matrix
        self.k_pos = k_pos
        self.k_neg = k_neg

        # Precompute sorted neighbors by GED
        # self.sorted_neighbors = {
        #     i: sorted(
        #         [(j, self._get_ged(i, j)) for j in self.indices if j != i],
        #         key=lambda x: x[1],
        #         reverse=True # True for GED range (0, 1], False otherwise
        #     )
        #     for i in self.indices
        # }

        # Precompute sorted neighbors by GED
        self.sorted_neighbors = self._compute_sorted_neighbors()
    
    def __len__(self):
        return len(self.indices)
    
    def _compute_sorted_neighbors(self):
        
        sorted_neighbors = {}
        
        idx_tensor = torch.tensor(self.indices)

        for i in self.indices:

            sims = self.ged_matrix[i, idx_tensor]

            order = torch.argsort(sims, descending=True, stable=True) # [1.0, ..., 0.0] larger = closer

            neighbors = idx_tensor[order].tolist()

            # remove self
            neighbors = [j for j in neighbors if j != i]

            sorted_neighbors[i] = neighbors

        return sorted_neighbors
    
    def _get_ged(self, i, j):
        return self.ged_dict.get((i, j), self.ged_dict.get((j, i), 1.0)) # 1.0 for GED range (0, 1], 0.0 otherwise
  
    def __getitem__(self, idx):
        anchor_idx = self.indices[idx]
        # anchor_graph = self.graphs[anchor_idx]

        neighbors = self.sorted_neighbors[anchor_idx]

        # Hard positive = sample one of the top-k closest
        pos_candidates = neighbors[:self.k_pos]
        pos_graph_idx = random.choice(pos_candidates)

        # Hard negative: sample one of the bottom-k farthest
        neg_candidates = neighbors[-self.k_neg:]
        neg_graph_idx = random.choice(neg_candidates)

        # pos_graph = self.graphs[pos_graph_idx]
        # neg_graph = self.graphs[neg_graph_idx]
        anchor_graph = self.graphs[anchor_idx].clone()
        pos_graph = self.graphs[pos_graph_idx].clone()
        neg_graph = self.graphs[neg_graph_idx].clone()

        return anchor_graph, pos_graph, neg_graph