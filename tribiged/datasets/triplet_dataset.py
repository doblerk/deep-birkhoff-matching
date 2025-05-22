import random

from torch.utils.data import Dataset


class TripletDataset(Dataset):

    def __init__(self, graphs, indices, ged_dict, k):
        super(TripletDataset, self).__init__()
        self.graphs = graphs
        self.indices = [int(i) for i in indices]
        self.ged_dict = ged_dict
        self.k = k

        # Precompute sorted neighbors by GED
        self.sorted_neighbors = {
            i: sorted(
                [(j, self._get_ged(i, j)) for j in self.indices if j != i],
                key=lambda x: x[1]
            )
            for i in self.indices
        }
    
    def __len__(self):
        return len(self.indices)
    
    def _get_ged(self, i, j):
        return self.ged_dict.get((i, j), self.ged_dict.get((j, i), 0.0))
  
    def __getitem__(self, idx):
        anchor_idx = self.indices[idx]
        anchor_graph = self.graphs[anchor_idx]

        neighbors = self.sorted_neighbors[anchor_idx]

        # Hard positive = sample one of the top-k closest
        pos_candidates = neighbors[:self.k]
        pos_graph_idx, _ = random.choice(pos_candidates)

        # Hard negative: sample one of the bottom-k farthest
        neg_candidates = neighbors[-self.k:]
        neg_graph_idx, _ = random.choice(neg_candidates)

        pos_graph = self.graphs[pos_graph_idx]
        neg_graph = self.graphs[neg_graph_idx]

        return anchor_graph, pos_graph, neg_graph