import torch
import numpy as np

from itertools import product, combinations

from torch.utils.data import Dataset


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
        
        elif pair_mode == 'cross':
            assert train_indices is not None and test_indices is not None
            self.pairs = list(product(test_indices, train_indices))
        
        elif pair_mode == 'all':
            assert train_indices is not None and test_indices is not None
            total_indices = train_indices + test_indices
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