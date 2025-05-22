import torch
import random

from itertools import product, combinations

from torch.utils.data import Dataset


class SiameseDataset(Dataset):

    def __init__(
            self,
            graphs,
            norm_ged_matrix,
            pair_mode='train', # 'train', 'val', 'test', 'all'
            train_indices=None,
            val_indices=None,
            test_indices=None,    
    ):
        super(SiameseDataset, self).__init__()
        self.graphs = graphs
        self.norm_ged_matrix = norm_ged_matrix
        self.pair_mode = pair_mode

        if pair_mode == 'train':
            assert train_indices is not None
            self.train_indices = train_indices
            self.pairs = None
        
        elif pair_mode == 'val':
            assert val_indices is not None
            self.val_indices = val_indices
            self.pairs = None
        
        elif pair_mode == 'test':
            assert train_indices is not None and test_indices is not None
            self.pairs = list(product(test_indices, train_indices))
        
        elif pair_mode == 'all':
            assert train_indices is not None and test_indices is not None
            self.pairs = list(combinations(range(len(graphs)), r=2))
        
        else:
            raise ValueError(f'Unknown pair_mode: {pair_mode}')
    
    def _get_ged(self, i, j):
        return self.ged_labels.get((i, j), self.ged_labels.get((j, i), 0.0))
    
    def __len__(self):
        if self.pair_mode == 'train':
            return len(self.train_indices)
        
        elif self.pair_mode == 'val':
            return len(self.val_indices)
        
        else:
            return len(self.pairs)

    def __getitem__(self, idx):
        if self.pair_mode == 'train':
            idx1 = self.train_indices[idx]

            if random.random() < 0.2:
                idx2 = idx1
                g1 = self.graphs[idx1]
                g2 = g1.clone()
                norm_ged = torch.tensor(0.0)
            
            else:
                idx2 = int(random.choice(self.train_indices))
                g1, g2 = self.graphs[idx1], self.graphs[idx2]
                norm_ged = self.norm_ged_matrix[idx1, idx2]
        
        elif self.pair_mode == 'val':
            idx1 = self.val_indices[idx]
            idx2 = int(random.choice(self.val_indices))
            g1, g2 = self.graphs[idx1], self.graphs[idx2]
            norm_ged = self.norm_ged_matrix[idx1, idx2]
        
        else:
            idx1, idx2 = self.pairs[idx]
            g1, g2 = self.graphs[idx1], self.graphs[idx2]
            norm_ged = self.norm_ged_matrix[idx1, idx2]

        # Order graphs consistently
        if g1.num_nodes > g2.num_nodes:
            g1, g2 = g2, g1

        if self.pair_mode == 'all':
            return g1, g2, norm_ged, idx1, idx2
        else:
            return g1, g2, norm_ged