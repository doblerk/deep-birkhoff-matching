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

    def __init__(self, graphs, indices, ged_labels):
        super(SiameseDataset, self).__init__()
        self.graphs = graphs
        self.indices = indices
        self.ged_labels = ged_labels # dictionnary mapping of "ground truth" ged -> constant lookup
    
    def _get_ged(self, i, j):
        return self.ged_labels.get((i, j), self.ged_labels.get((j, i), 0.0))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Remap idx to the correct index in graphs
        idx1 = self.indices[idx]

        # Randomly sample a new idx
        idx2 = np.random.choice(self.indices)

        # Graphs in batch1 are <= graphs in batch2
        graph1, graph2 = (
            (self.graphs[idx1], self.graphs[idx2]) 
            if self.graphs[idx1].num_nodes <= self.graphs[idx2].num_nodes
            else (self.graphs[idx2], self.graphs[idx1]) 
        )

        # Retrieve ground truth GED for this pair
        ged_value = self._get_ged(idx1, idx2)
        normalized_ged_value = ged_value / ( 0.5 * (graph1.num_nodes + graph2.num_nodes) )

        return graph1, graph2, torch.tensor(normalized_ged_value, dtype=torch.float)


class SiameseTestDataset(Dataset):

    def __init__(self, graphs, train_indices, test_indices, ged_labels):
        super(SiameseTestDataset, self).__init__()
        self.graphs = graphs
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.ged_labels = ged_labels

        # Generate all (test_idx, train_idx) pairs
        # self.pairs = list(product(test_indices, train_indices)) # deterministic and exhaustive pairs
        self.pairs = list(combinations(range(len(train_indices)+len(test_indices)), r=2))
    
    def _get_ged(self, i, j):
        return self.ged_labels.get((i, j), self.ged_labels.get((j, i), 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        test_idx, train_idx = self.pairs[idx]
        
        graph1, graph2 = (
            (self.graphs[test_idx], self.graphs[train_idx]) 
            if self.graphs[test_idx].num_nodes <= self.graphs[train_idx].num_nodes
            else (self.graphs[train_idx], self.graphs[test_idx]) 
        )

        ged_value = self._get_ged(test_idx, train_idx)
        normalized_ged_value = ged_value / ( 0.5 * (graph1.num_nodes + graph2.num_nodes) )

        return graph1, graph2, torch.tensor(normalized_ged_value, dtype=torch.float), test_idx, train_idx