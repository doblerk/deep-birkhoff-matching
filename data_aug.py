import torch
import numpy as np

from torch.utils.data import Dataset


class TripletDataset(Dataset):

    def __init__(self, graphs, ged_labels):
        super(TripletDataset, self).__init__()
        self.graphs = graphs
        self.labels = [g.y.item() for g in graphs]
        self.ged_labels = ged_labels
    
    def __len__(self):
        return len(self.graphs)

    def _get_ged(self, i, j):
        return self.ged_labels.get((i, j), self.ged_labels.get((j, i), 0.0))
  
    def __getitem__(self, anchor_idx):        
        anchor_graph = self.graphs[anchor_idx]
        anchor_label = self.labels[anchor_idx]

        same_class = []
        diff_class = []
        for idx in range(len(self.graphs)):
            if idx == anchor_idx:
                continue
            elif self.labels[idx] == anchor_label:
                same_class.append((idx, self._get_ged(anchor_idx, idx)))
            else:
                diff_class.append((idx, self._get_ged(anchor_idx, idx)))

        # Hard positive = same class, max GED
        pos_graph_idx, _ = max(same_class, key=lambda x: x[1])
        # Hard negative = different class, min GED
        neg_graph_idx, _ = min(diff_class, key=lambda x: x[1])

        pos_graph = self.graphs[pos_graph_idx]
        neg_graph = self.graphs[neg_graph_idx]

        return anchor_graph, pos_graph, neg_graph
        

class SiameseDataset(Dataset):

    def __init__(self, graphs, ged_labels):
        super(SiameseDataset, self).__init__()
        self.graphs = graphs
        self.ged_labels = ged_labels # dictionnary mapping of "ground truth" ged -> constant lookup
    
    def _get_ged(self, i, j):
        return self.ged_labels.get((i, j), self.ged_labels.get((j, i), 0.0))

    def __len__(self):
        return len(self.graphs) 

    def __getitem__(self, idx):
        # Randomly select a second graph
        idx2 = np.random.randint(0, len(self.graphs) - 1)

        # Graphs in batch1 are <= graphs in batch2
        graph1, graph2 = (
            (self.graphs[idx], self.graphs[idx2]) 
            if self.graphs[idx].num_nodes <= self.graphs[idx2].num_nodes
            else (self.graphs[idx2], self.graphs[idx]) 
        )

        # Retrieve ground truth GED for this pair
        ged_value = self._get_ged(idx, idx2)

        normalized_ged_value = ged_value / ( (graph1.num_nodes + graph2.num_nodes) / 2 )

        return graph1, graph2, torch.tensor(ged_value, dtype=torch.float)