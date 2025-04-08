import torch
import numpy as np

from torch.utils.data import Dataset


class TripletGraphDataset(Dataset):

    def __init__(self, graphs, ged_labels, num_triplets=5000):
        self.graphs = graphs
        self.ged_labels = ged_labels
        self.num_triplets = num_triplets
        self.triplets = self._generate_triplets()
  
    def _generate_triplets(self):
        triplets = []
        num_graphs = len(self.graphs)

        for _ in range(self.num_triplets):
            anchor_idx = torch.randint(0, num_graphs - 1, (1,)).item()
            anchor_graph = self.graphs[anchor_idx]

            # Rank all other graphs by GED
            candidates = [
            (i, self._get_ged(anchor_idx, i))
            for i in range(num_graphs) if i != anchor_idx
            ]
            candidates.sort(key=lambda x: x[1])

            # Find most similar and dissimlar graphs
            pos_graph_idx = candidates[0][0]
            neg_graph_idx = candidates[-1][0]

            pos_graph = self.graphs[pos_graph_idx]
            neg_graph = self.graphs[neg_graph_idx]

            # sample random node idx
            a_node = torch.randint(0, anchor_graph.x.shape[0] - 1, (1,)).item()
            p_node = torch.randint(0, pos_graph.x.shape[0] - 1, (1,)).item()
            n_node = torch.randint(0, neg_graph.x.shape[0] - 1, (1,)).item()

            triplets.append(((anchor_idx, a_node),
                            (pos_graph_idx, p_node),
                            (neg_graph_idx, n_node)))
        
        return triplets
  
    def _get_ged(self, i, j):
        return self.ged_labels.get((i, j), self.ged_labels.get((j, i), 0.0))

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]


class GraphPairDataset(Dataset):

    def __init__(self, graphs, ged_labels):
        super(GraphPairDataset, self).__init__()
        self.graphs = graphs
        self.ged_labels = ged_labels # dictionnary mapping of "ground truth" ged -> constant lookup

    def __len__(self):
        return len(self.graphs) 

    def __getitem__(self, idx):
        # Randomly select a second graph
        idx2 = np.random.randint(0, len(self.graphs) -1)

        graph1 = self.graphs[idx]
        graph2 = self.graphs[idx2]

        # Retrieve ground truth GED for this pair
        if (idx, idx2) in self.ged_labels:
            ged_value = self.ged_labels.get((idx, idx2))
        elif (idx2, idx) in self.ged_labels:
            ged_value = self.ged_labels.get((idx2, idx))
        elif idx == idx2:
            ged_value = 0.0
        else:
            ged_value = 0.0

        return graph1, graph2, torch.tensor(ged_value, dtype=torch.float)