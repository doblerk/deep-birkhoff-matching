import torch
import numpy as np

from torch.utils.data import Dataset


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