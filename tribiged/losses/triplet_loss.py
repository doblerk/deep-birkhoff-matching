import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # anchor = F.normalize(anchor, p=2, dim=1)
        # positive = F.normalize(positive, p=2, dim=1)
        # negative = F.normalize(negative, p=2, dim=1)
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        return F.relu(pos_dist - neg_dist + self.margin).mean()