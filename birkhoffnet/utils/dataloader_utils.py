import torch
from torch.utils.data import DataLoader
from birkhoffnet.utils.data_utils import ged_matrix_to_dict
from birkhoffnet.datasets.siamese_dataset import SiameseDataset
from birkhoffnet.datasets.triplet_dataset import TripletDataset


class DataLoaders:
    def __init__(self, dataset, train_indices, val_indices, test_indices, ged_matrix, batch_size=64*192):
        norm_ged_matrix = torch.exp(-ged_matrix) # -> range (0, 1] ?
        self.triplet_loader = self._create_triplet_loader(dataset, train_indices, norm_ged_matrix)
        self.train_loader, self.val_loader, self.test_loader = self._create_siamese_loaders(dataset, train_indices, val_indices, test_indices, norm_ged_matrix, batch_size)

    def _create_triplet_loader(self, dataset, train_indices, norm_ged_matrix):
        triplet_train = TripletDataset(dataset, train_indices, ged_matrix_to_dict(norm_ged_matrix), k=int(len(train_indices) * 0.4))
        return DataLoader(triplet_train, batch_size=len(triplet_train), shuffle=True)

    def _create_siamese_loaders(self, dataset, train_indices, val_indices, test_indices, norm_ged_matrix, batch_size):
        siamese_train = SiameseDataset(dataset, norm_ged_matrix, pair_mode='train', train_indices=train_indices)
        siamese_val = SiameseDataset(dataset, norm_ged_matrix, pair_mode='val', train_indices=train_indices, val_indices=val_indices)
        siamese_test = SiameseDataset(dataset, norm_ged_matrix, pair_mode='test', train_indices=train_indices, test_indices=test_indices)
        return (
            DataLoader(siamese_train, batch_size=len(siamese_train), shuffle=True, num_workers=0),
            DataLoader(siamese_val, batch_size=len(siamese_val), shuffle=False, num_workers=0),
            DataLoader(siamese_test, batch_size=batch_size, shuffle=False, num_workers=10)
        )