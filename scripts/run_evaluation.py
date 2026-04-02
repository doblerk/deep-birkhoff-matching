import argparse

# import torch
import numpy as np

from birkhoffnet.utils.config import load_data
from birkhoffnet.evaluation.knn_classifier import knn_classifier


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Path to parameters file')
    parser.add_argument('--distances', type=str, help='Path to ged file')
    return parser


def main(args):

    config, metadata, _ = load_data(args.params)
    distances = np.load(args.distances)

    knn_classifier(config, metadata, distances)

    # ged = torch.load("./res/new_analysis/data/AIDS/AIDS_ged_matrices.pt")
    # normalization_factor = ged["norm_factor_matrix"]
    # predicted_ged = -normalization_factor * torch.log(torch.tensor(distances))
    # knn_classifier(config, metadata, predicted_ged.numpy())

 
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)