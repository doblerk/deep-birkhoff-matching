import logging
import argparse
from pathlib import Path

import numpy as np

from birkhoffnet.utils.config import load_data
from birkhoffnet.evaluation.knn_classifier import knn_classifier


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Path to parameters file')
    return parser


def main(args):

    config, _, _, valid_indices, _, _, _ = load_data(args.params)

    output_dir = Path(config.output_dir)

    log_file = output_dir / "log_evaluation.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )

    distances_file = output_dir / "distances.npy"
    distances = np.load(distances_file)

    knn_classifier(config, distances, valid_indices)

    # ged = torch.load("./res/new_analysis/data/AIDS/AIDS_ged_matrices.pt")
    # normalization_factor = ged["norm_factor_matrix"]
    # predicted_ged = -normalization_factor * torch.log(torch.tensor(distances))
    # knn_classifier(config, metadata, predicted_ged.numpy())

 
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)