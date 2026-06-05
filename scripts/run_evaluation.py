import logging
import argparse
from pathlib import Path

import numpy as np

from birkhoffnet.utils.config import load_data
from birkhoffnet.evaluation.knn_classifier import knn_classifier
from birkhoffnet.evaluation.metrics import ranking_metrics


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Path to parameters file')
    return parser


def main(args):

    config, _, ged_data, valid_indices, _, _, test_indices = load_data(args.params)

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

    # true_distances = ged_data["ged_matrix"]

    # valid_idx_map = {orig_idx: i for i, orig_idx in enumerate(valid_indices)}
    # test_indices_orig  = set(test_indices.tolist())
    # test_indices = [valid_idx_map[i] for i in test_indices_orig]

    knn_classifier(config, distances, valid_indices)
    # rho, tau, pks = ranking_metrics(distances, true_distances, test_indices)
    # print(rho, ' ', tau, ' ', pks)
    # ged = torch.load("./res/new_analysis/data/AIDS/AIDS_ged_matrices.pt")
    # normalization_factor = ged["norm_factor_matrix"]
    # predicted_ged = -normalization_factor * torch.log(torch.tensor(distances))
    # knn_classifier(config, metadata, predicted_ged.numpy())

 
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)