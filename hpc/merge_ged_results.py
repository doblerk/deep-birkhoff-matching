import re
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from torch_geometric.datasets import TUDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser


def main(args):

    output_dir = Path(args.output_dir)

    # --------------------------------------------------
    # 1. Merge GED results
    # --------------------------------------------------

    files = list(output_dir.glob(f"ged_results_{args.dataset_name}_*.npy"))

    files_sorted = sorted(
        files,
        key=lambda p: int(re.search(r'_(\d+)_', p.name).group(1))
    )

    all_results = []

    for f in files_sorted:
        all_results.extend(np.load(f, allow_pickle=True))

    all_results = np.array(all_results, dtype=object)

    merged_file = output_dir / f"ged_results_{args.dataset_name}_full.npy"
    np.save(merged_file, all_results)

    print(f"Merged GED saved to {merged_file}")

    # --------------------------------------------------
    # 2. Load metadata
    # --------------------------------------------------

    with open(args.metadata) as f:
        metadata = json.load(f)

    valid_indices = metadata["valid_graph_indices"]
    n = metadata["num_graphs_filtered"]

    valid_idx_map = {orig_idx: i for i, orig_idx in enumerate(valid_indices)}

    # --------------------------------------------------
    # 3. Load dataset
    # --------------------------------------------------

    dataset = TUDataset(
        root=args.dataset_dir,
        name=args.dataset_name,
        use_node_attr=False
    )

    node_counts = torch.tensor(
        [dataset[i].num_nodes for i in valid_indices],
        dtype=torch.float32
    )

    # --------------------------------------------------
    # 4. Build GED matrix
    # --------------------------------------------------

    ged_matrix = torch.zeros((n, n), dtype=torch.float32)

    for a, b, ged, _, _ in all_results:
        i = valid_idx_map[a]
        j = valid_idx_map[b]

        ged_matrix[i, j] = ged
        ged_matrix[j, i] = ged
    
    # --------------------------------------------------
    # 5. Normalization factor
    # --------------------------------------------------

    norm_factor_matrix = 0.5 * (
        node_counts[:, None] + node_counts[None, :]
    )

    # --------------------------------------------------
    # 6. Normalized GED similarity
    # --------------------------------------------------

    norm_ged_matrix = torch.exp(-ged_matrix / norm_factor_matrix)

    norm_ged_matrix.fill_diagonal_(1.0)

    # --------------------------------------------------
    # 7. Save everything
    # --------------------------------------------------

    save_file = output_dir / f"{args.dataset_name}_ged_matrices.pt"

    torch.save(
        {
            "ged_matrix": ged_matrix,
            "norm_factor_matrix": norm_factor_matrix,
            "norm_ged_matrix": norm_ged_matrix,
            "node_counts": node_counts,
            "valid_indices": valid_indices
        },
        save_file
    )

    print(f"Saved matrices to {save_file}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)