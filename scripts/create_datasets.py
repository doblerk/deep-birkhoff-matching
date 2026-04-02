import json
import argparse
import numpy as np
from pathlib import Path
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--use_subset', action='store_true', help="Filter graphs using mean ± std node count")
    return parser


def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    # --- Single pass for stats ---
    node_counts = np.empty(len(dataset), dtype=np.int32)
    edge_counts = np.empty(len(dataset), dtype=np.int32)

    for i, g in enumerate(dataset):
        node_counts[i] = g.num_nodes
        edge_counts[i] = g.num_edges

    # --- Stats ---
    mu = node_counts.mean()
    sigma = node_counts.std()

    if args.use_subset:
        lower = mu - sigma
        upper = mu + sigma

        valid_indices = np.where(
            (node_counts >= lower) & (node_counts <= upper)
        )[0]
    else:
        valid_indices = np.arange(len(dataset), dtype=np.int32)
        lower, upper = None, None
    
    # --- Compute number of pairs without generating them ---
    n_valid = len(valid_indices)
    num_pairs = n_valid * (n_valid - 1) // 2

    # --- Splits ---
    train_idx, temp_idx = train_test_split(
        valid_indices,
        test_size=0.4,
        random_state=42,
        shuffle=True
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    # --- Save indices separately (faster reload, reusable) ---
    np.save(output_dir / f"{args.dataset_name}_valid_idx.npy", valid_indices)
    np.save(output_dir / f"{args.dataset_name}_train_idx.npy", train_idx)
    np.save(output_dir / f"{args.dataset_name}_val_idx.npy", val_idx)
    np.save(output_dir / f"{args.dataset_name}_test_idx.npy", test_idx)

    # --- Metadata ---
    data = {
        "dataset": args.dataset_name,
        "num_graphs_total": len(dataset),
        "num_graphs_filtered": int(n_valid),
        "num_total_pairs": int(num_pairs),

        "node_stats": {
            "min_nodes": int(node_counts.min()),
            "max_nodes": int(node_counts.max()),
            "mean_nodes": float(mu),
            "std_nodes": float(sigma),
            "lower_bound": float(lower) if lower is not None else None,
            "upper_bound": float(upper) if upper is not None else None
        },

        "edge_stats": {
            "min_edges": int(edge_counts.min()),
            "max_edges": int(edge_counts.max()),
            "mean_edges": float(edge_counts.mean()),
            "std_edges": float(edge_counts.std())
        },

        # "valid_graph_indices": valid_indices,

        # "splits": {
        #     "train": train_idx.tolist(),
        #     "val": val_idx.tolist(),
        #     "test": test_idx.tolist()
        # }
    }

    output_file = output_dir / f"{args.dataset_name}_metadata.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)