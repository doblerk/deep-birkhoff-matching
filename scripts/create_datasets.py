import json
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):

    output_dir = Path(args.output_dir)

    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    node_counts = np.array([g.num_nodes for g in dataset])
    edge_counts = np.array([g.num_edges for g in dataset])

    mu = node_counts.mean()
    sigma = node_counts.std()

    lower = mu - sigma
    upper = mu + sigma

    valid_indices = [
        i for i, g in enumerate(dataset)
        if lower <= g.num_nodes <= upper
    ]

    pairs = np.array(list(combinations(valid_indices, r=2)), dtype=np.int32)
    np.save(output_dir / f"{args.dataset_name}_pairs.npy", pairs)

    # splits
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

    data = {
        "dataset": args.dataset_name,
        "num_graphs_total": len(dataset),
        "num_graphs_filtered": len(valid_indices),
        "num_total_pairs": len(pairs),

        "node_stats": {
            "min_nodes": int(node_counts.min()),
            "max_nodes": int(node_counts.max()),
            "mean_nodes": float(mu),
            "std_nodes": float(sigma),
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        },

        "edge_stats": {
            "min_edges": int(edge_counts.min()),
            "max_edges": int(edge_counts.max()),
            "mean_edges": float(edge_counts.mean()),
            "std_edges": float(edge_counts.std())
        },

        "valid_graph_indices": valid_indices,

        "splits": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx
        }
    }

    output_file = output_dir / f"{args.dataset_name}_metadata.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)