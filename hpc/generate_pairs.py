import random
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations
from torch_geometric.datasets import TUDataset


def generate_all_pairs(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    pairs = list(combinations(range(len(dataset)), 2))

    np.save(output_dir / f"{args.dataset_name}_all_pairs.npy", pairs)


def generate_sub_pairs(args, k=1000, seed=42):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    pairs = list(combinations(range(len(dataset)), 2))

    random.seed(seed)
    sub_pairs = random.sample(pairs, k)

    pairs_file = output_dir / f"{args.dataset_name}_pairs_{k}.npy"
    np.save(pairs_file, np.array(sub_pairs))


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    return parser


def main(args):
    generate_all_pairs(args)
    generate_sub_pairs(args)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
