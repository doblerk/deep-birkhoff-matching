import os
import time
import json
import logging
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from itertools import combinations
from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant


# ---------- COST FUNCTIONS ----------

def node_subst_cost(n1, n2):
    return 0.0 if n1 == n2 else 1.0

def node_del_cost(node):
    return 1.0

def node_ins_cost(node):
    return 1.0

def edge_subst_cost(e1, e2):
    return 0.0

def edge_del_cost(node):
    return 1.0

def edge_ins_cost(node):
    return 1.0

# ------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--pairs_file', type=str, help='Path to pairs file')
    parser.add_argument('--metadata', type=str, help='Dataset metadata info')
    parser.add_argument('--timeout', type=float, default=None, help='Path to pairs directory')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    return parser

def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'log_ged_{args.dataset_name}_{args.start_idx}_{args.end_idx}.txt'
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    logging.info("Loading dataset...")
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset_name)

    if not hasattr(dataset[0], 'x') or dataset[0].x is None:
        dataset.transform = Constant(value=1.0)
        logging.info("Dataset missing node features 'x', applied Constant transform.")

    # --------------------------------------------------
    # Load metadata
    # --------------------------------------------------
    logging.info(f"Loading metadata info from {args.metadata}...")
    with open(args.metadata, "r") as f:
        info = json.load(f)
    
    valid_idx = info["valid_graph_indices"]
    n_filtered = info["num_graphs_filtered"]

    logging.info(f"Number of filtered graphs: {n_filtered}")

    # --------------------------------------------------
    # Load pairs
    # --------------------------------------------------
    logging.info("Loading pairs...")
    pairs = np.load(
        args.pairs_file,
        allow_pickle=True,
        mmap_mode="r"
    )

    total_pairs = len(pairs)
    logging.info(f"Total pairs: {total_pairs}")

    # --------------------------------------------------
    # Determine job slice
    # --------------------------------------------------
    start = args.start_idx
    end = min(args.end_idx, total_pairs)

    subset_pairs = pairs[start:end]

    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    logging.info(f"Job {job_id} processing pairs {start}–{end-1}")

    # --------------------------------------------------
    # Convert only required graphs to networkx
    # --------------------------------------------------
    logging.info("Converting graphs to networkx...")
    dataset_nx = {
        i: to_networkx(dataset[i], node_attrs='x', to_undirected=True)
        for i in range(len(dataset))
    }

    # --------------------------------------------------
    # Compute GED
    # --------------------------------------------------
    results = []

    for idx, (a, b) in enumerate(subset_pairs):

        logging.info(f"[{idx+1}/{len(subset_pairs)}] GED graph {a} vs {b}")

        g1 = dataset_nx[a]
        g2 = dataset_nx[b]

        t0 = time.time()

        ged = nx.graph_edit_distance(
            g1, g2,
            node_subst_cost=node_subst_cost,
            node_del_cost=node_del_cost,
            node_ins_cost=node_ins_cost,
            edge_subst_cost=edge_subst_cost,
            edge_del_cost=edge_del_cost,
            edge_ins_cost=edge_ins_cost,
            timeout=args.timeout
        )

        elapsed = time.time() - t0

        logging.info(f"GED({a},{b}) = {ged} | time={elapsed:.2f}s")

        hit_timeout = elapsed >= args.timeout

        results.append((a, b, ged, elapsed, hit_timeout))

    results = np.array(results, dtype=object)

    part_file = output_dir / f"ged_results_{args.dataset_name}_{start}_{end - 1}.npy"

    np.save(part_file, results)

    logging.info(f"Saved results to {part_file}")
        

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
