import re
import logging
import argparse
import statistics
from pathlib import Path

from birkhoffnet.utils.config import load_data


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Path to parameters file')
    return parser


def main(args):
    
    config, _, _ = load_data(args.params)

    parent_dir = Path(config.output_dir)

    root = parent_dir / config.dataset

    log_file = root / "log_summary_results.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )

    results = []

    for config_dir in root.iterdir():

        if not config_dir.is_dir():
            continue

        accs = []
        f1s = []

        for seed_dir in config_dir.iterdir():

            log = seed_dir / "log_knn.txt"
            if not log.exists():
                continue

            text = log.read_text()

            acc = float(re.search(r'Acc mean \+/- std: ([0-9.]+)', text).group(1))
            f1 = float(re.search(r'F1 mean \+/- std: ([0-9.]+)', text).group(1))

            accs.append(acc)
            f1s.append(f1)

        results.append({
            "config": config_dir.name,
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.stdev(accs),
            "f1_mean": statistics.mean(f1s),
            "f1_std": statistics.stdev(f1s)
        })
    
    results.sort(key=lambda x: x["acc_mean"], reverse=True)

    logging.info("===== FINAL RESULTS =====")

    for rank, r in enumerate(results, start=1):

        logging.info(
            f"[{rank}] {r['config']} | "
            f"Acc: {r['acc_mean']:.4f} +/- {r['acc_std']:.4f} | "
            f"F1: {r['f1_mean']:.4f} +/- {r['f1_std']:.4f}"
        )


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)