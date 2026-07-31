import logging
import argparse

import torch
import numpy as np
import torch.nn.functional as F

from time import time
from pathlib import Path

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures, Constant

from birkhoffnet.datasets.siamese_dataset import SiameseDataset
from birkhoffnet.losses.ged_loss import GEDLoss
from birkhoffnet.utils.dataloader_utils import DataLoaders
from birkhoffnet.utils.model_utils import ModelFactory
from birkhoffnet.utils.trainer_utils import SiameseTrainer
from birkhoffnet.utils.config import load_data


@torch.no_grad()
def infer_ged(loader, encoder, alpha_layer, cost_builder, criterion, device, num_graphs):

    distance_matrix = torch.zeros((num_graphs, num_graphs), dtype=torch.float32, device=device)
    
    t0 = time()

    for batch in loader:

        batch1, batch2, _, idx1, idx2 = batch
        batch1, batch2, idx1, idx2 = batch1.to(device), batch2.to(device), idx1.to(device), idx2.to(device)

        n_nodes_1 = batch1.batch.bincount()
        n_nodes_2 = batch2.batch.bincount()

        normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

        node_repr_b1, graph_repr_b1 = encoder(
                batch1.x,
                batch1.edge_index,
                batch1.batch
        )

        node_repr_b2, graph_repr_b2 = encoder(
            batch2.x,
            batch2.edge_index,
            batch2.batch
        )

        cost_matrices, masks1, masks2 = cost_builder(
            node_repr_b1,
            graph_repr_b1,
            batch1.batch,
            node_repr_b2,
            graph_repr_b2,
            batch2.batch
        )

        soft_assignments, _ = alpha_layer(
                graph_repr_b1,
                graph_repr_b2
        )
        
        assignment_masks = masks1.unsqueeze(2) * masks2.unsqueeze(1)
        soft_assignments = soft_assignments * assignment_masks

        row_sums = soft_assignments.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_assignments = soft_assignments / row_sums

        col_sums = soft_assignments.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        soft_assignments = soft_assignments / col_sums

        predicted_ged = criterion(cost_matrices, soft_assignments)

        # normalized_predicted_ged = torch.exp(
        #     -predicted_ged / normalization_factor
        # )

        distance_matrix[idx1, idx2] = predicted_ged
    
    t1 = time()
    runtime = t1 - t0
    logging.info(f'Runtime: {runtime:.4f}')

    distance_matrix = torch.maximum(distance_matrix, distance_matrix.T)

    # return distance_matrix.cpu().numpy()

    # predicted_ged = -normalization_factor * torch.log(torch.tensor(distance_matrix))

    return distance_matrix.to(torch.int32).cpu().numpy()


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Path to parameters file')
    return parser


def main(args):

    config, _, ged_data, valid_idx, train_idx, val_idx, test_idx = load_data(args.params)

    device = torch.device(config.device)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    use_attrs = True
    transform = NormalizeFeatures() if use_attrs else None

    dataset_full = TUDataset(
        root=config.dataset_dir,
        name=config.dataset,
        use_node_attr=use_attrs,
        transform=transform
    )

    if not hasattr(dataset_full[0], 'x') or dataset_full[0].x is None:
        dataset_full.transform = Constant(value=1.0)

    log_file = Path(config.output_dir) / "log_inference.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )

    # --------------------------------------------------
    # 2. Metadata filtering
    # --------------------------------------------------

    valid_indices = valid_idx.tolist()

    dataset = [dataset_full[i] for i in valid_indices]

    # valid_indices from metadata
    valid_idx_map = {orig_idx: i for i, orig_idx in enumerate(valid_indices)}

    # train/val/test in original indices
    train_indices_orig = set(train_idx.tolist())
    val_indices_orig   = set(val_idx.tolist())
    test_indices_orig  = set(test_idx.tolist())

    # map to 0..n_filtered-1
    train_indices = [valid_idx_map[i] for i in train_indices_orig]
    val_indices = [valid_idx_map[i] for i in val_indices_orig]
    test_indices = [valid_idx_map[i] for i in test_indices_orig]

    # --------------------------------------------------
    # 3. Load GED matrices
    # --------------------------------------------------

    norm_ged_matrix = ged_data["norm_ged_matrix"]
    node_counts = ged_data["node_counts"]

    max_nodes = int(torch.max(node_counts).item())

    # --------------------------------------------------
    # 4. Initialize models
    # --------------------------------------------------

    components = ModelFactory.initialize(
        num_features=dataset_full.num_features,
        max_graph_size=max_nodes,
        config=config
    )

    encoder = components.modules.encoder
    alpha_layer = components.modules.alpha_layer
    cost_builder = components.modules.cost_builder
    
    criterion = GEDLoss(use_scale=True).to(config.device)

    # --------------------------------------------------
    # 5. Load checkpoints
    # --------------------------------------------------

    ckpt_encoder_path = f"{config.output_dir}/ckpt_encoder.pth"
    ckpt_encoder = torch.load(ckpt_encoder_path, map_location=device)

    encoder.load_state_dict(ckpt_encoder["encoder"])

    ckpt_ged_path = f"{config.output_dir}/ckpt_ged.pth"
    ckpt_ged = torch.load(ckpt_ged_path, map_location=device)

    alpha_layer.load_state_dict(ckpt_ged["alpha_layer"])
    cost_builder.load_state_dict(ckpt_ged["cost_builder"])
    criterion.load_state_dict(ckpt_ged["criterion"])

    encoder.eval()
    alpha_layer.eval()
    cost_builder.eval()
    criterion.eval()

    # --------------------------------------------------
    # 6. Initialize data loader
    # --------------------------------------------------

    siamese_all = SiameseDataset(
        dataset, 
        norm_ged_matrix, 
        pair_mode='all'
    )

    loaders = DataLoaders(
        dataset, 
        train_indices,
        val_indices,
        test_indices,
        norm_ged_matrix
    )

    siamese_all_loader = DataLoader(
        siamese_all, 
        batch_size=4096, 
        shuffle=False, 
        num_workers=10,
        pin_memory=True
    )

    # --------------------------------------------------
    # 7. Infer all graph pairs
    # --------------------------------------------------

    siamese_trainer = SiameseTrainer(
        encoder,
        components.modules.alpha_layer,
        components.alpha_tracker,
        components.perm_pool,
        components.modules.cost_builder,
        components.criterion,
        loaders.graph_loader,
        config=config,
    )

    distances = siamese_trainer.infer(
        siamese_all_loader,
        len(dataset)
    )

    output_file = Path(config.output_dir) / "distances.npy"
    with open(output_file, 'wb') as file:
        np.save(file, distances)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)