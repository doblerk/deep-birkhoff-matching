import argparse

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant

from birkhoffnet.utils.config import load_data
from birkhoffnet.utils.dataloader_utils import DataLoaders
from birkhoffnet.utils.model_utils import ModelFactory
from birkhoffnet.utils.trainer_utils import TripletTrainer, SiameseTrainer


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Path to parameters file')
    return parser


def main(args):
    
    config, _, ged_data, valid_idx, train_idx, val_idx, test_idx = load_data(args.params)

    device = torch.device(config.device)

    # --------------------------------------------------
    # 1. Load base dataset
    # --------------------------------------------------
    
    dataset_full = TUDataset(
        root=config.dataset_dir, 
        name=config.dataset,
        use_node_attr=False
    )

    if not hasattr(dataset_full[0], 'x') or dataset_full[0].x is None:
        dataset_full.transform = Constant(value=1.0)

    # --------------------------------------------------
    # 2. Load metadata
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
    # 4. Instantiate DataLoaders
    # --------------------------------------------------

    loaders = DataLoaders(
        dataset, 
        train_indices,
        val_indices,
        test_indices,
        norm_ged_matrix
    )

    # 5. Initialize models
    components = ModelFactory.initialize(
        num_features=dataset_full.num_features,
        max_graph_size=max_nodes,
        config=config
    )

    encoder = components.modules.encoder
    optimizer = components.optimizers.encoder
    scheduler = components.optimizers.encoder_scheduler

    # --------------------------------------------------
    # Encoder setup: train or load
    # --------------------------------------------------

    if config.encoder.mode == "train":

        triplet_trainer = TripletTrainer(
            encoder,
            optimizer,
            scheduler,
            config=config
        )

        encoder = triplet_trainer.train(loaders.triplet_loader)

        encoder.eval()
        encoder.freeze_params(encoder)
    
    elif config.encoder.mode == "load":
        
        ckpt_path = f"{config.output_dir}/{config.encoder.checkpoint}"
        ckpt = torch.load(ckpt_path, map_location=device)

        encoder.load_state_dict(ckpt["encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])

        encoder.eval()
        encoder.freeze_params(encoder)
    
    else:
        raise ValueError("encoder.mode must be 'train' or 'load'.")

    # --------------------------------------------------
    # Offline Hungarian initialization
    # --------------------------------------------------

    if config.model.perm_strategy == "offline_hungarian":

        components.perm_pool.initialize(
            config.model.perm_strategy,
            encoder,
            loaders.train_loader
        )

        perm_vectors = components.perm_pool.get_vectors()

        components.modules.alpha_layer.set_permutations(perm_vectors)

    # --------------------------------------------------
    # Siamese training
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

    siamese_trainer.train(
        loaders.train_loader,
        loaders.val_loader,
        loaders.test_loader,
    )


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)