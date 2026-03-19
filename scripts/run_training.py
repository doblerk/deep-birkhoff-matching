import argparse

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant

from birkhoffnet.utils.config import load_config, load_metadata
from birkhoffnet.utils.dataloader_utils import DataLoaders
from birkhoffnet.utils.model_utils import ModelFactory
from birkhoffnet.utils.trainer_utils import TripletTrainer, SiameseTrainer


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Path to parameters file')
    parser.add_argument('--metadata', type=str, help='Path to metadata file')
    parser.add_argument('--ged_data', type=str, help='Path to ged file')
    return parser


def main(args):
    
    config = load_config(args.params)
    metadata = load_metadata(args.metadata)

    device = torch.device(config.device)

    # --------------------------------------------------
    # 1. Load base dataset
    # --------------------------------------------------
    
    dataset = TUDataset(
        root=config.dataset_dir, 
        name=config.dataset,
        use_node_attr=False
    )

    # --------------------------------------------------
    # 2. Load metadata
    # --------------------------------------------------
    
    valid_indices = metadata["valid_graph_indices"]

    # valid_indices from metadata
    valid_idx_map = {orig_idx: i for i, orig_idx in enumerate(valid_indices)}

    # train/val/test in original indices
    train_indices_orig = set(metadata["splits"]["train"])
    val_indices_orig = set(metadata["splits"]["val"])
    test_indices_orig = set(metadata["splits"]["test"])

    # map to 0..n_filtered-1
    train_indices = [valid_idx_map[i] for i in train_indices_orig]
    val_indices = [valid_idx_map[i] for i in val_indices_orig]
    test_indices = [valid_idx_map[i] for i in test_indices_orig]

    # --------------------------------------------------
    # 3. Load GED matrices
    # --------------------------------------------------

    data = torch.load(args.ged_data)

    norm_ged_matrix = data["norm_ged_matrix"]
    node_counts = data["node_counts"]

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
        num_features=dataset.num_features,
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