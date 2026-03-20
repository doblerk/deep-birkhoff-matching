import json
import torch
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelConfig:
    embedding_dim: int
    k: int
    num_layers: int
    perm_strategy: str


@dataclass
class EncoderConfig:
    mode: str
    checkpoint: str


@dataclass
class PermutationEvolution:
    evolve: bool
    evolve_every: int
    num_replace: int


@dataclass
class TrainingParams:
    epochs_triplet: int
    epochs_siamese: int
    lr_triplet: float
    lr_siamese: float
    weight_decay: float
    triplet_margin: float
    use_entropy: bool
    entropy_weight: float


@dataclass
class AlphaTrackerConfig:
    warmup: int
    window: int
    ema_decay: float


@dataclass
class Config:
    dataset: str
    metadata_dir: str
    dataset_dir: str
    output_dir: str
    device: torch.device
    model: ModelConfig
    encoder: EncoderConfig
    perm_evo: PermutationEvolution
    training: TrainingParams
    alpha_tracker: AlphaTrackerConfig

def load_json(path: str):
    with open(path) as f:
        data = json.load(f)
    return data

def load_data(path: str) -> Config:
    data = load_json(path)

    config = Config(
        dataset=data["dataset"],
        metadata_dir=data["metadata_dir"],
        dataset_dir=data["dataset_dir"],
        output_dir=data["output_dir"],
        device=torch.device(data["device"]),
        model=ModelConfig(**data["model"]),
        encoder=EncoderConfig(**data["encoder"]),
        perm_evo=PermutationEvolution(**data["permutation_evolution"]),
        training=TrainingParams(**data["training"]),
        alpha_tracker=AlphaTrackerConfig(**data["alpha_tracker"]),
    )

    metadata_dir = Path(config.metadata_dir) / config.dataset

    metadata_path = metadata_dir / f"{config.dataset}_metadata.json"
    ged_path = metadata_dir / f"{config.dataset}_ged_matrices.pt"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    if not ged_path.exists():
        raise FileNotFoundError(ged_path)

    metadata = load_json(metadata_path)
    ged_data = torch.load(ged_path)

    return config, metadata, ged_data