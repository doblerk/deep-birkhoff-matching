import json
import torch
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
    freeze_after_evolve: bool


@dataclass
class TrainingParams:
    epochs_triplet: int
    epochs_siamese: int
    lr: float
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
    device: torch.device
    model: ModelConfig
    encoder: EncoderConfig
    perm_evo: PermutationEvolution
    training: TrainingParams
    alpha_tracker: AlphaTrackerConfig
    dataset_dir: str
    output_dir: str

def load_json(path: str):
    with open(path) as f:
        data = json.load(f)
    return data

def load_config(path: str) -> Config:
    data = load_json(path)

    return Config(
        dataset=data["dataset"],
        device=torch.device(data["device"]),
        model=ModelConfig(**data["model"]),
        encoder=EncoderConfig(**data["encoder"]),
        perm_evo=PermutationEvolution(**data["permutation_evolution"]),
        training=TrainingParams(**data["training"]),
        alpha_tracker=AlphaTrackerConfig(**data["alpha_tracker"]),
        dataset_dir=data["dataset_dir"],
        output_dir=data["output_dir"],
    )


def load_metadata(path: str):
    return load_json(path)