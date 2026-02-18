import json
import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    embedding_dim: int
    k: int
    num_layers: int


@dataclass
class TrainingParams:
    epochs_triplet: int
    epochs_siamese: int
    lr: float
    weight_decay: float
    triplet_margin: float


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
    training: TrainingParams
    alpha_tracker: AlphaTrackerConfig
    output_dir: str


def load_config(path: str) -> Config:
    with open(path) as f:
        data = json.load(f)

    return Config(
        dataset=data["dataset"],
        device=torch.device(data["device"]),
        model=ModelConfig(**data["model"]),
        training=TrainingParams(**data["training"]),
        alpha_tracker=AlphaTrackerConfig(**data["alpha_tracker"]),
        output_dir=data["output_dir"],
    )