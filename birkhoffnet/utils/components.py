import torch
from dataclasses import dataclass


@dataclass
class Optimizers:
    encoder: torch.optim.Optimizer
    encoder_scheduler: torch.optim.lr_scheduler


@dataclass
class ModelModules:
    encoder: torch.nn.Module
    alpha_layer: torch.nn.Module
    cost_builder: torch.nn.Module


@dataclass
class ModelComponents:
    modules: ModelModules
    optimizers: Optimizers
    alpha_tracker: object
    perm_pool: object
    criterion: torch.nn.Module