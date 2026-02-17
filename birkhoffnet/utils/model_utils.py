# model_utils.py
import torch
from dataclasses import dataclass
from birkhoffnet.models.gnn_models import Model
from birkhoffnet.losses.ged_loss import GEDLoss
from birkhoffnet.utils.train_utils import AlphaTracker
from birkhoffnet.utils.permutation import PermutationPool
from birkhoffnet.models.alpha_layers import AlphaPermutationLayer, AlphaMLP
from birkhoffnet.models.cost_matrix_builder import CostMatrixBuilder

@dataclass
class ModelComponents:
    encoder: torch.nn.Module
    encoder_optimizer: torch.optim.Optimizer
    alpha_layer: torch.nn.Module
    alpha_tracker: object
    perm_pool: object
    cost_builder: torch.nn.Module
    criterion: torch.nn.Module


class ModelFactory:

    @staticmethod
    def initialize(num_features, embedding_dim, max_graph_size, k, device):
        encoder = Model(num_features, embedding_dim, 1).to(device)
        encoder_optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=1e-3, weight_decay=1e-6
        )

        perm_pool = PermutationPool(max_n=max_graph_size, k=k)
        perm_matrices = perm_pool.get_matrix_batch().to(device)

        alpha_layer = AlphaPermutationLayer(
            perm_matrices,
            AlphaMLP(encoder.output_dim, k)
        ).to(device)

        alpha_tracker = AlphaTracker(k, warmup=10, window=5)

        cost_builder = CostMatrixBuilder(
            embedding_dim=embedding_dim,
            max_graph_size=max_graph_size,
            use_learned_sub=False,
            model_indel=None,
            rank=None
        ).to(device)

        criterion = GEDLoss().to(device)

        return ModelComponents(
            encoder=encoder,
            encoder_optimizer=encoder_optimizer,
            alpha_layer=alpha_layer,
            alpha_tracker=alpha_tracker,
            perm_pool=perm_pool,
            cost_builder=cost_builder,
            criterion=criterion
        )