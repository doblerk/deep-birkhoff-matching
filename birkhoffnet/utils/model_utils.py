# model_utils.py
import torch
from birkhoffnet.models.gnn_models import Model
from birkhoffnet.losses.ged_loss import GEDLoss
from birkhoffnet.utils.config import Config
from birkhoffnet.utils.components import ModelComponents, ModelModules, Optimizers
from birkhoffnet.utils.callbacks import AlphaTracker
from birkhoffnet.utils.permutation import PermutationPool
from birkhoffnet.models.alpha_layers import AlphaPermutationLayer, AlphaMLP
from birkhoffnet.models.cost_matrix_builder import CostMatrixBuilder


class ModelFactory:

    @staticmethod
    def initialize(num_features: int, max_graph_size: int, config: Config):

        encoder = Model(
            num_features,
            config.model.embedding_dim,
            config.model.num_layers,
            use_attention=False,
            attn_concat=False
        ).to(config.device)

        encoder_optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=config.training.lr_triplet,
            weight_decay=config.training.weight_decay
        )

        encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoder_optimizer,
            T_max=config.training.epochs_triplet,
            eta_min=1e-4
        )

        perm_pool = PermutationPool(
            max_n=max_graph_size,
            k=config.model.k,
            config=config
        )

        if config.model.perm_strategy == "offline_hungarian":
            perm_pool.initialize("random")
        else:
            perm_pool.initialize(config.model.perm_strategy)

        perm_vectors = perm_pool.get_vectors().to(config.device)

        alpha_layer = AlphaPermutationLayer(
            perm_vectors,
            AlphaMLP(encoder.output_dim, config.model.k),
            config.training.entropy_weight,
            config.training.epochs_siamese
        ).to(config.device)

        alpha_tracker = AlphaTracker(
            config.model.k,
            warmup=config.alpha_tracker.warmup,
            window=config.alpha_tracker.window
        )

        cost_builder = CostMatrixBuilder(
            embedding_dim=config.model.embedding_dim,
            max_graph_size=max_graph_size,
            use_learned_sub=True,
            model_indel=None,
            rank=None
        ).to(config.device)

        criterion = GEDLoss().to(config.device)

        return ModelComponents(
            modules=ModelModules(
                encoder=encoder,
                alpha_layer=alpha_layer,
                cost_builder=cost_builder
            ),
            optimizers=Optimizers(
                encoder=encoder_optimizer,
                encoder_scheduler=encoder_scheduler
            ),
            alpha_tracker=alpha_tracker,
            perm_pool=perm_pool,
            criterion=criterion
        )