import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from birkhoffnet.losses.triplet_loss import TripletLoss


# =========================================================
# Configuration
# =========================================================

@dataclass
class TrainingConfig:
    device: torch.device
    epochs_triplet: int = 2001
    epochs_siamese: int = 201
    lr: float = 1e-3
    weight_decay: float = 1e-6
    triplet_margin: float = 0.8
    output_dir: str = "./"


# =========================================================
# Triplet Trainer
# =========================================================

class TripletTrainer:

    def __init__(self, encoder: torch.nn.Module, optimizer, config: TrainingConfig):
        self.encoder = encoder
        self.optimizer = optimizer
        self.config = config
        self.criterion = TripletLoss(margin=config.triplet_margin)

    def train(self, loader):

        self.encoder.train()

        for epoch in range(self.config.epochs_triplet):

            total_loss = 0
            total_samples = 0

            for anchor_graphs, pos_graphs, neg_graphs in loader:

                a_batch = anchor_graphs.to(self.config.device)
                p_batch = pos_graphs.to(self.config.device)
                n_batch = neg_graphs.to(self.config.device)

                self.optimizer.zero_grad()

                _, a_emb = self.encoder(a_batch.x, a_batch.edge_index, a_batch.batch)
                _, p_emb = self.encoder(p_batch.x, p_batch.edge_index, p_batch.batch)
                _, n_emb = self.encoder(n_batch.x, n_batch.edge_index, n_batch.batch)

                loss = self.criterion(a_emb, p_emb, n_emb)

                loss.backward()
                self.optimizer.step()

                batch_size = anchor_graphs.i.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples

            if epoch % 10 == 0:
                print(f"[Triplet] Epoch {epoch+1}/{self.config.epochs_triplet} - Loss: {avg_loss:.4f}")

        self._save_checkpoint()
        return self.encoder

    def _save_checkpoint(self):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, f"{self.config.output_dir}/ckpt_encoder.pth")


# =========================================================
# Siamese Trainer
# =========================================================

class SiameseTrainer:

    def __init__(
        self,
        encoder,
        alpha_layer,
        alpha_tracker,
        perm_pool,
        cost_builder,
        criterion,
        config: TrainingConfig,
    ):
        self.encoder = encoder
        self.alpha_layer = alpha_layer
        self.alpha_tracker = alpha_tracker
        self.perm_pool = perm_pool
        self.cost_builder = cost_builder
        self.criterion = criterion
        self.config = config

        self.optimizer = torch.optim.AdamW(
            list(alpha_layer.parameters())
            + list(cost_builder.parameters())
            + list(criterion.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader):

        for epoch in range(self.config.epochs_siamese):

            self._train_one_epoch(train_loader, epoch)

            if epoch % 10 == 0:
                val_loss = self.evaluate(val_loader)
                print(
                    f"[GED] Epoch {epoch+1}/{self.config.epochs_siamese} "
                    f"- Val MSE: {val_loss:.4f} "
                    f"- RMSE: {np.sqrt(val_loss):.4f} "
                    f"- Scale: {self.criterion.scale.item():.4f}"
                )

        test_loss = self.evaluate(test_loader)
        print(
            f"[GED] Final Test MSE: {test_loss:.4f} "
            f"- RMSE: {np.sqrt(test_loss):.4f}"
        )

    # --------------------------------------------------------
    # Internal Training Step
    # --------------------------------------------------------

    def _train_one_epoch(self, loader, epoch):

        self.alpha_layer.train()
        self.criterion.train()

        for batch1, batch2, ged_labels in loader:

            batch1 = batch1.to(self.config.device)
            batch2 = batch2.to(self.config.device)
            ged_labels = ged_labels.to(self.config.device)

            n_nodes_1 = batch1.batch.bincount()
            n_nodes_2 = batch2.batch.bincount()
            normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

            self.optimizer.zero_grad()

            with torch.no_grad():
                node_repr_b1, graph_repr_b1 = self.encoder(
                    batch1.x, batch1.edge_index, batch1.batch
                )
                node_repr_b2, graph_repr_b2 = self.encoder(
                    batch2.x, batch2.edge_index, batch2.batch
                )

            cost_matrices, masks1, masks2 = self.cost_builder(
                node_repr_b1, graph_repr_b1, batch1.batch,
                node_repr_b2, graph_repr_b2, batch2.batch
            )

            soft_assignments, alphas, entropy = self.alpha_layer(
                graph_repr_b1, graph_repr_b2
            )

            self.alpha_tracker.collect(alphas)

            assignment_masks = masks1.unsqueeze(2) * masks2.unsqueeze(1)
            soft_assignments = soft_assignments * assignment_masks

            row_sums = soft_assignments.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            soft_assignments = soft_assignments / row_sums

            predicted_ged = self.criterion(cost_matrices, soft_assignments)
            normalized_predicted = torch.exp(-predicted_ged / normalization_factor)

            loss = F.mse_loss(normalized_predicted, ged_labels, reduction="mean")

            loss.backward()
            self.optimizer.step()

        sorted_idx, _ = self.alpha_tracker.update()
        if sorted_idx is not None:
            self.perm_pool.mate_permutations(sorted_idx, k=2)
            # self.alpha_layer.freeze_module()
            # self.alpha_layer.start_freeze_timer()
        
        # if alpha_layer.is_frozen():
        #     alpha_layer.update_freeze_timer()

        #     if alpha_layer.freeze_timer == 0:
        #         alpha_layer.unfreeze_module()
        #         alpha_layer.reset_freeze_timer()


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader):

        self.alpha_layer.eval()
        self.cost_builder.eval()
        self.criterion.eval()

        total_loss = 0
        total_samples = 0

        for batch1, batch2, ged_labels in loader:

            batch1 = batch1.to(self.config.device)
            batch2 = batch2.to(self.config.device)
            ged_labels = ged_labels.to(self.config.device)

            n_nodes_1 = batch1.batch.bincount()
            n_nodes_2 = batch2.batch.bincount()
            normalization_factor = 0.5 * (n_nodes_1 + n_nodes_2)

            node_repr_b1, graph_repr_b1 = self.encoder(
                batch1.x, batch1.edge_index, batch1.batch
            )
            node_repr_b2, graph_repr_b2 = self.encoder(
                batch2.x, batch2.edge_index, batch2.batch
            )

            cost_matrices, masks1, masks2 = self.cost_builder(
                node_repr_b1, graph_repr_b1, batch1.batch,
                node_repr_b2, graph_repr_b2, batch2.batch
            )

            soft_assignments, _, _ = self.alpha_layer(
                graph_repr_b1, graph_repr_b2
            )

            assignment_masks = masks1.unsqueeze(2) * masks2.unsqueeze(1)
            soft_assignments = soft_assignments * assignment_masks
            row_sums = soft_assignments.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            soft_assignments = soft_assignments / row_sums

            predicted_ged = self.criterion(cost_matrices, soft_assignments)
            normalized_predicted = torch.exp(-predicted_ged / normalization_factor)

            loss = F.mse_loss(normalized_predicted, ged_labels)

            total_loss += loss.item() * ged_labels.size(0)
            total_samples += ged_labels.size(0)

        return total_loss / total_samples