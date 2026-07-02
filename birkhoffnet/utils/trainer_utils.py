import logging
import torch
import numpy as np
import torch.nn.functional as F
from time import time
from birkhoffnet.utils.config import Config
from birkhoffnet.losses.triplet_loss import TripletLoss


# =========================================================
# Triplet Trainer
# =========================================================

class TripletTrainer:

    def __init__(self, 
                 encoder: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler, 
                 config: Config):
        self.encoder = encoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.criterion = TripletLoss(margin=config.training.triplet_margin)

    def train(self, loader):

        self.encoder.train()

        best_loss = float('inf')
        patience_counter = 0
        patience = 30
        tol = 1e-4

        for epoch in range(self.config.training.epochs_triplet):

            total_loss = 0
            total_gap = 0
            total_active = 0
            total_samples = 0

            for anchor_graphs, pos_graphs, neg_graphs in loader:

                a_batch = anchor_graphs.to(self.config.device)
                p_batch = pos_graphs.to(self.config.device)
                n_batch = neg_graphs.to(self.config.device)

                self.optimizer.zero_grad()

                _, a_emb = self.encoder(a_batch.x, a_batch.edge_index, a_batch.batch)
                _, p_emb = self.encoder(p_batch.x, p_batch.edge_index, p_batch.batch)
                _, n_emb = self.encoder(n_batch.x, n_batch.edge_index, n_batch.batch)
                
                a_emb = F.normalize(a_emb, p=2, dim=1)
                p_emb = F.normalize(p_emb, p=2, dim=1)
                n_emb = F.normalize(n_emb, p=2, dim=1)

                # ------ diagnostics monitoring ------
                d_ap = torch.norm(a_emb - p_emb, dim=1)
                d_an = torch.norm(a_emb - n_emb, dim=1)
                gap = d_an - d_ap
                active = (d_ap - d_an + self.config.training.triplet_margin > 0).float()
                total_gap += gap.sum().item()
                total_active += active.sum().item()
                # ------------------------------------

                logging.info(
                    f"d_ap={d_ap.mean().item():.4f} "
                    f"d_an={d_an.mean().item():.4f} "
                    f"gap={gap.mean().item():.4f} "
                    f"active={active.mean().item():.4f}"
                )

                loss = self.criterion(a_emb, p_emb, n_emb)

                loss.backward()
                self.optimizer.step()

                batch_size = anchor_graphs.batch_size
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            self.scheduler.step()

            avg_loss = total_loss / total_samples
            avg_gap = total_gap / total_samples
            avg_active = total_active / total_samples

            logging.info(
                f"[Triplet] Epoch {epoch+1}/{self.config.training.epochs_triplet}: "
                f"Loss={avg_loss:.4f} "
                f"Gap={avg_gap:.4f} "
                f"Active={avg_active:.4f}"
            )

            # ------ Early stopping ------
            if avg_loss < best_loss - tol:
                best_loss = avg_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best loss={best_loss:.4f} at epoch {best_epoch+1}"
                )
                break

        return self.encoder

    def _save_checkpoint(self):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "criterion": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
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
        graph_loader,
        config: Config,
    ):
        self.encoder = encoder
        self.alpha_layer = alpha_layer
        self.alpha_tracker = alpha_tracker
        self.perm_pool = perm_pool
        self.cost_builder = cost_builder
        self.criterion = criterion
        self.config = config

        self.optimizer = torch.optim.AdamW([
            {"params": alpha_layer.model.parameters(), "lr": config.training.lr_siamese},
            {"params": cost_builder.parameters(), "lr": config.training.lr_siamese},
            {"params": criterion.parameters(), "lr": config.training.lr_siamese},
            {"params": alpha_layer.log_temperature, "lr": config.training.lr_siamese},
        ], weight_decay=config.training.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs_siamese,
            eta_min=1e-5
        )

        self.node_cache, self.mask_cache, self.graph_cache = self._precompute_embeddings(graph_loader)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def train(self, train_loader, val_loader, test_loader):

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        patience = 20
        tol = 1e-5

        for epoch in range(self.config.training.epochs_siamese):

            self._train_one_epoch(train_loader, epoch)

            val_loss = self.evaluate(val_loader)

            if val_loss < best_val_loss - tol:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
            
            if epoch % 1 == 0:
                logging.info(
                    f"[GED] Epoch {epoch+1}/{self.config.training.epochs_siamese} "
                    f"- Val MSE: {val_loss:.4f} "
                    f"- RMSE: {np.sqrt(val_loss):.4f} "
                    f"- Scale: {self.criterion.scale.item():.4f}"
                )

            if patience_counter >= patience:
                logging.info(
                    f"Early stopping triggered at epoch {epoch+1}. "
                    f"Best epoch was {best_epoch+1} "
                    f"with val MSE={best_val_loss:.6f}"
                )
                break
                    
        # Restore best checkpoint before testing
        self._load_checkpoint()

        test_loss = self.evaluate(test_loader)
        
        logging.info(
            f"[GED] Final Test MSE: {test_loss:.6f} "
            f"- RMSE: {np.sqrt(test_loss):.6f}"
        )

    # --------------------------------------------------------
    # Internal Training Step
    # --------------------------------------------------------

    def _train_one_epoch(self, loader, epoch):

        self.alpha_layer.train()
        self.criterion.train()

        for graphs_a, graphs_b, target_similarity in loader:

            graphs_a = graphs_a.to(self.config.device)
            graphs_b = graphs_b.to(self.config.device)
            target_similarity = target_similarity.to(self.config.device)

            graph_ids_a = graphs_a.graph_id.cpu()
            graph_ids_b = graphs_b.graph_id.cpu()

            node_emb_a, graph_emb_a, node_mask_a = self._get_cached_embeddings(graph_ids_a)
            node_emb_b, graph_emb_b, node_mask_b = self._get_cached_embeddings(graph_ids_b)

            num_nodes_a = node_mask_a.sum(dim=1)
            num_nodes_b = node_mask_b.sum(dim=1)
            avg_num_nodes = 0.5 * (num_nodes_a + num_nodes_b)

            self.optimizer.zero_grad()

            cost_matrix, node_mask_a, node_mask_b = self.cost_builder(
                node_emb_a,
                node_mask_a,
                graph_emb_a,
                node_emb_b,
                node_mask_b,
            )

            assignment_matrix, alphas = self.alpha_layer(
                graph_emb_a,
                graph_emb_b,
            )

            if self.config.perm_evo.evolve:
                self.alpha_tracker.collect(alphas)

            assignment_matrix = self._normalize_assignment(
                assignment_matrix,
                node_mask_a,
                node_mask_b,
            )

            pred_ged = self.criterion(
                cost_matrix,
                assignment_matrix,
            )

            pred_similarity = torch.exp(-pred_ged / avg_num_nodes)

            loss = self.alpha_layer.loss_fn(
                pred_similarity,
                target_similarity,
                use_entropy=self.config.training.use_entropy,
                alphas=alphas,
                epoch=epoch,
            )

            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()

        # ----------------------------------
        # Genetic permutation evolution
        # ----------------------------------

        if self.config.perm_evo.evolve:

            sorted_idx = self.alpha_tracker.update()

            if sorted_idx is not None:

                k = self.config.model.k
                ratio = self.config.perm_evo.replace_ratio
                n_replace = max(1, int(k * ratio))

                self.perm_pool.mate_permutations(
                    sorted_idx, 
                    k=n_replace
                )

                new_perms = self.perm_pool.get_vectors()
                self.alpha_layer.set_permutations(new_perms)

                # print("Vectors match:", torch.equal(self.perm_pool.get_vectors(), self.alpha_layer.perm_vectors))

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

        for graphs_a, graphs_b, target_similarity in loader:

            graphs_a = graphs_a.to(self.config.device)
            graphs_b = graphs_b.to(self.config.device)
            target_similarity = target_similarity.to(self.config.device)

            graph_ids_a = graphs_a.graph_id
            graph_ids_b = graphs_b.graph_id

            node_emb_a, graph_emb_a, node_mask_a = self._get_cached_embeddings(graph_ids_a)
            node_emb_b, graph_emb_b, node_mask_b = self._get_cached_embeddings(graph_ids_b)

            num_nodes_a = node_mask_a.sum(dim=1)
            num_nodes_b = node_mask_b.sum(dim=1)
            avg_num_nodes = 0.5 * (num_nodes_a + num_nodes_b)

            cost_matrix, node_mask_a, node_mask_b = self.cost_builder(
                node_emb_a,
                node_mask_a,
                graph_emb_a,
                node_emb_b,
                node_mask_b,
            )

            assignment_matrix, _ = self.alpha_layer(
                graph_emb_a,
                graph_emb_b,
            )

            assignment_matrix = self._normalize_assignment(
                assignment_matrix,
                node_mask_a,
                node_mask_b,
            )

            pred_ged = self.criterion(
                cost_matrix,
                assignment_matrix,
            )

            # print(predicted_ged[:20].to(torch.int32).detach().cpu().numpy())
            # unormalized_ged = - normalization_factor * torch.log(ged_labels.clamp(min=1e-8))
            # print(unormalized_ged[:20].to(torch.int32).detach().cpu().numpy())

            pred_similarity = torch.exp(
                -pred_ged / avg_num_nodes
            )

            loss = self.alpha_layer.loss_fn(
                pred_similarity,
                target_similarity,
            )

            batch_size = target_similarity.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples
    
    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def infer(self, loader, num_graphs):

        self.alpha_layer.eval()
        self.cost_builder.eval()
        self.criterion.eval()

        distance_matrix = torch.zeros(
            (num_graphs, num_graphs), 
            dtype=torch.float32, 
            device=self.config.device
        )
        
        start_time = time()

        for graphs_a, graphs_b, _, graph_ids_a, graph_ids_b in loader:

            graphs_a = graphs_a.to(self.config.device)
            graphs_b = graphs_b.to(self.config.device)

            graph_ids_a = graph_ids_a.to(self.config.device)
            graph_ids_b = graph_ids_b.to(self.config.device)

            node_emb_a, graph_emb_a, node_mask_a = self._get_cached_embeddings(graph_ids_a)
            node_emb_b, graph_emb_b, node_mask_b = self._get_cached_embeddings(graph_ids_b)

            cost_matrix, node_mask_a, node_mask_b = self.cost_builder(
                node_emb_a,
                node_mask_a,
                graph_emb_a,
                node_emb_b,
                node_mask_b,
            )

            assignment_matrix, _ = self.alpha_layer(
                graph_emb_a,
                graph_emb_b,
            )

            assignment_matrix = self._normalize_assignment(
                assignment_matrix,
                node_mask_a,
                node_mask_b,
            )

            pred_ged = self.criterion(
                cost_matrix,
                assignment_matrix,
            )

            distance_matrix[graph_ids_a, graph_ids_b] = pred_ged
            distance_matrix[graph_ids_b, graph_ids_a] = pred_ged
        
        runtime = time() - start_time

        logging.info(f'Inference runtime: {runtime:.4f}')

        return distance_matrix.to(torch.int32).cpu().numpy()
    
    # --------------------------------------------------------
    # Precomputed Embeddings
    # --------------------------------------------------------

    @torch.no_grad()
    def _precompute_embeddings(self, loader):

        device = self.config.device

        all_node = []
        all_graph = []

        max_nodes = 0
        node_dim = None

        for batch in loader:
            batch = batch.to(device)

            node_repr, graph_repr = self.encoder(
                batch.x, batch.edge_index, batch.batch
            )

            node_splits = batch.batch.bincount().tolist()
            node_split = torch.split(node_repr, node_splits)

            for i, gid in enumerate(batch.graph_id.tolist()):
                n = node_split[i].detach().cpu()

                max_nodes = max(max_nodes, n.size(0))
                node_dim = n.size(1)

                all_node.append((gid, n))
                all_graph.append((gid, graph_repr[i].detach().cpu()))

        num_graphs = max(g for g, _ in all_node) + 1

        node_cache = torch.zeros((num_graphs, max_nodes, node_dim), device=device)
        mask_cache = torch.zeros((num_graphs, max_nodes), dtype=torch.bool, device=device)
        graph_cache = torch.zeros((num_graphs, graph_repr.size(1)), device=device)

        for (gid, n) in all_node:
            node_cache[gid, :n.size(0)] = n
            mask_cache[gid, :n.size(0)] = True

        for (gid, g) in all_graph:
            graph_cache[gid] = g

        return node_cache, mask_cache, graph_cache
    
    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def _get_cached_embeddings(self, graph_ids):
        return (
            self.node_cache[graph_ids].to(self.config.device),
            self.graph_cache[graph_ids].to(self.config.device),
            self.mask_cache[graph_ids].to(self.config.device),
        )

    def _normalize_assignment(self, S, mask1, mask2):
        S = S * (mask1.unsqueeze(2) & mask2.unsqueeze(1))

        S = S / S.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        S = S / S.sum(dim=-2, keepdim=True).clamp(min=1e-8)

        return S

    def _save_checkpoint(self):
        torch.save({
            "alpha_layer": self.alpha_layer.state_dict(),
            "cost_builder": self.cost_builder.state_dict(),
            "criterion": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "perms": self.alpha_layer.perm_vectors
        }, f'{self.config.output_dir}/ckpt_ged.pth')
    
    def _load_checkpoint(self):
        checkpoint = torch.load(
            f"{self.config.output_dir}/ckpt_ged.pth",
            map_location=self.config.device
        )
        self.alpha_layer.load_state_dict(checkpoint["alpha_layer"])
        self.cost_builder.load_state_dict(checkpoint["cost_builder"])
        self.criterion.load_state_dict(checkpoint["criterion"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])