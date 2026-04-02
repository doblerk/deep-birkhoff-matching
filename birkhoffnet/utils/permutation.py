import numpy as np

import torch

from typing import Tuple
from scipy.optimize import linear_sum_assignment
from birkhoffnet.utils.config import Config


class PermutationPool:
    def __init__(
            self, 
            max_n: int, 
            k: int,
            config: Config, 
            seed: int = 42,
            encoder=None,
            dataloader=None):
        """
        Args:
            max_n (int): Maximum graph size (i.e., full matrix size: max_n x max_n)
            k (int): Number of permutation matrices to generate
            seed (int): RNG seed for reproducibility
        """
        self.rng = torch.Generator().manual_seed(seed) #np.random.default_rng(seed)
        self.device = config.device #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.max_n = max_n
        self.k = k

        self.perm_vectors = None

    # --------------------------------------------------
    # Random initialization
    # --------------------------------------------------

    def _init_random_permutations(self) -> torch.Tensor:
        """Generate permutation vectors of max_n length, padded/embedded."""
        perms = torch.zeros((self.k, self.max_n), dtype=torch.long)

        # Identity always first
        perms[0] = torch.arange(0, self.max_n, dtype=torch.long)

        for i in range(1, self.k):
            perms[i] = torch.randperm(
                self.max_n, 
                generator=self.rng
            )

            assert self._is_valid_permutation(perms[i])
        
        return perms.to(self.device)
    
    # --------------------------------------------------
    # Identity perturbation initialization
    # --------------------------------------------------
    
    def _init_identity_perturbations(self) -> torch.Tensor:
        """Generate small perturbations around the identity matrix."""
        perms = torch.zeros((self.k, self.max_n), dtype=torch.long)
        
        # Identity always first
        perms[0] = torch.arange(0, self.max_n, dtype=torch.long)
        
        max_perturb_ratio = 0.4
        max_swaps = max(1, int(self.max_n * max_perturb_ratio / 2))

        for i in range(1, self.k):
            n_swaps = torch.randint(
                1, max_swaps + 1, (1,), 
                generator=self.rng
            ).item()

            perms[i] = self._perturb_identity(n_swaps)
        
        return perms.to(self.device)
    
    def _perturb_identity(self, n_swaps):
        perm = torch.arange(self.max_n)
        
        for _ in range(n_swaps):
            i, j = torch.randint(
                0, self.max_n, (2,), 
                generator=self.rng
            )

            tmp = perm[i].item()
            perm[i] = perm[j]
            perm[j] = tmp
        
        assert self._is_valid_permutation(perm)
        
        return perm
    
    # --------------------------------------------------
    # Hungarian initialization
    # --------------------------------------------------

    def _init_hungarian_permutations(self, encoder, loader):        
        perms_with_cost = []

        encoder.eval()

        with torch.no_grad():
            for batch1, batch2, _ in loader:

                batch1 = batch1.to(self.device)
                batch2 = batch2.to(self.device)

                h1, _ = encoder(
                    batch1.x, 
                    batch1.edge_index, 
                    batch1.batch
                )
                h2, _ = encoder(
                    batch2.x, 
                    batch2.edge_index, 
                    batch2.batch
                )

                n1 = batch1.batch.bincount()
                n2 = batch2.batch.bincount()

                splits1 = torch.split(h1, n1.tolist())
                splits2 = torch.split(h2, n2.tolist())

                for emb1, emb2 in zip(splits1, splits2):

                    C = torch.cdist(emb1, emb2)

                    n_rows, n_cols = C.shape
                    cost = torch.full(
                        (self.max_n, self.max_n),
                        1e6,
                        device=self.device
                    )

                    cost[:n_rows, :n_cols] = C

                    row, col = linear_sum_assignment(
                        cost.cpu().numpy()
                    )

                    total_cost = cost[row[:n_rows], col[:n_rows]].sum()
                    
                    perm = torch.as_tensor(col, dtype=torch.long)

                    perms_with_cost.append((perm, total_cost))

                    assert self._is_valid_permutation(perm)

        # Keep unique
        unique_perms = self._unique_permutations(perms_with_cost)

        # Build pool
        pool = torch.zeros((self.k, self.max_n), dtype=torch.long)

        # Identity always first
        pool[0] = torch.arange(self.max_n)

        # Keep only the top-k lowest assignment cost (best matches)
        unique_perms.sort(key=lambda x: x[1])

        n_insert = min(len(unique_perms), self.k - 1)
        
        assert self.k - 1 <= len(unique_perms), "The number of permutations must be <= than the number of unique permutations."

        for i in range(n_insert):
            pool[i + 1] = unique_perms[i][0]

        return pool.to(self.device)
    
    # --------------------------------------------------
    # PMX Crossover
    # --------------------------------------------------

    def _resolve(self, value: int, seg_from: torch.Tensor, seg_to: torch.Tensor) -> int:
        """Recursively resolve conflicts for a single value."""
        if value in seg_to:
            pos = (seg_to == value).nonzero(as_tuple=True)[0].item()
            return self._resolve(seg_from[pos], seg_from, seg_to)
        else:
            return value
    
    def _partially_mapped_crossover(self, p1: torch.Tensor, p2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        size = len(p1)
        c1 = p1.clone()
        c2 = p2.clone()

        # Random crossover break points
        bp1 = torch.randint(0, size - 1, (1,)).item()
        bp2 = torch.randint(bp1 + 1, size, (1,)).item()

        # Crossover segments
        seg1 = p1[bp1:bp2 + 1]
        seg2 = p2[bp1:bp2 + 1]

        # Exchange segments
        c1[bp1:bp2 + 1] = seg2
        c2[bp1:bp2 + 1] = seg1

        # Fill up mapping outside the crossover region
        for i in list(range(0, bp1)) + list(range(bp2 + 1, size)):
            c1[i] = self._resolve(c1[i].item(), seg1, seg2)
            c2[i] = self._resolve(c2[i].item(), seg2, seg1)

        return c1, c2
    
    def _mutate(self, perm, prob=0.2):
        if torch.rand(1).item() < prob:
            i, j = torch.randint(0, self.n, (2,))
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
        return perm
    
    # --------------------------------------------------
    # Evolution
    # --------------------------------------------------
    
    def mate_permutations(self, sorted_idx: torch.Tensor, k: int = 2) -> None:
        """
        Replaces the k worst perms with offspring produced by the k best perms.
        """
        # best_idx = sorted_idx[-k:]
        n = len(sorted_idx)

        elite_size = max(k, int(0.5 * n))
        elite_idx = sorted_idx[-elite_size:]

        worst_idx = sorted_idx[:k]

        # For each worst individual, generate a new child from the best parents
        for wi in worst_idx:
            # Randomly choose 2 distinct parents from top-k
            # parent_indices = torch.randperm(k)[:2]
            # p1 = self.perm_vectors[best_idx[parent_indices[0]]]
            # p2 = self.perm_vectors[best_idx[parent_indices[1]]]
            parents = elite_idx[torch.randperm(elite_size)[:2]]

            p1 = self.perm_vectors[parents[0]]
            p2 = self.perm_vectors[parents[1]]

            # Perform mating
            c1, c2 = self._partially_mapped_crossover(p1, p2)

            # Randomly choose one child
            child = c1 if torch.rand(1).item() < 0.5 else c2

            child = self._mutate(child)

            assert self._is_valid_permutation(child), "Invalid permutation generated!"

            self.perm_vectors[wi] = child
    
    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _unique_permutations(self, perms):
        unique = []
        identity = torch.arange(self.max_n, dtype=torch.long)

        for p, c in perms:

            if not self._is_valid_permutation(p):
                continue
            
            # Skip identity
            if torch.equal(p, identity):
                continue
            
            # Skip duplicates
            if any(torch.equal(p, up) for up, _ in unique):
                continue

            unique.append((p, c))
        return unique
    
    def _is_valid_permutation(self, perm):
        n = perm.numel()
        return torch.equal(
            torch.sort(perm).values,
            torch.arange(n, device=perm.device)
        )
    
    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def initialize(self, strategy="random", encoder=None, dataloader=None):
        if strategy == "random":
            self.perm_vectors = self._init_random_permutations()
        
        elif strategy == "identity_perturb":
            self.perm_vectors = self._init_identity_perturbations()
        
        elif strategy == "offline_hungarian":
            if encoder is None or dataloader is None:
                raise ValueError("Hungarian init requires encoder and dataloader")
            
            self.perm_vectors = self._init_hungarian_permutations(
                encoder,
                dataloader,
            )
        
        else:
            raise ValueError("unknown permutation strategy.")

    def get_vectors(self) -> torch.Tensor:
        return self.perm_vectors
    
    def get_matrix_batch(self):
        """
        Returns a batch of k permutation matrices of shape (k, max_n, max_n)
        One-hot encoded, inactive rows filled with zeros
        """
        matrices = torch.zeros((self.k, self.max_n, self.max_n), device=self.device, dtype=torch.float32)
        matrices.scatter_(2, self.perm_vectors.unsqueeze(-1), 1.0)
        return matrices
    

    # def mate_permutations(self, sorted_idx: torch.Tensor, k: int = 2) -> None:
    # """
    # Replace all permutations except the first (elite) one
    # using crossover between two random parents from the top-k best.
    # """
    # elite_idx = sorted_idx[0]          # keep this one unchanged
    # best_idx = sorted_idx[:k]          # top-k pool for parent selection
    # replace_idx = sorted_idx[1:]       # everyone except the first

    # for wi in replace_idx:
    #     # Randomly choose 2 distinct parents from top-k
    #     parent_indices = torch.randperm(k)[:2]
    #     p1 = self.perm_vectors[best_idx[parent_indices[0]]]
    #     p2 = self.perm_vectors[best_idx[parent_indices[1]]]

    #     # Perform crossover
    #     c1, c2 = self._partially_mapped_crossover(p1, p2)

    #     # Randomly pick one child
    #     child = c1 if torch.rand(1).item() < 0.5 else c2
    #     print(f'Changing perm: {self.perm_vectors[wi]} with parent indices {parent_indices}')
    #     self.perm_vectors[wi] = child
    #     print(f'Changed to: {self.perm_vectors[wi]}')

    # # ✅ Preserve elite (the first permutation)
    # self.perm_vectors[elite_idx] = self.perm_vectors[elite_idx]