import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from tribiged.utils.permutation import PermutationPool


class AlphaTracker:
    def __init__(self, k: int, warmup: int = 100, window: int = 50, ema_decay : float = 0.0):
        """
        Tracks alpha weights across batches and epochs, with warmup and windowed ranking.

        Args:
            k (int): number of alpha components.
            warmup (int): number of epochs to skip before tracking.
            window (int): number of epochs in one tracking window.
            ema_decay (float): optional decay for EMA smoothing (0 = no EMA).
        """
        self.k = k
        self.warmup = warmup
        self.window = window
        self.ema_decay = ema_decay

        self.epoch_total = 0        # lifetime epoch counter
        self.epoch_in_window = 0    # counter since last reset
        
        # stores mean alpha per epoch
        self.history = []
        # stores batch-wise alphas within an epoch
        self._epoch_accum = []

        # for EMA smoothing
        self._ema_mean = torch.zeros(k)

    def collect(self, alphas: torch.Tensor):
        """
        Collects alphas from one batch only after warmup. """
        if self.epoch_total > self.warmup:
            self.epoch_in_window += 1
            self._epoch_accum.append(alphas.detach().cpu())

    def update(self, top_m: int = 2, strategy: str = 'mean'):
        """
        Called once per epoch. Averages accumulated batches, updates history,
        and triggers ranking if window boundary reached.
        """
        self.epoch_total += 1

        if len(self._epoch_accum) == 0:
            return None, None

        epoch_mean = torch.cat(self._epoch_accum, dim=0).mean(dim=0)
        self._epoch_accum.clear()

        if self.ema_decay > 0.0:
            self._ema_mean = self.ema_decay * self._ema_mean + (1 - self.ema_decay) * epoch_mean
            epoch_mean = self._ema_mean
        
        self.history.append(epoch_mean)

        # if self.epoch_total > self.warmup:
            # mean_alphas = alphas.mean(dim=0) # (k,)
            # self.history.append(mean_alphas)

        # If window reached, rank and reset
        if self.epoch_in_window == self.window:
            ranking = self._rank_alphas(top_m, strategy)
            self._reset_window()
            return ranking
        
        return None, None
    
    def _reset_window(self):
        """Clears history & reset epoch counter."""
        self.epoch_in_window = 0
        self.history.clear()
    
    def get_history(self):
        """Returns stacked tensor of shape (num_epochs_in_window, k)."""
        return torch.stack(self.history, dim=0)
    
    def _get_usage_stats(self, top_m=2):
        """Computes mean usage and top-k frequency stats."""
        history = self.get_history()
        mean_usage = history.mean(dim=0)

        topk_idx = history.topk(top_m, dim=1).indices
        freq_top = torch.bincount(
            topk_idx.flatten(), minlength=self.k
        ) / history.size(0)

        return mean_usage, freq_top
    
    def _rank_alphas(self, top_m=2, strategy='mean'):
        """
        Rank alphas from most underused to most used.
        Stategy: "mean" | "freq" | "mean+freq"
        """
        mean_usage, freq_top = self._get_usage_stats(top_m=top_m)

        if strategy == 'mean':
            scores = mean_usage
        elif strategy == 'freq':
            scores = freq_top
        elif strategy == 'mean+freq':
            scores = (mean_usage + freq_top) / 2.0
        else:
            raise ValueError(f'Unknown strategy {strategy}')
        
        sorted_idx = torch.argsort(scores) # ascending = underused first
        return sorted_idx, scores