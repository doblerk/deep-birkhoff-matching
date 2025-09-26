import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from tribiged.utils.permutation import PermutationPool


class AlphaTracker:
    def __init__(self, k: int, warmup: int = 100, window: int = 50):
        self.k = k
        self.warmup = warmup
        self.window = window
        self.epoch_total = 0        # lifetime epoch counter
        self.epoch_in_window = 0    # counter since last reset
        self.history = list()

    def update(self, alphas: torch.Tensor, top_m: int = 2, strategy: str = 'mean'):
        """alphas: (N, k) from current epoch"""
        self.epoch_total += 1
        self.epoch_in_window += 1

        if self.epoch_total > self.warmup:
            mean_alphas = alphas.mean(dim=0).detach().cpu() # (k,)
            self.history.append(mean_alphas)

        # If window boundary is reached -> rank alphas, then reset
        if self.epoch_in_window == self.window:
            ranking = self._rank_alphas(top_m, strategy)
            self._reset()
            return ranking
        
        return None, None
    
    def _reset(self):
        """Clears history & reset epoch counter"""
        self.epoch_in_window = 0
        self.history.clear()
    
    def get_history(self):
        if len(self.history) == 0:
            return None
        return torch.stack(self.history, dim=0)
    
    def _get_usage_stats(self, top_m=2):
        history = self.get_history()
        if history is None:
            return None, None
        
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
        if mean_usage is None:
            return None, None

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