import npy as np
from scipy.stats import spearmanr, kendalltau


def precision_at_k(pred_matrix, true_matrix, k=10):
    pred_flat = pred_matrix.flatten()
    true_flat = true_matrix.flatten()

    topk_pred_indices = np.argpartition(-pred_flat, k)[:k]
    topk_true_indices = np.argpartition(-true_flat, k)[:k]

    topk_pred_set = set(topk_pred_indices)
    topk_true_set = set(topk_true_indices)

    intersection_count = len(topk_pred_set & topk_true_set)

    return intersection_count / k


def compute_rank_correlations(pred_matrix, true_matrix, k=10):
    # Spearman's rho
    spearman_rho, _ = spearmanr(pred_matrix, true_matrix, axis=None)

    # Kendall's tau
    kendall_tau, _ = kendalltau(pred_matrix.flatten(), true_matrix.flatten())

    # Precision at K
    p_at_k = precision_at_k(pred_matrix, true_matrix, k)
    return spearman_rho, kendall_tau, p_at_k