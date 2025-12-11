import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, svd


def batched_diagnostics(M_batch):
    """
    Computes diagnostics for a batch of DSM for convergence monitoring:
        - spectral gap (1 - |lambda2|)
        - mean singular value
        - mean row entropy (how peaked rows are)
    """
    B, _, _ = M_batch.shape

    spectral_gaps = np.zeros(B)
    top_singular_values = np.zeros(B)
    mean_singular_values = np.zeros(B)
    mean_entropies = np.zeros(B)

    eps = 1e-12

    for b in range(B):
        M = M_batch[b]

        # Eigenvalues
        ev = eigvals(M)
        radii = np.abs(ev)
        order = np.argsort(radii)[::-1]
        leading_mod = radii[order[0]]
        second_mod = radii[order[1]]
        spectral_gaps[b] = leading_mod - second_mod

        # Singular values
        sv = svd(M, compute_uv=False)
        top_singular_values[b] = sv[0]
        mean_singular_values[b] = sv.mean()

        # Row entropies
        row_entropy = -np.sum(M * np.log(M + eps), axis=1)
        mean_entropies[b] = row_entropy.mean()

    return {
        "spectral_gap": spectral_gaps,
        "top_singular_value": top_singular_values,
        "mean_singular_value": mean_singular_values,
        "mean_entropy": mean_entropies
    }


def accumulate_epoch_stats(M_batch, history):
    stats = batched_diagnostics(M_batch)
    # Average across batch for monitoring
    history["spectral_gap"].append(stats["spectral_gap"].mean())
    history["top_singular_value"].append(stats["top_singular_value"].mean())
    history["mean_singular_value"].append(stats["mean_singular_value"].mean())
    history["mean_entropy"].append(stats["mean_entropy"].mean())
    return history


def plot_history(history):
    epochs = np.arange(len(history["spectral_gap"]))
    plt.figure(figsize=(12,6))
    for i, key in enumerate(history.keys(), 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, history[key], marker="o")
        plt.title(key)
        plt.xlabel("epoch")
        plt.ylabel(key)
    plt.tight_layout()
    plt.show()