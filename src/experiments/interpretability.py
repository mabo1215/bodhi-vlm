# -*- coding: utf-8 -*-
"""Interpretability figures: sensitive vs non-sensitive distributions and t-SNE/PCA."""
import os
import numpy as np
from experiments.synthetic import generate_synthetic_layers, add_privacy_noise
from utils.grouping import bua_style


def _plot_sensitive_distribution(layers_noised, G_list, Gp_list, layer_id=0, bins=25, out_path="bodhi_vlm_sensitive_dist.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feat = np.asarray(layers_noised[layer_id])
    G_idx, Gp_idx = G_list[layer_id], Gp_list[layer_id]
    sens_flat = np.asarray(feat[Gp_idx]).flatten() if len(Gp_idx) > 0 else np.array([])
    nons_flat = np.asarray(feat[G_idx]).flatten() if len(G_idx) > 0 else np.array([])
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    if len(sens_flat) > 0:
        ax.hist(sens_flat, bins=bins, alpha=0.6, label="Sensitive ($G'_i$)", color="C1", density=True)
    if len(nons_flat) > 0:
        ax.hist(nons_flat, bins=bins, alpha=0.6, label="Non-sensitive ($G_i$)", color="C0", density=True)
    ax.set_xlabel("Feature value (noised)")
    ax.set_ylabel("Density")
    ax.set_title("Sensitive vs. non-sensitive feature distribution (BUA, layer 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _plot_tsne_interpretability(layers_noised, G_list, Gp_list, out_path="bodhi_vlm_tsne.png", perplexity=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    all_feat, all_sensitive = [], []
    for feat, G_idx, Gp_idx in zip(layers_noised, G_list, Gp_list):
        feat = np.asarray(feat)
        n = len(feat)
        sens_mask = np.zeros(n, dtype=bool)
        sens_mask[Gp_idx] = True
        all_feat.append(feat)
        all_sensitive.append(sens_mask)
    X = np.vstack(all_feat)
    sens_labels = np.concatenate(all_sensitive)
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, X.shape[0] - 1, X.shape[1]))
        X = pca.fit_transform(X)
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, max(5, X.shape[0] // 4)))
        X_2d = tsne.fit_transform(X)
        title_suffix = "t-SNE"
    except Exception:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        title_suffix = "PCA"
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(X_2d[~sens_labels, 0], X_2d[~sens_labels, 1], c="C0", alpha=0.5, s=12, label="Non-sensitive ($G_i$)")
    ax.scatter(X_2d[sens_labels, 0], X_2d[sens_labels, 1], c="C1", alpha=0.5, s=12, label="Sensitive ($G'_i$)")
    ax.set_xlabel(f"{title_suffix} 1")
    ax.set_ylabel(f"{title_suffix} 2")
    ax.set_title(f"Feature space by BUA grouping ({title_suffix})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def run(config: dict, out_dir: str) -> None:
    """
    Run interpretability experiment.

    Config keys:
      - n_samples: int
      - n_layers: int
      - epsilon: float

    Writes bodhi_vlm_sensitive_dist.png and bodhi_vlm_tsne.png into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_samples = config.get("n_samples", 200)
    n_layers = config.get("n_layers", 4)
    epsilon = config.get("epsilon", 0.1)
    layers_orig, sensitive_per_layer = generate_synthetic_layers(
        n_samples=n_samples, n_layers=n_layers, sensitive_ratio=0.3, seed=42
    )
    layers_noised = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=epsilon)
    G_list, Gp_list = bua_style(layers_noised, sensitive_per_layer, k=3)
    _plot_sensitive_distribution(
        layers_noised, G_list, Gp_list,
        out_path=os.path.join(out_dir, "bodhi_vlm_sensitive_dist.png"),
    )
    _plot_tsne_interpretability(
        layers_noised, G_list, Gp_list,
        out_path=os.path.join(out_dir, "bodhi_vlm_tsne.png"),
    )
