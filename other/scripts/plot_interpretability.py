# -*- coding: utf-8 -*-
"""
Bodhi VLM interpretability figures for Section 4.4:
(1) Sensitive vs. non-sensitive feature distribution.
(2) t-SNE (or PCA) visualization of features colored by sensitive/non-sensitive.

Usage (from project root or paper/):

  python paper/scripts/plot_interpretability.py --out_dir paper/images
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from run_experiments import (  # type: ignore
    generate_synthetic_layers,
    add_privacy_noise,
)
from grouping import bua_style  # type: ignore


def plot_sensitive_distribution(
    layers_noised: list,
    G_list: list,
    Gp_list: list,
    layer_id: int = 0,
    bins: int = 25,
    out_path: str = "bodhi_vlm_sensitive_dist.png",
) -> None:
    """Plot distribution of feature values: sensitive vs. non-sensitive (one layer)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat = np.asarray(layers_noised[layer_id])
    G_idx, Gp_idx = G_list[layer_id], Gp_list[layer_id]
    if len(Gp_idx) > 0:
        sens_flat = np.asarray(feat[Gp_idx]).flatten()
    else:
        sens_flat = np.array([])
    if len(G_idx) > 0:
        nons_flat = np.asarray(feat[G_idx]).flatten()
    else:
        nons_flat = np.array([])

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


def plot_tsne_interpretability(
    layers_noised: list,
    G_list: list,
    Gp_list: list,
    out_path: str = "bodhi_vlm_tsne.png",
    perplexity: int = 30,
) -> None:
    """t-SNE of all layer features colored by sensitive (red) vs non-sensitive (blue)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Stack all layers into (N_total, D); record sensitive label per point
    all_feat = []
    all_sensitive = []
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bodhi VLM interpretability figures")
    parser.add_argument("--out_dir", type=str, default="paper/images", help="Output directory (e.g. paper/images)")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers_orig, sensitive_per_layer = generate_synthetic_layers(
        n_samples=args.n_samples,
        n_layers=args.n_layers,
        sensitive_ratio=0.3,
        seed=42,
    )
    layers_noised = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=args.epsilon)
    G_list, Gp_list = bua_style(layers_noised, sensitive_per_layer, k=3)

    plot_sensitive_distribution(
        layers_noised,
        G_list,
        Gp_list,
        layer_id=0,
        out_path=str(out_dir / "bodhi_vlm_sensitive_dist.png"),
    )
    plot_tsne_interpretability(
        layers_noised,
        G_list,
        Gp_list,
        out_path=str(out_dir / "bodhi_vlm_tsne.png"),
    )
    print("Done.")


if __name__ == "__main__":
    main()

