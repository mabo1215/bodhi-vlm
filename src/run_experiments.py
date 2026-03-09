# -*- coding: utf-8 -*-
"""
Bodhi VLM experiment runner: BUA vs TDA, EMPA vs Chi-square / K-L / MMD.
Generates synthetic hierarchical features (or accepts numpy arrays), adds privacy
noise, runs grouping and metrics, and outputs comparison table and plots.
Usage:
  python run_experiments.py [--out_dir OUT] [--bins BINS] [--epsilon EPS]
"""

import argparse
import os
import sys
import numpy as np

# Add parent so scripts can import from same package if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import (
    compare_metrics,
    empa_bias_and_weights,
    histogram_from_samples,
    rmse,
)
from grouping import bua_style, tda_style


def generate_synthetic_layers(
    n_samples: int = 200,
    n_layers: int = 4,
    dim: int = 8,
    sensitive_ratio: float = 0.3,
    seed: int = 42,
) -> tuple:
    """Generate synthetic multi-layer features and sensitive labels."""
    rng = np.random.default_rng(seed)
    layers = []
    sensitive_per_layer = []
    for _ in range(n_layers):
        f = rng.standard_normal((n_samples, dim)) * 2 + rng.uniform(-1, 1, (1, dim))
        layers.append(f)
        sens = rng.random(n_samples) < sensitive_ratio
        sensitive_per_layer.append(sens)
    return layers, sensitive_per_layer


def add_privacy_noise(
    layers: list,
    sensitive_per_layer: list,
    epsilon: float = 0.1,
    scale: float = 1.0,
) -> list:
    """Add Gaussian noise to sensitive positions (scale ~ 1/epsilon)."""
    noised = []
    sigma = scale / (epsilon + 1e-8)
    rng = np.random.default_rng(43)
    for feat, sens in zip(layers, sensitive_per_layer):
        f = np.array(feat, dtype=float)
        noise = rng.standard_normal(f.shape) * sigma
        mask = np.broadcast_to(sens.reshape(-1, 1), f.shape)
        f = f + noise * mask
        noised.append(f)
    return noised


def run_one_config(
    layers_orig: list,
    layers_noised: list,
    sensitive_per_layer: list,
    bins: int = 20,
    k_mdav: int = 3,
) -> dict:
    """Run BUA, TDA, EMPA, and baseline metrics for one (orig, noised) pair."""
    # Flatten for global metrics (use last layer as "output" proxy)
    flat_orig = np.concatenate([l.flatten() for l in layers_orig])
    flat_nois = np.concatenate([l.flatten() for l in layers_noised])
    baseline = compare_metrics(flat_orig, flat_nois, bins=bins)

    # BUA grouping on noised layers
    G_bua, Gp_bua = bua_style(layers_noised, sensitive_per_layer, k=k_mdav)
    sens_bua = np.concatenate([layers_noised[i][Gp_bua[i]] for i in range(len(Gp_bua)) if len(Gp_bua[i]) > 0])
    nons_bua = np.concatenate([layers_noised[i][G_bua[i]] for i in range(len(G_bua)) if len(G_bua[i]) > 0])
    if sens_bua.size == 0:
        sens_bua = np.zeros((1, layers_noised[0].shape[1]))
    if nons_bua.size == 0:
        nons_bua = np.zeros((1, layers_noised[0].shape[1]))
    bias_bua, _ = empa_bias_and_weights(sens_bua, nons_bua, n_components=5)

    # TDA grouping on noised layers
    G_tda, Gp_tda = tda_style(layers_noised, sensitive_per_layer, k=k_mdav)
    sens_tda = np.concatenate([layers_noised[i][Gp_tda[i]] for i in range(len(Gp_tda)) if len(Gp_tda[i]) > 0])
    nons_tda = np.concatenate([layers_noised[i][G_tda[i]] for i in range(len(G_tda)) if len(G_tda[i]) > 0])
    if sens_tda.size == 0:
        sens_tda = np.zeros((1, layers_noised[0].shape[1]))
    if nons_tda.size == 0:
        nons_tda = np.zeros((1, layers_noised[0].shape[1]))
    bias_tda, _ = empa_bias_and_weights(sens_tda, nons_tda, n_components=5)

    return {
        "chi2": baseline["chi2"],
        "kl": baseline["kl"],
        "mmd": baseline["mmd"],
        "rmse": baseline["rmse"],
        "empa_bias_bua": bias_bua,
        "empa_bias_tda": bias_tda,
    }


def main():
    parser = argparse.ArgumentParser(description="Bodhi VLM experiments: BUA/TDA + EMPA vs baselines")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory for CSV and plots")
    parser.add_argument("--bins", type=int, default=20, help="Histogram bins for Chi2/KL")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.1, 0.01], help="Privacy budgets")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Bodhi VLM experiment: BUA/TDA + EMPA vs Chi2, K-L, MMD")
    print("=" * 60)

    rows = []
    for eps in args.epsilon:
        layers_orig, sensitive_per_layer = generate_synthetic_layers(
            n_samples=args.n_samples,
            n_layers=args.n_layers,
            sensitive_ratio=0.3,
            seed=args.seed,
        )
        layers_noised = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=eps)
        res = run_one_config(
            layers_orig,
            layers_noised,
            sensitive_per_layer,
            bins=args.bins,
        )
        row = {"epsilon": eps, **res}
        rows.append(row)
        print(f"  epsilon={eps}: rmse={res['rmse']:.4f}, chi2={res['chi2']:.4f}, kl={res['kl']:.6f}, mmd={res['mmd']:.4f}, EMPA_bua={res['empa_bias_bua']:.4f}, EMPA_tda={res['empa_bias_tda']:.4f}")

    # Summary table
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.out_dir, "bodhi_vlm_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
    except ImportError:
        # No pandas: write simple CSV
        csv_path = os.path.join(args.out_dir, "bodhi_vlm_metrics.csv")
        with open(csv_path, "w") as f:
            f.write("epsilon,chi2,kl,mmd,rmse,empa_bias_bua,empa_bias_tda\n")
            for r in rows:
                f.write(f"{r['epsilon']},{r['chi2']},{r['kl']},{r['mmd']},{r['rmse']},{r['empa_bias_bua']},{r['empa_bias_tda']}\n")
        print(f"\nSaved: {csv_path}")

    # Plot if matplotlib available (paper-ready figures)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps_list = [r["epsilon"] for r in rows]
        # Figure 1: EMPA bias vs epsilon
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.8))
        ax.plot(eps_list, [r["empa_bias_bua"] for r in rows], "o-", label="BUA+EMPA bias", markersize=8)
        ax.plot(eps_list, [r["empa_bias_tda"] for r in rows], "s-", label="TDA+EMPA bias", markersize=8)
        ax.set_xlabel("Privacy budget $\\epsilon$")
        ax.set_ylabel("EMPA bias")
        ax.legend()
        ax.set_title("Bodhi VLM: EMPA bias vs budget")
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "bodhi_vlm_empa_bias.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {os.path.join(args.out_dir, 'bodhi_vlm_empa_bias.png')}")

        # Figure 2: rMSE and metrics vs epsilon (for paper)
        fig2, axes = plt.subplots(1, 2, figsize=(6, 2.6))
        ax1, ax2 = axes[0], axes[1]
        ax1.semilogy(eps_list, [r["rmse"] for r in rows], "o-", color="C0", label="rMSE", markersize=8)
        ax1.set_xlabel("Privacy budget $\\epsilon$")
        ax1.set_ylabel("rMSE")
        ax1.set_title("rMSE vs $\\epsilon$")
        ax1.set_xscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.semilogy(eps_list, [r["chi2"] for r in rows], "^-", color="C1", label="Chi-square", markersize=6)
        ax2.semilogy(eps_list, [r["kl"] for r in rows], "s-", color="C2", label="K-L div.", markersize=6)
        ax2.semilogy(eps_list, [r["mmd"] for r in rows], "d-", color="C3", label="MMD", markersize=6)
        ax2.set_xlabel("Privacy budget $\\epsilon$")
        ax2.set_ylabel("Metric value")
        ax2.set_title("Baseline metrics vs $\\epsilon$")
        ax2.set_xscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.out_dir, "bodhi_vlm_metrics_vs_epsilon.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {os.path.join(args.out_dir, 'bodhi_vlm_metrics_vs_epsilon.png')}")
    except ImportError:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
