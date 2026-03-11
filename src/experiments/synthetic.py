# -*- coding: utf-8 -*-
"""Synthetic BUA/TDA + EMPA experiments: generate bodhi_vlm_metrics.csv and figures."""
import os
import numpy as np
from utils.metrics import compare_metrics, empa_bias_and_weights, histogram_from_samples, rmse
from utils.grouping import bua_style, tda_style


def generate_synthetic_layers(
    n_samples: int = 200,
    n_layers: int = 4,
    dim: int = 8,
    sensitive_ratio: float = 0.3,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)
    layers, sensitive_per_layer = [], []
    for _ in range(n_layers):
        f = rng.standard_normal((n_samples, dim)) * 2 + rng.uniform(-1, 1, (1, dim))
        layers.append(f)
        sensitive_per_layer.append(rng.random(n_samples) < sensitive_ratio)
    return layers, sensitive_per_layer


def add_privacy_noise(
    layers: list,
    sensitive_per_layer: list,
    epsilon: float = 0.1,
    scale: float = 1.0,
    seed: int = 43,
) -> list:
    sigma = scale / (epsilon + 1e-8)
    rng = np.random.default_rng(seed)
    noised = []
    for feat, sens in zip(layers, sensitive_per_layer):
        f = np.array(feat, dtype=float)
        noise = rng.standard_normal(f.shape) * sigma
        mask = np.broadcast_to(sens.reshape(-1, 1), f.shape)
        noised.append(f + noise * mask)
    return noised


def _empa_bias_random_partition(layers_noised, n_sensitive_per_layer, seed: int):
    """EMPA bias when using a random partition (same sensitive count per layer) as baseline."""
    rng = np.random.default_rng(seed)
    sens_list, nons_list = [], []
    for i, feat in enumerate(layers_noised):
        feat = np.asarray(feat)
        n = len(feat)
        n_sens = min(n_sensitive_per_layer[i], n)
        idx = rng.permutation(n)
        sens_list.append(feat[idx[:n_sens]])
        nons_list.append(feat[idx[n_sens:]])
    sens = np.vstack(sens_list) if sens_list else np.zeros((1, layers_noised[0].shape[1]))
    nons = np.vstack(nons_list) if nons_list else np.zeros((1, layers_noised[0].shape[1]))
    if sens.size == 0:
        sens = np.zeros((1, layers_noised[0].shape[1]))
    if nons.size == 0:
        nons = np.zeros((1, layers_noised[0].shape[1]))
    bias, _ = empa_bias_and_weights(sens, nons, n_components=5)
    return bias


def _run_one_config(layers_orig, layers_noised, sensitive_per_layer, bins=20, k_mdav=3, seed_random: int = 99) -> dict:
    flat_orig = np.concatenate([l.flatten() for l in layers_orig])
    flat_nois = np.concatenate([l.flatten() for l in layers_noised])
    baseline = compare_metrics(flat_orig, flat_nois, bins=bins)
    G_bua, Gp_bua = bua_style(layers_noised, sensitive_per_layer, k=k_mdav)
    sens_bua = np.concatenate([layers_noised[i][Gp_bua[i]] for i in range(len(Gp_bua)) if len(Gp_bua[i]) > 0])
    nons_bua = np.concatenate([layers_noised[i][G_bua[i]] for i in range(len(G_bua)) if len(G_bua[i]) > 0])
    if sens_bua.size == 0:
        sens_bua = np.zeros((1, layers_noised[0].shape[1]))
    if nons_bua.size == 0:
        nons_bua = np.zeros((1, layers_noised[0].shape[1]))
    bias_bua, _ = empa_bias_and_weights(sens_bua, nons_bua, n_components=5)
    G_tda, Gp_tda = tda_style(layers_noised, sensitive_per_layer, k=k_mdav)
    sens_tda = np.concatenate([layers_noised[i][Gp_tda[i]] for i in range(len(Gp_tda)) if len(Gp_tda[i]) > 0])
    nons_tda = np.concatenate([layers_noised[i][G_tda[i]] for i in range(len(G_tda)) if len(G_tda[i]) > 0])
    if sens_tda.size == 0:
        sens_tda = np.zeros((1, layers_noised[0].shape[1]))
    if nons_tda.size == 0:
        nons_tda = np.zeros((1, layers_noised[0].shape[1]))
    bias_tda, _ = empa_bias_and_weights(sens_tda, nons_tda, n_components=5)
    n_sens_per_layer = [max(1, int(np.sum(s))) for s in sensitive_per_layer]
    bias_random = _empa_bias_random_partition(layers_noised, n_sens_per_layer, seed=seed_random)
    return {
        "chi2": baseline["chi2"],
        "kl": baseline["kl"],
        "mmd": baseline["mmd"],
        "rmse": baseline["rmse"],
        "wass1": baseline.get("wass1", float("nan")),
        "empa_bias_bua": bias_bua,
        "empa_bias_tda": bias_tda,
        "empa_bias_random": bias_random,
    }


def run(config: dict, out_dir: str) -> None:
    """
    Run the synthetic experiment.

    Config keys:
      - bins: int
      - epsilon: list of floats
      - seeds: list of ints (or single base seed via `seed`)
      - n_samples: int
      - n_layers: int
    """
    os.makedirs(out_dir, exist_ok=True)
    bins = config.get("bins", 20)
    epsilons = config.get("epsilon", [0.1, 0.01])
    seeds = config.get("seeds")
    if seeds is None:
        seeds = [config.get("seed", 42)]
    n_samples = config.get("n_samples", 200)
    n_layers = config.get("n_layers", 4)

    rows = []
    for eps in epsilons:
        for s in seeds:
            layers_orig, sensitive_per_layer = generate_synthetic_layers(
                n_samples=n_samples, n_layers=n_layers, sensitive_ratio=0.3, seed=s
            )
            layers_noised = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=eps, seed=s + 1)
            res = _run_one_config(layers_orig, layers_noised, sensitive_per_layer, bins=bins, seed_random=s + 100)
            row = {"epsilon": eps, "seed": s, **res}
            rows.append(row)
            print(f"  synthetic epsilon={eps}, seed={s}: rmse={res['rmse']:.4f}, EMPA_bua={res['empa_bias_bua']:.4f}")

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "bodhi_vlm_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        summary = df.groupby("epsilon").agg(
            chi2_mean=("chi2", "mean"), chi2_std=("chi2", "std"),
            kl_mean=("kl", "mean"), kl_std=("kl", "std"),
            mmd_mean=("mmd", "mean"), mmd_std=("mmd", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            wass1_mean=("wass1", "mean"), wass1_std=("wass1", "std"),
            empa_bias_bua_mean=("empa_bias_bua", "mean"), empa_bias_bua_std=("empa_bias_bua", "std"),
            empa_bias_tda_mean=("empa_bias_tda", "mean"), empa_bias_tda_std=("empa_bias_tda", "std"),
        ).reset_index()
        summary.to_csv(os.path.join(out_dir, "bodhi_vlm_metrics_summary.csv"), index=False)
    except ImportError:
        csv_path = os.path.join(out_dir, "bodhi_vlm_metrics.csv")
        with open(csv_path, "w") as f:
            f.write("epsilon,seed,chi2,kl,mmd,rmse,wass1,empa_bias_bua,empa_bias_tda,empa_bias_random\n")
            for r in rows:
                f.write(f"{r['epsilon']},{r['seed']},{r['chi2']},{r['kl']},{r['mmd']},{r['rmse']},{r['wass1']},{r['empa_bias_bua']},{r['empa_bias_tda']},{r.get('empa_bias_random', float('nan'))}\n")
        print(f"Saved: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Deduplicate by epsilon for plotting (one point per epsilon; use mean over seeds if multiple)
        eps_unique = sorted(set(r["epsilon"] for r in rows))
        def mean_per_eps(key):
            return [np.nanmean([r[key] for r in rows if r["epsilon"] == e]) for e in eps_unique]
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))
        ax.plot(eps_unique, mean_per_eps("empa_bias_bua"), "o-", color="C0", label="BUA+EMPA", markersize=8, lw=2)
        ax.plot(eps_unique, mean_per_eps("empa_bias_tda"), "s-", color="C1", label="TDA+EMPA", markersize=8, lw=2)
        ax.plot(eps_unique, mean_per_eps("empa_bias_random"), "v--", color="C2", label="Random partition+EMPA (baseline)", markersize=7, lw=1.5)
        ax.set_xlabel("Privacy budget $\\epsilon$")
        ax.set_ylabel("EMPA bias")
        ax.legend(loc="best", fontsize=8)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bodhi_vlm_empa_bias.png"), dpi=150, bbox_inches="tight")
        plt.close()
        fig2, axes = plt.subplots(1, 2, figsize=(7, 3))
        ax0 = axes[0]
        ax0.semilogy(eps_unique, mean_per_eps("rmse"), "o-", color="C0", label="rMSE", markersize=8, lw=2)
        ax0.set_xlabel("Privacy budget $\\epsilon$")
        ax0.set_ylabel("rMSE", color="C0")
        ax0.tick_params(axis="y", labelcolor="C0")
        ax0.set_xscale("log")
        ax0.grid(True, alpha=0.3)
        ax0r = ax0.twinx()
        chi7 = [x / 1e7 for x in mean_per_eps("chi2")]
        ax0r.plot(eps_unique, chi7, "^-", color="C1", label="Chi-square ($\\times 10^{-7}$)", markersize=6, lw=1.5)
        ax0r.set_ylabel("Chi-square ($\\times 10^{-7}$)", color="C1")
        ax0r.tick_params(axis="y", labelcolor="C1")
        ax0.legend(loc="upper left", fontsize=7)
        ax0r.legend(loc="upper right", fontsize=7)
        axes[1].semilogy(eps_unique, mean_per_eps("chi2"), "^-", color="C1", label="Chi-square", markersize=6)
        axes[1].semilogy(eps_unique, mean_per_eps("kl"), "s-", color="C2", label="K-L div.", markersize=6)
        axes[1].semilogy(eps_unique, mean_per_eps("mmd"), "d-", color="C3", label="MMD", markersize=6)
        axes[1].set_xlabel("Privacy budget $\\epsilon$")
        axes[1].set_ylabel("Metric value")
        axes[1].set_xscale("log")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "bodhi_vlm_metrics_vs_epsilon.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: bodhi_vlm_empa_bias.png, bodhi_vlm_metrics_vs_epsilon.png")
    except ImportError:
        pass
