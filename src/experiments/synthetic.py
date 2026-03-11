# -*- coding: utf-8 -*-
"""Synthetic BUA/TDA + EMPA experiments: generate bodhi_vlm_metrics.csv and figures."""
import os
import numpy as np
from utils.metrics import (
    compare_metrics,
    empa_bias_and_weights,
    histogram_from_samples,
    rmse,
    moment_features,
    moment_reg_rmse,
    noise_mle_rmse_with_true,
)
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


def _empa_bias_random_partition(layers_noised, n_sensitive_per_layer, seed: int, n_components: int = 5):
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
    bias, _ = empa_bias_and_weights(sens, nons, n_components=n_components)
    return bias


def _run_one_config(
    layers_orig,
    layers_noised,
    sensitive_per_layer,
    bins=20,
    k_mdav=3,
    seed_random: int = 99,
    n_components: int = 5,
) -> dict:
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
    bias_bua, _ = empa_bias_and_weights(sens_bua, nons_bua, n_components=n_components)
    G_tda, Gp_tda = tda_style(layers_noised, sensitive_per_layer, k=k_mdav)
    sens_tda = np.concatenate([layers_noised[i][Gp_tda[i]] for i in range(len(Gp_tda)) if len(Gp_tda[i]) > 0])
    nons_tda = np.concatenate([layers_noised[i][G_tda[i]] for i in range(len(G_tda)) if len(G_tda[i]) > 0])
    if sens_tda.size == 0:
        sens_tda = np.zeros((1, layers_noised[0].shape[1]))
    if nons_tda.size == 0:
        nons_tda = np.zeros((1, layers_noised[0].shape[1]))
    bias_tda, _ = empa_bias_and_weights(sens_tda, nons_tda, n_components=n_components)
    n_sens_per_layer = [max(1, int(np.sum(s))) for s in sensitive_per_layer]
    bias_random = _empa_bias_random_partition(
        layers_noised, n_sens_per_layer, seed=seed_random, n_components=n_components
    )
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
    list_eps, list_orig, list_noised = [], [], []
    for eps in epsilons:
        for s in seeds:
            layers_orig, sensitive_per_layer = generate_synthetic_layers(
                n_samples=n_samples, n_layers=n_layers, sensitive_ratio=0.3, seed=s
            )
            layers_noised = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=eps, seed=s + 1)
            flat_orig = np.concatenate([l.flatten() for l in layers_orig])
            flat_noised = np.concatenate([l.flatten() for l in layers_noised])
            list_eps.append(eps)
            list_orig.append(flat_orig)
            list_noised.append(flat_noised)
            res = _run_one_config(layers_orig, layers_noised, sensitive_per_layer, bins=bins, seed_random=s + 100)
            row = {"epsilon": eps, "seed": s, **res}
            rows.append(row)
            print(f"  synthetic epsilon={eps}, seed={s}: rmse={res['rmse']:.4f}, EMPA_bua={res['empa_bias_bua']:.4f}")

    # Task-matched baselines: MomentReg (regress epsilon from moment features), NoiseMLE (grid MLE)
    eps_grid = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    moment_reg_rms = float("nan")
    noise_mle_rms_list = []
    if len(list_eps) >= 2:
        moment_reg_rms = moment_reg_rmse(np.array(list_eps), list(zip(list_orig, list_noised)))
    for i in range(len(list_eps)):
        rms_i = noise_mle_rmse_with_true(
            list_orig[i], list_noised[i], list_eps[i], eps_grid, scale=1.0
        )
        noise_mle_rms_list.append(rms_i)
    noise_mle_rms_mean = float(np.nanmean(noise_mle_rms_list)) if noise_mle_rms_list else float("nan")
    with open(os.path.join(out_dir, "bodhi_vlm_baselines.txt"), "w") as f:
        f.write(f"moment_reg_rmse,{moment_reg_rms}\n")
        f.write(f"noise_mle_rmse_mean,{noise_mle_rms_mean}\n")
    print(f"  Task-matched baselines: MomentReg rMSE={moment_reg_rms:.4f}, NoiseMLE rMSE (mean)={noise_mle_rms_mean:.4f}")

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

    # Decomposition and ablation (optional, same out_dir)
    run_decomposition_experiment(config, out_dir)
    run_ablation_experiment(config, out_dir)


def run_decomposition_experiment(config: dict, out_dir: str) -> None:
    """
    (1) Fixed partition, vary epsilon: one partition (BUA at epsilon=0.1), re-noise at each epsilon, report EMPA bias.
    (2) Fixed epsilon, vary partition: one noised set at epsilon=0.1, report bias for BUA, TDA, random.
    """
    os.makedirs(out_dir, exist_ok=True)
    seeds = config.get("seeds") or [config.get("seed", 42)]
    epsilons = config.get("epsilon", [0.1, 0.01])
    n_samples = config.get("n_samples", 200)
    n_layers = config.get("n_layers", 4)
    bins = config.get("bins", 20)
    seed0 = seeds[0]

    # (1) Fixed partition, vary epsilon
    layers_orig, sensitive_per_layer = generate_synthetic_layers(
        n_samples=n_samples, n_layers=n_layers, sensitive_ratio=0.3, seed=seed0
    )
    layers_noised_anchor = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=0.1, seed=seed0 + 1)
    G_bua_fixed, Gp_bua_fixed = bua_style(layers_noised_anchor, sensitive_per_layer, k=3)

    fixed_partition_rows = []
    for eps in epsilons:
        layers_noised = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=eps, seed=seed0 + 1)
        sens = np.concatenate([layers_noised[i][Gp_bua_fixed[i]] for i in range(len(Gp_bua_fixed)) if len(Gp_bua_fixed[i]) > 0])
        nons = np.concatenate([layers_noised[i][G_bua_fixed[i]] for i in range(len(G_bua_fixed)) if len(G_bua_fixed[i]) > 0])
        if sens.size == 0:
            sens = np.zeros((1, layers_noised[0].shape[1]))
        if nons.size == 0:
            nons = np.zeros((1, layers_noised[0].shape[1]))
        bias, _ = empa_bias_and_weights(sens, nons, n_components=5)
        fixed_partition_rows.append({"epsilon": eps, "empa_bias_fixed_partition": bias})

    # (2) Fixed epsilon, vary partition (one noised set, BUA vs TDA vs random)
    layers_noised_fixed = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=0.1, seed=seed0 + 1)
    res = _run_one_config(layers_orig, layers_noised_fixed, sensitive_per_layer, bins=bins, seed_random=seed0 + 100)
    fixed_epsilon_rows = [
        {"partition": "BUA", "empa_bias": res["empa_bias_bua"]},
        {"partition": "TDA", "empa_bias": res["empa_bias_tda"]},
        {"partition": "random", "empa_bias": res["empa_bias_random"]},
    ]

    try:
        import pandas as pd
        pd.DataFrame(fixed_partition_rows).to_csv(
            os.path.join(out_dir, "decomposition_fixed_partition_vs_epsilon.csv"), index=False
        )
        pd.DataFrame(fixed_epsilon_rows).to_csv(
            os.path.join(out_dir, "decomposition_fixed_epsilon_vs_partition.csv"), index=False
        )
        print("Saved: decomposition_fixed_partition_vs_epsilon.csv, decomposition_fixed_epsilon_vs_partition.csv")
    except ImportError:
        pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.8))
        ep_vals = [r["epsilon"] for r in fixed_partition_rows]
        bias_vals = [r["empa_bias_fixed_partition"] for r in fixed_partition_rows]
        ax.plot(ep_vals, bias_vals, "o-", color="C0", markersize=8, lw=2)
        ax.set_xlabel("Privacy budget $\\epsilon$")
        ax.set_ylabel("EMPA bias (fixed BUA partition)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "decomposition_fixed_partition_bias_vs_epsilon.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: decomposition_fixed_partition_bias_vs_epsilon.png")
    except ImportError:
        pass


def run_ablation_experiment(config: dict, out_dir: str) -> None:
    """Vary K in {2,4,8}, sensitive_ratio in {0.2, 0.3, 0.4}, partition type (BUA, TDA, random)."""
    os.makedirs(out_dir, exist_ok=True)
    seeds = config.get("seeds") or [config.get("seed", 42)]
    n_samples = config.get("n_samples", 200)
    n_layers = config.get("n_layers", 4)
    bins = config.get("bins", 20)
    K_list = [2, 4, 8]
    ratios = [0.2, 0.3, 0.4]
    eps = 0.1

    rows = []
    for K in K_list:
        for ratio in ratios:
            for s in seeds:
                layers_orig, sensitive_per_layer = generate_synthetic_layers(
                    n_samples=n_samples, n_layers=n_layers, sensitive_ratio=ratio, seed=s
                )
                layers_noised = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=eps, seed=s + 1)
                res = _run_one_config(
                    layers_orig, layers_noised, sensitive_per_layer,
                    bins=bins, seed_random=s + 100, n_components=K
                )
                rows.append({
                    "K": K, "sensitive_ratio": ratio, "seed": s,
                    "empa_bias_bua": res["empa_bias_bua"],
                    "empa_bias_tda": res["empa_bias_tda"],
                    "empa_bias_random": res["empa_bias_random"],
                })
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "ablation_K_ratio_partition.csv"), index=False)
        summary = df.groupby(["K", "sensitive_ratio"]).agg(
            empa_bua_mean=("empa_bias_bua", "mean"), empa_bua_std=("empa_bias_bua", "std"),
            empa_tda_mean=("empa_bias_tda", "mean"), empa_tda_std=("empa_bias_tda", "std"),
            empa_random_mean=("empa_bias_random", "mean"), empa_random_std=("empa_bias_random", "std"),
        ).reset_index()
        summary.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False)
        print("Saved: ablation_K_ratio_partition.csv, ablation_summary.csv")
    except ImportError:
        pass
