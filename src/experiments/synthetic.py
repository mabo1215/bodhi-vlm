# -*- coding: utf-8 -*-
"""Synthetic BUA/TDA + EMPA experiments: generate bodhi_vlm_metrics.csv and figures."""
import os
import numpy as np
from utils.metrics import (
    compare_metrics,
    empa_bias_and_weights,
    empa_reference_discrepancy,
    histogram_from_samples,
    rmse,
    moment_features,
    moment_reg_rmse,
    noise_mle_rmse_with_true,
    confidence_interval_95,
    budget_ranking_spearman,
    budget_ranking_accuracy_from_pairs,
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
    family: str = "gaussian",
) -> list:
    """Add noise to sensitive positions. family: 'gaussian' or 'laplace'."""
    c = scale / (epsilon + 1e-8)
    rng = np.random.default_rng(seed)
    noised = []
    for feat, sens in zip(layers, sensitive_per_layer):
        f = np.array(feat, dtype=float)
        if family == "laplace":
            # Laplace(0, b), b = c
            u = rng.uniform(1e-10, 1 - 1e-10, f.shape)
            noise = np.where(u < 0.5, c * np.log(2 * u), -c * np.log(2 * (1 - u)))
        else:
            sigma = c
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
        # Budget ranking: for each seed, (eps, score) pairs; fraction of seeds with correct rank
        pairs_per_seed = []
        for s in seeds:
            pairs = [(r["epsilon"], r["empa_bias_bua"]) for r in rows if r["seed"] == s]
            pairs_per_seed.append(pairs)
        br_acc = budget_ranking_accuracy_from_pairs(pairs_per_seed, lower_is_better=False)
        with open(os.path.join(out_dir, "bodhi_vlm_budget_ranking.txt"), "w") as f:
            f.write(f"budget_ranking_accuracy_empa_bua,{br_acc}\n")
        # 95% CI for each epsilon (empa_bias_bua)
        try:
            ci_rows = []
            for eps in epsilons:
                vals = [r["empa_bias_bua"] for r in rows if r["epsilon"] == eps]
                if len(vals) >= 2:
                    lo, hi = confidence_interval_95(np.array(vals))
                    ci_rows.append({"epsilon": eps, "empa_bua_ci_low": lo, "empa_bua_ci_high": hi})
            if ci_rows:
                pd.DataFrame(ci_rows).to_csv(os.path.join(out_dir, "bodhi_vlm_ci_per_epsilon.csv"), index=False)
        except Exception:
            pass
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
    # Out-of-family: observed noise family != reference family
    run_out_of_family_experiment(config, out_dir)
    # N5: noise only on top vs bottom half of layers to see BUA vs TDA diverge
    run_bua_tda_divergence_experiment(config, out_dir)
    # N4: threshold sensitivity (percentile 80, 85, 90, 95)
    run_threshold_sensitivity_experiment(config, out_dir)


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


def run_out_of_family_experiment(config: dict, out_dir: str) -> None:
    """
    Out-of-family test: observed noise from family A, reference from family B.
    Report EMPA reference discrepancy (L2 between fitted weights) for (obs=Gaussian, ref=Laplace) and (obs=Laplace, ref=Gaussian).
    """
    os.makedirs(out_dir, exist_ok=True)
    seeds = config.get("seeds") or [config.get("seed", 42)]
    epsilons = config.get("epsilon", [0.1, 0.01])
    n_samples = config.get("n_samples", 200)
    n_layers = config.get("n_layers", 4)
    k_mdav = 3
    n_components = 5

    rows = []
    for eps in epsilons:
        for s in seeds:
            layers_orig, sensitive_per_layer = generate_synthetic_layers(
                n_samples=n_samples, n_layers=n_layers, sensitive_ratio=0.3, seed=s
            )
            layers_g = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=eps, seed=s + 1, family="gaussian")
            layers_l = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=eps, seed=s + 2, family="laplace")
            # Partition from observed (Gaussian)
            G_bua, Gp_bua = bua_style(layers_g, sensitive_per_layer, k=k_mdav)
            sens_obs_g = np.concatenate([layers_g[i][Gp_bua[i]] for i in range(len(Gp_bua)) if len(Gp_bua[i]) > 0])
            nons_obs_g = np.concatenate([layers_g[i][G_bua[i]] for i in range(len(G_bua)) if len(G_bua[i]) > 0])
            sens_ref_l = np.concatenate([layers_l[i][Gp_bua[i]] for i in range(len(Gp_bua)) if len(Gp_bua[i]) > 0])
            nons_ref_l = np.concatenate([layers_l[i][G_bua[i]] for i in range(len(G_bua)) if len(G_bua[i]) > 0])
            if sens_obs_g.size == 0:
                sens_obs_g = np.zeros((1, layers_g[0].shape[1]))
            if nons_obs_g.size == 0:
                nons_obs_g = np.zeros((1, layers_g[0].shape[1]))
            if sens_ref_l.size == 0:
                sens_ref_l = np.zeros((1, layers_l[0].shape[1]))
            if nons_ref_l.size == 0:
                nons_ref_l = np.zeros((1, layers_l[0].shape[1]))
            d_obsG_refL = empa_reference_discrepancy(sens_obs_g, nons_obs_g, sens_ref_l, nons_ref_l, n_components=n_components)
            # Same with observed=Laplace, reference=Gaussian
            G_bua_l, Gp_bua_l = bua_style(layers_l, sensitive_per_layer, k=k_mdav)
            sens_obs_l = np.concatenate([layers_l[i][Gp_bua_l[i]] for i in range(len(Gp_bua_l)) if len(Gp_bua_l[i]) > 0])
            nons_obs_l = np.concatenate([layers_l[i][G_bua_l[i]] for i in range(len(G_bua_l)) if len(G_bua_l[i]) > 0])
            sens_ref_g = np.concatenate([layers_g[i][Gp_bua_l[i]] for i in range(len(Gp_bua_l)) if len(Gp_bua_l[i]) > 0])
            nons_ref_g = np.concatenate([layers_g[i][G_bua_l[i]] for i in range(len(G_bua_l)) if len(G_bua_l[i]) > 0])
            if sens_obs_l.size == 0:
                sens_obs_l = np.zeros((1, layers_l[0].shape[1]))
            if nons_obs_l.size == 0:
                nons_obs_l = np.zeros((1, layers_l[0].shape[1]))
            if sens_ref_g.size == 0:
                sens_ref_g = np.zeros((1, layers_g[0].shape[1]))
            if nons_ref_g.size == 0:
                nons_ref_g = np.zeros((1, layers_g[0].shape[1]))
            d_obsL_refG = empa_reference_discrepancy(sens_obs_l, nons_obs_l, sens_ref_g, nons_ref_g, n_components=n_components)
            rows.append({
                "epsilon": eps, "seed": s,
                "discrepancy_obsGaussian_refLaplace": d_obsG_refL,
                "discrepancy_obsLaplace_refGaussian": d_obsL_refG,
            })
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "out_of_family_discrepancy.csv"), index=False)
        print("Saved: out_of_family_discrepancy.csv (obs/reference family mismatch)")
    except ImportError:
        with open(os.path.join(out_dir, "out_of_family_discrepancy.csv"), "w") as f:
            f.write("epsilon,seed,discrepancy_obsGaussian_refLaplace,discrepancy_obsLaplace_refGaussian\n")
            for r in rows:
                f.write(f"{r['epsilon']},{r['seed']},{r['discrepancy_obsGaussian_refLaplace']},{r['discrepancy_obsLaplace_refGaussian']}\n")


def run_bua_tda_divergence_experiment(config: dict, out_dir: str) -> None:
    """
    N5: Noise only on top half vs bottom half of layers to see if BUA and TDA diverge.
    - late_only: noise only on layers 2,3 (top); early_only: noise only on layers 0,1 (bottom).
    """
    os.makedirs(out_dir, exist_ok=True)
    seeds = config.get("seeds") or [config.get("seed", 42)]
    n_samples = config.get("n_samples", 200)
    n_layers = 4
    eps = 0.1
    bins = config.get("bins", 20)
    rows = []
    for s in seeds:
        layers_orig, sensitive_per_layer = generate_synthetic_layers(
            n_samples=n_samples, n_layers=n_layers, sensitive_ratio=0.3, seed=s
        )
        # Late-only noise (top layers 2,3)
        layers_late = [layers_orig[i].copy() for i in range(n_layers)]
        for i in (2, 3):
            sigma = 1.0 / (eps + 1e-8)
            rng = np.random.default_rng(s + 1)
            mask = np.broadcast_to(sensitive_per_layer[i].reshape(-1, 1), layers_late[i].shape)
            layers_late[i] = layers_late[i] + rng.standard_normal(layers_late[i].shape) * sigma * mask
        res_late = _run_one_config(layers_orig, layers_late, sensitive_per_layer, bins=bins, seed_random=s + 100)
        # Early-only noise (bottom layers 0,1)
        layers_early = [layers_orig[i].copy() for i in range(n_layers)]
        for i in (0, 1):
            sigma = 1.0 / (eps + 1e-8)
            rng = np.random.default_rng(s + 2)
            mask = np.broadcast_to(sensitive_per_layer[i].reshape(-1, 1), layers_early[i].shape)
            layers_early[i] = layers_early[i] + rng.standard_normal(layers_early[i].shape) * sigma * mask
        res_early = _run_one_config(layers_orig, layers_early, sensitive_per_layer, bins=bins, seed_random=s + 101)
        rows.append({
            "seed": s, "regime": "noise_late_only",
            "empa_bias_bua": res_late["empa_bias_bua"], "empa_bias_tda": res_late["empa_bias_tda"],
        })
        rows.append({
            "seed": s, "regime": "noise_early_only",
            "empa_bias_bua": res_early["empa_bias_bua"], "empa_bias_tda": res_early["empa_bias_tda"],
        })
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "bua_tda_divergence_noise_placement.csv"), index=False)
        print("Saved: bua_tda_divergence_noise_placement.csv (N5: noise late-only vs early-only)")
    except ImportError:
        pass


def run_threshold_sensitivity_experiment(config: dict, out_dir: str) -> None:
    """
    N4: Sensitivity to threshold percentile. Simulate score = oracle + small noise, threshold at 80, 85, 90, 95.
    """
    os.makedirs(out_dir, exist_ok=True)
    seeds = config.get("seeds") or [config.get("seed", 42)]
    n_samples = config.get("n_samples", 200)
    n_layers = config.get("n_layers", 4)
    eps = 0.1
    percentiles = [0.80, 0.85, 0.90, 0.95]
    rows = []
    for s in seeds:
        layers_orig, sensitive_per_layer = generate_synthetic_layers(
            n_samples=n_samples, n_layers=n_layers, sensitive_ratio=0.3, seed=s
        )
        layers_noised = add_privacy_noise(layers_orig, sensitive_per_layer, epsilon=eps, seed=s + 1)
        for q in percentiles:
            # Simulate score: 1 for sensitive, 0 for non-sensitive, plus small noise; then threshold at q
            rng = np.random.default_rng(s + int(q * 100))
            sens_per_layer_q = []
            for sens in sensitive_per_layer:
                score = sens.astype(float) + rng.uniform(-0.1, 0.1, sens.shape)
                thresh = np.percentile(score, q * 100)
                sens_per_layer_q.append(score >= thresh)
            res = _run_one_config(layers_orig, layers_noised, sens_per_layer_q, bins=20, seed_random=s + 200)
            rows.append({"seed": s, "percentile": q, "empa_bias_bua": res["empa_bias_bua"], "empa_bias_tda": res["empa_bias_tda"]})
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "threshold_sensitivity.csv"), index=False)
        summary = df.groupby("percentile").agg(
            empa_bua_mean=("empa_bias_bua", "mean"), empa_bua_std=("empa_bias_bua", "std"),
            empa_tda_mean=("empa_bias_tda", "mean"), empa_tda_std=("empa_bias_tda", "std"),
        ).reset_index()
        summary.to_csv(os.path.join(out_dir, "threshold_sensitivity_summary.csv"), index=False)
        print("Saved: threshold_sensitivity.csv, threshold_sensitivity_summary.csv (N4: tau in {80,85,90,95}%)")
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
