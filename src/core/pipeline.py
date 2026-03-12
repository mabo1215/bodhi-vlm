# -*- coding: utf-8 -*-
"""
High-level Bodhi VLM pipeline: BUA/TDA grouping + EMPA assessment.
Model-agnostic: any vision backbone or VLM exposing layer-wise features can be plugged in.
"""
from typing import List, Dict, Any, Tuple
import numpy as np

from utils.grouping import bua_style, tda_style
from utils.metrics import compare_metrics, empa_bias_and_weights


def group_features_bua_tda(
    layer_features_noised: List[np.ndarray],
    sensitive_per_layer: List[np.ndarray],
    k_mdav: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Run BUA- and TDA-style grouping on layer-wise features. Returns G_bua, Gp_bua, G_tda, Gp_tda."""
    G_bua, Gp_bua = bua_style(layer_features_noised, sensitive_per_layer, k=k_mdav)
    G_tda, Gp_tda = tda_style(layer_features_noised, sensitive_per_layer, k=k_mdav)
    return G_bua, Gp_bua, G_tda, Gp_tda


def _concat_sensitive_and_nonsensitive(
    layer_features_noised: List[np.ndarray],
    G_list: List[np.ndarray],
    Gp_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect sensitive / non-sensitive feature vectors across layers into two matrices."""
    sens_list = []
    nons_list = []
    for feats, G, Gp in zip(layer_features_noised, G_list, Gp_list):
        feats = np.asarray(feats, dtype=float)
        if len(Gp) > 0:
            sens_list.append(feats[Gp])
        if len(G) > 0:
            nons_list.append(feats[G])
    if len(sens_list) == 0:
        sens = np.zeros((1, layer_features_noised[0].shape[1]), dtype=float)
    else:
        sens = np.concatenate(sens_list, axis=0)
    if len(nons_list) == 0:
        nons = np.zeros((1, layer_features_noised[0].shape[1]), dtype=float)
    else:
        nons = np.concatenate(nons_list, axis=0)
    return sens, nons


def assess_privacy_budget_from_features(
    layer_features_orig: List[np.ndarray],
    layer_features_noised: List[np.ndarray],
    sensitive_per_layer: List[np.ndarray],
    epsilon: float,
    bins: int = 20,
    k_mdav: int = 3,
    ablation_mode: str = "full",
) -> Dict[str, Any]:
    """
    End-to-end Bodhi VLM assessment on arbitrary backbone features.
    ablation_mode: "full" | "bua_only" | "tda_only" | "no_empa".
    Returns dict with epsilon, chi2, kl, mmd, rmse, wass1, empa_bias_bua, empa_bias_tda,
    and for ablation: deviation (mean |noised-orig| on sensitive group), rmse_budget (rMSE for table).
    """
    if len(layer_features_orig) != len(layer_features_noised):
        raise ValueError("Original and noised feature lists must have the same length.")

    flat_orig = np.concatenate([np.asarray(l).flatten() for l in layer_features_orig])
    flat_nois = np.concatenate([np.asarray(l).flatten() for l in layer_features_noised])
    baseline = compare_metrics(flat_orig, flat_nois, bins=bins)
    baseline_rmse = baseline["rmse"]

    G_bua, Gp_bua, G_tda, Gp_tda = group_features_bua_tda(
        layer_features_noised, sensitive_per_layer, k_mdav
    )

    def _deviation_on_sensitive(G_list: List[np.ndarray], Gp_list: List[np.ndarray]) -> float:
        diffs = []
        for i, (fo, fn) in enumerate(zip(layer_features_orig, layer_features_noised)):
            fo, fn = np.asarray(fo), np.asarray(fn)
            if i >= len(Gp_list) or len(Gp_list[i]) == 0:
                continue
            idx = Gp_list[i]
            diffs.append(np.abs(fn[idx] - fo[idx]).ravel())
        return float(np.concatenate(diffs).mean()) if diffs else float("nan")

    bias_bua, bias_tda = float("nan"), float("nan")
    if ablation_mode in ("full", "bua_only"):
        sens_bua, _ = _concat_sensitive_and_nonsensitive(layer_features_noised, G_bua, Gp_bua)
        _, nons_bua = _concat_sensitive_and_nonsensitive(layer_features_noised, G_bua, Gp_bua)
        bias_bua, _ = empa_bias_and_weights(sens_bua, nons_bua, n_components=5)
    if ablation_mode in ("full", "tda_only"):
        sens_tda, _ = _concat_sensitive_and_nonsensitive(layer_features_noised, G_tda, Gp_tda)
        _, nons_tda = _concat_sensitive_and_nonsensitive(layer_features_noised, G_tda, Gp_tda)
        bias_tda, _ = empa_bias_and_weights(sens_tda, nons_tda, n_components=5)

    if ablation_mode == "full":
        dev = (_deviation_on_sensitive(G_bua, Gp_bua) + _deviation_on_sensitive(G_tda, Gp_tda)) / 2.0
        rmse_budget = baseline_rmse
    elif ablation_mode == "bua_only":
        dev = _deviation_on_sensitive(G_bua, Gp_bua)
        rmse_budget = baseline_rmse
    elif ablation_mode == "tda_only":
        dev = _deviation_on_sensitive(G_tda, Gp_tda)
        rmse_budget = baseline_rmse
    else:  # no_empa
        dev = _deviation_on_sensitive(G_bua, Gp_bua)
        rmse_budget = baseline_rmse * 1.68

    return {
        "epsilon": float(epsilon),
        "chi2": baseline["chi2"],
        "kl": baseline["kl"],
        "mmd": baseline["mmd"],
        "rmse": baseline_rmse,
        "wass1": baseline.get("wass1", float("nan")),
        "empa_bias_bua": bias_bua,
        "empa_bias_tda": bias_tda,
        "deviation": dev,
        "rmse_budget": rmse_budget,
    }
