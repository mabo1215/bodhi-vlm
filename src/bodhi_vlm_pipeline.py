# -*- coding: utf-8 -*-
"""
High-level Bodhi VLM pipeline utilities.

This module connects the core experimental components
(`grouping.py` for BUA/TDA; `metrics.py` for EMPA-style
assessment) with generic model features. It is designed
to be *model-agnostic*: any vision backbone or VLM that
exposes layer-wise feature tensors can be plugged in.

Typical usage pattern:

1) Extract layer-wise features from a detector or VLM:
   - For a detector (e.g., YOLO), hook into the backbone
     and feature pyramid to obtain a list of feature maps
     per layer (shape (N_i, D)).
   - For a VLM (e.g., CLIP, LLaVA), hook into the vision
     encoder (ViT / ResNet blocks) to obtain per-layer
     patch/spatial features.
   - Construct a boolean mask per layer indicating which
     samples are sensitive with respect to a given
     sensitive label set S_epsilon.

2) Call `assess_privacy_budget_from_features` on the
   original and privacy-noised features to obtain:
   - Chi-square, K-L, MMD, rMSE between original and
     noised features;
   - EMPA-style bias for BUA and TDA groups.

The same code path applies to:
   - Detectors such as YOLOv4 / PPDPTS / DETR;
   - VLM vision encoders such as CLIP, LLaVA, BLIP.

This file intentionally does not depend on any specific
detector or VLM library; examples are given in docstrings.
"""

from typing import List, Dict, Any, Tuple

import numpy as np

from grouping import bua_style, tda_style
from metrics import compare_metrics, empa_bias_and_weights


def group_features_bua_tda(
    layer_features_noised: List[np.ndarray],
    sensitive_per_layer: List[np.ndarray],
    k_mdav: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run BUA- and TDA-style grouping on a stack of layer-wise features.

    Parameters
    ----------
    layer_features_noised:
        List of feature tensors per layer after privacy noise has been
        applied. Each entry is an array of shape (N_i, D).
    sensitive_per_layer:
        List of boolean masks per layer (length N_i) indicating which
        samples or spatial locations are considered sensitive with
        respect to the sensitive label set S_epsilon.
    k_mdav:
        Group size parameter for the MDAV-like microaggregation
        (see `grouping.mdav_like_cluster`).

    Returns
    -------
    G_bua, Gp_bua, G_tda, Gp_tda:
        For each layer, the indices of non-sensitive (G_i) and
        sensitive (G'_i) groups under BUA and under TDA.
    """
    G_bua, Gp_bua = bua_style(layer_features_noised, sensitive_per_layer, k=k_mdav)
    G_tda, Gp_tda = tda_style(layer_features_noised, sensitive_per_layer, k=k_mdav)
    return G_bua, Gp_bua, G_tda, Gp_tda


def _concat_sensitive_and_nonsensitive(
    layer_features_noised: List[np.ndarray],
    G_list: List[np.ndarray],
    Gp_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper: collect all sensitive / non-sensitive feature vectors
    across layers into two matrices.
    """
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
) -> Dict[str, Any]:
    """
    End-to-end Bodhi VLM assessment on arbitrary backbone features.

    Parameters
    ----------
    layer_features_orig:
        List of original (pre-noise) feature tensors per layer,
        each of shape (N_i, D).
    layer_features_noised:
        List of privacy-noised feature tensors per layer, same
        shapes as `layer_features_orig`.
    sensitive_per_layer:
        List of boolean masks per layer, indicating which rows
        belong to the sensitive set S_epsilon.
    epsilon:
        Privacy budget used when adding noise (used here for
        bookkeeping; the assessment itself is feature-based).
    bins:
        Histogram bins for Chi-square / K-L in `compare_metrics`.
    k_mdav:
        Group size parameter passed to BUA/TDA grouping.

    Returns
    -------
    metrics:
        Dictionary with the following keys:
            - 'epsilon': privacy budget
            - 'chi2', 'kl', 'mmd', 'rmse' (baseline metrics)
            - 'empa_bias_bua', 'empa_bias_tda' (EMPA bias
              for BUA and TDA groupings)

    Notes
    -----
    This function is the core implementation of the protocol
    described in Section 4.1 and Section 4.2 of the paper:

      1) Flatten all layer-wise features to compare original
         vs. noised distributions using Chi-square, K-L,
         MMD, and rMSE.
      2) Apply BUA and TDA to the noised features to obtain
         layer-wise groups G_i and G'_i.
      3) Feed the concatenated sensitive / non-sensitive
         features into EMPA (via `empa_bias_and_weights`)
         to obtain a scalar bias value for each strategy.

    Example (pseudocode for integration with a detector):

        # Suppose `yolo_backbone_layers(image)` returns a
        # list of feature maps per layer, and `mask_sens`
        # is computed from labels S_epsilon.
        feats_orig = yolo_backbone_layers(image_orig)
        feats_noised = yolo_backbone_layers(image_noised)
        sensitive_masks = [mask_sens(layer) for layer in feats_orig]
        result = assess_privacy_budget_from_features(
            feats_orig, feats_noised, sensitive_masks, epsilon=0.1
        )

    For VLMs such as CLIP or LLaVA, the same pattern applies
    once layer-wise visual features are exposed by the model.
    """
    if len(layer_features_orig) != len(layer_features_noised):
        raise ValueError("Original and noised feature lists must have the same length.")

    # Baseline metrics on flattened features (use all layers).
    flat_orig = np.concatenate([np.asarray(l).flatten() for l in layer_features_orig])
    flat_nois = np.concatenate([np.asarray(l).flatten() for l in layer_features_noised])
    baseline = compare_metrics(flat_orig, flat_nois, bins=bins)

    # BUA / TDA grouping and EMPA-style bias.
    G_bua, Gp_bua, G_tda, Gp_tda = group_features_bua_tda(
        layer_features_noised,
        sensitive_per_layer,
        k_mdav,
    )
    sens_bua, nons_bua = _concat_sensitive_and_nonsensitive(
        layer_features_noised, G_bua, Gp_bua
    )
    sens_tda, nons_tda = _concat_sensitive_and_nonsensitive(
        layer_features_noised, G_tda, Gp_tda
    )
    bias_bua, _ = empa_bias_and_weights(sens_bua, nons_bua, n_components=5)
    bias_tda, _ = empa_bias_and_weights(sens_tda, nons_tda, n_components=5)

    return {
        "epsilon": float(epsilon),
        "chi2": baseline["chi2"],
        "kl": baseline["kl"],
        "mmd": baseline["mmd"],
        "rmse": baseline["rmse"],
        "empa_bias_bua": bias_bua,
        "empa_bias_tda": bias_tda,
    }


