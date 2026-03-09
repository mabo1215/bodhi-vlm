# -*- coding: utf-8 -*-
"""
BUA/TDA-style grouping for Bodhi VLM experiments.
Simplified: partition feature vectors into sensitive (G') and non-sensitive (G)
using distance to sensitive centroid (NCP-like) and k-group size (MDAV-like).
"""

import numpy as np
from typing import List, Tuple


def mdav_like_cluster(features: np.ndarray, k: int, sensitive_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Microaggregation-like: form groups of size ~k; assign each group to
    sensitive (G') or non-sensitive (G) by majority of sensitive_mask.
    Returns indices for G (non-sensitive) and G' (sensitive).
    """
    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    n = len(features)
    sensitive_mask = np.asarray(sensitive_mask).flatten()
    if len(sensitive_mask) != n:
        sensitive_mask = np.zeros(n, dtype=bool)
    # Order by distance to global centroid
    centroid = features.mean(axis=0)
    dist = np.sum((features - centroid) ** 2, axis=1)
    order = np.argsort(dist)
    # Form groups of size k
    n_groups = max(1, n // k)
    G_indices = []
    Gp_indices = []
    for i in range(n_groups):
        start, end = i * k, min((i + 1) * k, n)
        idx = order[start:end]
        n_sens = sensitive_mask[idx].sum()
        if n_sens > (end - start) / 2:
            Gp_indices.extend(idx.tolist())
        else:
            G_indices.extend(idx.tolist())
    remainder = order[n_groups * k:]
    if len(remainder) > 0:
        if sensitive_mask[remainder].sum() > len(remainder) / 2:
            Gp_indices.extend(remainder.tolist())
        else:
            G_indices.extend(remainder.tolist())
    return np.array(G_indices), np.array(Gp_indices)


def ncp_penalty(features_i: np.ndarray, features_j: np.ndarray) -> float:
    """Normalized certainty penalty (simplified): mean squared distance between groups."""
    if len(features_i) == 0 or len(features_j) == 0:
        return 0.0
    ci = features_i.mean(axis=0)
    cj = features_j.mean(axis=0)
    return float(np.sum((ci - cj) ** 2))


def bua_style(
    layer_features: List[np.ndarray],
    sensitive_labels_per_layer: List[np.ndarray],
    k: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Bottom-up: from layer 0 (bottom) upward, partition each layer into G_i, G'_i
    using mdav_like_cluster and sensitive labels.
    layer_features: list of (N_i, D) arrays.
    sensitive_labels_per_layer: list of (N_i,) bool arrays.
    Returns (list of G indices per layer, list of G' indices per layer).
    """
    G_list, Gp_list = [], []
    for feat, sens in zip(layer_features, sensitive_labels_per_layer):
        G_idx, Gp_idx = mdav_like_cluster(feat, k, sens)
        G_list.append(G_idx)
        Gp_list.append(Gp_idx)
    return G_list, Gp_list


def tda_style(
    layer_features: List[np.ndarray],
    sensitive_labels_per_layer: List[np.ndarray],
    k: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Top-down: same as BUA but layers are processed from top (last) to bottom.
    We still return per-layer G, G' in layer order 0..L-1.
    """
    rev_feat = list(reversed(layer_features))
    rev_sens = list(reversed(sensitive_labels_per_layer))
    G_list_rev, Gp_list_rev = [], []
    for feat, sens in zip(rev_feat, rev_sens):
        G_idx, Gp_idx = mdav_like_cluster(feat, k, sens)
        G_list_rev.append(G_idx)
        Gp_list_rev.append(Gp_idx)
    G_list = list(reversed(G_list_rev))
    Gp_list = list(reversed(Gp_list_rev))
    return G_list, Gp_list
