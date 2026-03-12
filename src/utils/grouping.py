# -*- coding: utf-8 -*-
"""
BUA/TDA-style grouping for Bodhi VLM experiments.
Partition feature vectors into sensitive (G') and non-sensitive (G)
using MDAV-like clustering and sensitive labels.
"""
import numpy as np
from typing import List, Optional, Tuple


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
    centroid = features.mean(axis=0)
    dist = np.sum((features - centroid) ** 2, axis=1)
    order = np.argsort(dist)
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
    """Bottom-up: partition each layer into G_i, G'_i using mdav_like_cluster."""
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
    """Top-down: same as BUA but layers processed from top to bottom."""
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


def map_indices_layer_i_to_im1(
    n_i: int,
    n_im1: int,
    shape_i: Tuple[int, int],
    shape_im1: Tuple[int, int],
) -> np.ndarray:
    """
    Correspondence Pi_{i -> i-1}: for each linear index in layer i (row-major),
    return the linear index in layer i-1 (nearest spatial correspondence).
    shape = (H, W). Used for TDA cross-layer links (YOLO/DETR/CLIP/BLIP: different H,W per layer).
    """
    Hi, Wi = int(shape_i[0]), int(shape_i[1])
    Him1, Wim1 = int(shape_im1[0]), int(shape_im1[1])
    if Hi * Wi == 0 or Him1 * Wim1 == 0:
        return np.zeros(n_i, dtype=np.int64)  # fallback
    idx_im1 = np.zeros(n_i, dtype=np.int64)
    for idx in range(min(n_i, Hi * Wi)):
        h, w = divmod(idx, Wi)
        h_n = min(int(round(h * Him1 / max(Hi, 1))), Him1 - 1)
        w_n = min(int(round(w * Wim1 / max(Wi, 1))), Wim1 - 1)
        idx_im1[idx] = min(h_n * Wim1 + w_n, n_im1 - 1)
    return idx_im1


def tda_style_with_mapping(
    layer_features: List[np.ndarray],
    sensitive_labels_per_layer: List[np.ndarray],
    k: int = 3,
    layer_shapes: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    TDA with explicit cross-layer mapping Pi. If layer_shapes is None, same as tda_style.
    layer_shapes[i] = (H_i, W_i) for layer i (spatial size before flatten). Used for YOLO/DETR/CLIP/BLIP.
    """
    if layer_shapes is None or len(layer_shapes) != len(layer_features):
        return tda_style(layer_features, sensitive_labels_per_layer, k=k)
    rev_feat = list(reversed(layer_features))
    rev_sens = list(reversed(sensitive_labels_per_layer))
    rev_shapes = list(reversed(layer_shapes))
    G_list_rev, Gp_list_rev = [], []
    for feat, sens in zip(rev_feat, rev_sens):
        G_idx, Gp_idx = mdav_like_cluster(feat, k, sens)
        G_list_rev.append(G_idx)
        Gp_list_rev.append(Gp_idx)
    G_list = list(reversed(G_list_rev))
    Gp_list = list(reversed(Gp_list_rev))
    # Optionally refine using Pi: for each layer i > 0, L_i^(s) = Pi(Gp_i) cap Gp_{i-1}. We keep current partition; Pi is available for analysis.
    return G_list, Gp_list
