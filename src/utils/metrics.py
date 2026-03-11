# -*- coding: utf-8 -*-
"""
Bodhi VLM experiment metrics: Chi-square, K-L divergence, MMD, rMSE, EMPA-style bias.
Used for privacy budget assessment comparison.
"""
import numpy as np
from scipy import stats
from typing import Tuple, Optional


def chi_square_stat(p_obs: np.ndarray, p_exp: np.ndarray, bins: int = 10) -> float:
    """Chi-square statistic between observed and expected histograms (same bins)."""
    p_obs = np.asarray(p_obs).flatten()
    p_exp = np.asarray(p_exp).flatten()
    p_exp = np.maximum(p_exp, 1e-10)
    return np.sum((p_obs - p_exp) ** 2 / p_exp)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """K-L divergence sum_i p_i log(p_i/q_i)."""
    p = np.asarray(p).flatten() + eps
    q = np.asarray(q).flatten() + eps
    p, q = p / p.sum(), q / q.sum()
    return np.sum(p * (np.log(p) - np.log(q)))


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> float:
    """MMD with RBF kernel (squared distance)."""
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if gamma is None:
        gamma = 1.0 / (x.shape[1] + 1e-8)
    n, m = len(x), len(y)
    kxx = np.exp(-gamma * ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    kyy = np.exp(-gamma * ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    kxy = np.exp(-gamma * ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean square error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def histogram_from_samples(samples: np.ndarray, bins: int = 20, range_lim: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Normalized histogram (probability) from 1D or flattened samples."""
    s = np.asarray(samples).flatten()
    if range_lim is None:
        range_lim = (s.min(), s.max())
    if range_lim[1] <= range_lim[0]:
        range_lim = (s.min(), s.max() + 1e-8)
    hist, _ = np.histogram(s, bins=bins, range=range_lim, density=False)
    return hist.astype(float) / (hist.sum() + 1e-10)


def empa_bias_and_weights(
    sensitive_features: np.ndarray,
    non_sensitive_features: np.ndarray,
    n_components: int = 5,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Tuple[float, np.ndarray]:
    """EMPA-style mixture: return bias (L2 distance to uniform weights) and mixture weights."""
    from numpy import log, exp
    all_f = np.vstack([
        np.asarray(sensitive_features).reshape(-1, 1) if np.asarray(sensitive_features).ndim == 1 else sensitive_features,
        np.asarray(non_sensitive_features).reshape(-1, 1) if np.asarray(non_sensitive_features).ndim == 1 else non_sensitive_features,
    ])
    if all_f.shape[0] < n_components:
        n_components = max(1, all_f.shape[0] // 2)
    data = np.asarray(all_f)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, d = data.shape
    rng = np.random.default_rng(42)
    means = data[rng.choice(n, size=n_components, replace=False)]
    var = np.ones(n_components) * (data.var() + 1e-8)
    weights = np.ones(n_components) / n_components
    for _ in range(max_iter):
        log_prob = np.zeros((n, n_components))
        for k in range(n_components):
            log_prob[:, k] = np.log(weights[k] + 1e-10) - 0.5 * np.sum((data - means[k]) ** 2, axis=1) / (var[k] + 1e-10)
        log_prob -= log_prob.max(axis=1, keepdims=True)
        resp = exp(log_prob)
        resp /= resp.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0)
        weights = nk / n
        for k in range(n_components):
            means[k] = (resp[:, k:k+1] * data).sum(axis=0) / (nk[k] + 1e-10)
            var[k] = (resp[:, k] * ((data - means[k]) ** 2).sum(axis=1)).sum() / (nk[k] * d + 1e-10)
    uniform = np.ones(n_components) / n_components
    bias = float(np.sqrt(np.sum((weights - uniform) ** 2)))
    return bias, weights


def compare_metrics(
    original: np.ndarray,
    noised: np.ndarray,
    bins: int = 20,
    max_samples: int = 5000,
) -> dict:
    """
    Compare original vs noised samples using Chi-square, K-L, MMD, rMSE, wass1.
    If sample count exceeds max_samples, subsample to avoid O(N^2) MMD.
    """
    orig = np.asarray(original).reshape(-1, 1) if np.asarray(original).ndim == 1 else original
    nois = np.asarray(noised).reshape(-1, 1) if np.asarray(noised).ndim == 1 else noised
    n = len(orig)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        orig = orig[idx]
        nois = nois[idx]
    range_lim = (min(orig.min(), nois.min()), max(orig.max(), nois.max()))
    if range_lim[1] - range_lim[0] < 1e-10:
        range_lim = (range_lim[0], range_lim[0] + 1.0)
    h_orig = histogram_from_samples(orig, bins=bins, range_lim=range_lim)
    h_nois = histogram_from_samples(nois, bins=bins, range_lim=range_lim)
    chi2 = chi_square_stat(h_nois, h_orig)
    kl = kl_divergence(h_nois, h_orig)
    mmd = mmd_rbf(orig, nois)
    rms = rmse(orig, nois)
    orig_flat = np.asarray(original).flatten()
    nois_flat = np.asarray(noised).flatten()
    wass1 = float(stats.wasserstein_distance(orig_flat, nois_flat))
    return {"chi2": chi2, "kl": kl, "mmd": mmd, "rmse": rms, "wass1": wass1}
