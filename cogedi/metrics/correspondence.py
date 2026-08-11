from __future__ import annotations

import numpy as np
from typing import Dict


def correspondence_metrics(cloud: np.ndarray, target_point: np.ndarray) -> Dict[str, float]:
    """Compute summary, distance, and likelihood metrics for a point cloud.

    Args:
        cloud: Array of shape [N, D] containing predicted points.
        target_point: Array of shape [D] containing the ground-truth point.
    """
    if cloud.ndim != 2:
        raise ValueError("Cloud must be [N, D]")

    mean = cloud.mean(axis=0)
    median = np.median(cloud, axis=0)
    var = cloud.var(axis=0)

    mean_dist = float(np.linalg.norm(mean - target_point))
    median_dist = float(np.linalg.norm(median - target_point))
    var_mean = float(var.mean())
    var_trace = float(var.sum())

    centered = cloud - mean
    if cloud.shape[0] > 1:
        cov = np.cov(centered, rowvar=False)
    else:
        cov = np.eye(cloud.shape[1], dtype=np.float32)

    eps = 1e-6
    cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps
    inv_cov = np.linalg.pinv(cov)
    diff = (target_point - mean).reshape(-1, 1)
    mahal = float(np.sqrt(np.maximum(diff.T @ inv_cov @ diff, 0.0)).squeeze())

    dim = int(cloud.shape[1])
    sign, logdet_cov = np.linalg.slogdet(cov)
    if sign <= 0:
        logdet_cov = float(np.log(np.linalg.det(cov + np.eye(dim, dtype=cov.dtype) * eps)))
    quad = float((diff.T @ inv_cov @ diff).squeeze())
    gaussian_loglik = float(-0.5 * (dim * np.log(2.0 * np.pi) + logdet_cov + quad))

    # Isotropic Gaussian KDE log-likelihood with Scott-like bandwidth.
    n = max(int(cloud.shape[0]), 1)
    sigma_scalar = float(np.sqrt(max(var_mean, eps)))
    bw = max(sigma_scalar * (n ** (-1.0 / (dim + 4.0))), 1e-6)
    deltas = cloud - target_point.reshape(1, -1)
    sq_dists = np.sum(deltas * deltas, axis=1)
    log_kernel = -0.5 * sq_dists / (bw * bw)
    log_kernel_max = float(np.max(log_kernel))
    log_mean_exp = log_kernel_max + float(np.log(np.mean(np.exp(log_kernel - log_kernel_max))))
    kde_norm = float(-0.5 * dim * np.log(2.0 * np.pi * bw * bw))
    kde_loglik = float(kde_norm + log_mean_exp)

    return {
        "mean_dist": mean_dist,
        "median_dist": median_dist,
        "var_mean": var_mean,
        "var_trace": var_trace,
        "mahalanobis": mahal,
        "gaussian_loglik": gaussian_loglik,
        "kde_loglik": kde_loglik,
    }
