from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def _as_geometry_points(points: np.ndarray) -> np.ndarray:
    """Project any point representation onto the first three spatial coordinates."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 spatial dimensions, got shape {arr.shape}")
    return arr[..., :3].reshape(-1, 3)


def chamfer_distance_l2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute symmetric Chamfer distance using *L2* (non-squared) distances.

    This metric measures the average closest-point distance in both directions:

    1) For each point in A, find the nearest neighbor in B and average those
       Euclidean distances.
    2) For each point in B, find the nearest neighbor in A and average those
       Euclidean distances.

    The Chamfer distance is the sum of these two means. It penalizes both
    missing coverage (A far from B) and extra mass (B far from A) and is
    widely used for point-cloud similarity.
    """
    a_xyz = _as_geometry_points(a)
    b_xyz = _as_geometry_points(b)
    tree_a = cKDTree(a_xyz)
    tree_b = cKDTree(b_xyz)
    d_a_to_b, _ = tree_b.query(a_xyz)
    d_b_to_a, _ = tree_a.query(b_xyz)
    return float(np.mean(d_a_to_b) + np.mean(d_b_to_a))


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the symmetric Hausdorff distance between two point sets.

    The Hausdorff distance is the maximum of the directed nearest-neighbor
    distances. It captures the worst-case deviation between sets:

    H(A,B) = max( max_a min_b ||a-b||, max_b min_a ||b-a|| )

    This metric is sensitive to outliers but provides a strict notion of
    shape mismatch.
    """
    a_xyz = _as_geometry_points(a)
    b_xyz = _as_geometry_points(b)
    tree_a = cKDTree(a_xyz)
    tree_b = cKDTree(b_xyz)
    d_a_to_b, _ = tree_b.query(a_xyz)
    d_b_to_a, _ = tree_a.query(b_xyz)
    return float(max(np.max(d_a_to_b), np.max(d_b_to_a)))


def point_to_surface_distance(points: np.ndarray, mesh: trimesh.Trimesh) -> float:
    """
    Compute mean point-to-surface distance from points to a mesh surface.

    This measures how far each predicted point is from the *continuous*
    target surface (not just reference vertices). It uses the mesh's
    closest-point query and returns the average Euclidean distance.

    If mesh proximity queries are unavailable, this metric should be
    approximated by point-to-point distances against dense reference samples.
    """
    points_xyz = _as_geometry_points(points)
    pq = trimesh.proximity.ProximityQuery(mesh)
    dist = pq.distance(points_xyz)
    return float(np.mean(dist))


def earth_movers_distance(a: np.ndarray, b: np.ndarray, max_points: int) -> float:
    """
    Approximate Earth Mover's Distance (Wasserstein-1) between two point sets.

    For computational tractability, both point sets are downsampled to the
    same size N = min(len(A), len(B), max_points). The cost matrix contains
    pairwise L2 distances, and the optimal one-to-one transport is solved via
    the Hungarian algorithm. The returned EMD is the mean transport cost.

    This approximation is suitable for moderate N (e.g., 1024–4096). Larger
    values can be expensive in memory and time.
    """
    a_xyz = _as_geometry_points(a)
    b_xyz = _as_geometry_points(b)
    rng = np.random.default_rng(42)
    n = min(len(a_xyz), len(b_xyz), max_points)
    if n <= 0:
        return float("nan")
    idx_a = rng.choice(len(a_xyz), size=n, replace=False)
    idx_b = rng.choice(len(b_xyz), size=n, replace=False)
    a_sub = a_xyz[idx_a]
    b_sub = b_xyz[idx_b]
    cost = cdist(a_sub, b_sub, metric="euclidean")
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def f_score(a: np.ndarray, b: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """
    Compute F-score for point-set matching at a distance threshold.

    Precision: fraction of points in A within 'threshold' of set B.
    Recall:    fraction of points in B within 'threshold' of set A.
    F-score:   harmonic mean of precision and recall.

    This metric captures how well two point clouds overlap at a given
    tolerance and is commonly used for geometric reconstruction evaluation.
    """
    a_xyz = _as_geometry_points(a)
    b_xyz = _as_geometry_points(b)
    tree_a = cKDTree(a_xyz)
    tree_b = cKDTree(b_xyz)
    d_a_to_b, _ = tree_b.query(a_xyz)
    d_b_to_a, _ = tree_a.query(b_xyz)
    precision = float(np.mean(d_a_to_b <= threshold)) if len(d_a_to_b) > 0 else 0.0
    recall = float(np.mean(d_b_to_a <= threshold)) if len(d_b_to_a) > 0 else 0.0
    denom = precision + recall
    f = 0.0 if denom == 0.0 else float(2.0 * precision * recall / denom)
    return f, precision, recall
