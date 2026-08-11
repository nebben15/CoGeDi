from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import importlib
import os

import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from tqdm import tqdm


def euclidean_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    return torch.norm(pred - target, dim=-1).mean(dim=-1)


@dataclass
class MeshGeodesicDistance:
    mesh_path: object
    cache_all_pairs: bool = True
    max_all_pairs: int = 8000
    chunk_size: int = 4096

    def __post_init__(self) -> None:
        mesh = self.mesh_path
        if isinstance(mesh, (str, os.PathLike)):
            mesh = trimesh.load(mesh, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("mesh_path must load a Trimesh or Scene")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        edges = np.asarray(mesh.edges_unique, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh vertices must be Nx3")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("mesh edges must be Ex2")

        weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        graph = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)))
        graph = graph + graph.T

        self._vertices = vertices
        self._graph = graph.tocsr()
        self._tree = cKDTree(vertices)
        self._all_pairs: Optional[np.ndarray] = None

        if self.cache_all_pairs and len(vertices) <= self.max_all_pairs:
            total = len(vertices)
            all_pairs = np.empty((total, total), dtype=np.float32)
            iter_starts = range(0, total, self.chunk_size)
            for start in tqdm(iter_starts, desc="Precomputing geodesics", ncols=100):
                end = min(start + self.chunk_size, total)
                src = np.arange(start, end)
                dist_matrix = dijkstra(self._graph, directed=False, indices=src).astype(np.float32)
                all_pairs[start:end] = dist_matrix
            self._all_pairs = all_pairs

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")
        if pred.ndim != 3:
            raise ValueError("pred/target must be shaped [B, N, D]")
        if pred.shape[-1] != 3:
            raise ValueError("geodesic distance requires 3D points")

        device = pred.device
        dtype = pred.dtype

        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        target_np = target.detach().cpu().numpy().astype(np.float32)

        batch, points, _ = pred_np.shape
        pred_flat = pred_np.reshape(-1, 3)
        target_flat = target_np.reshape(-1, 3)

        _, pred_idx = self._tree.query(pred_flat)
        _, target_idx = self._tree.query(target_flat)

        dist_flat = np.empty(pred_idx.shape[0], dtype=np.float32)
        if self._all_pairs is not None:
            dist_flat = self._all_pairs[pred_idx, target_idx]
        else:
            for start in range(0, len(pred_idx), self.chunk_size):
                end = min(start + self.chunk_size, len(pred_idx))
                src = pred_idx[start:end]
                dist_matrix = dijkstra(self._graph, directed=False, indices=src)
                dist_flat[start:end] = dist_matrix[np.arange(len(src)), target_idx[start:end]]

        dist = dist_flat.reshape(batch, points).mean(axis=1)
        return torch.from_numpy(dist).to(device=device, dtype=dtype)


def build_geodesic_distance(mesh, *, cache_all_pairs: bool, max_all_pairs: int, chunk_size: int) -> MeshGeodesicDistance:
    return MeshGeodesicDistance(
        mesh_path=mesh,
        cache_all_pairs=cache_all_pairs,
        max_all_pairs=max_all_pairs,
        chunk_size=chunk_size,
    )


def geodesic_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    mesh,
    *,
    cache_all_pairs: bool = True,
    max_all_pairs: int = 8000,
    chunk_size: int = 4096,
) -> torch.Tensor:
    if isinstance(mesh, MeshGeodesicDistance):
        distance_fn = mesh
    else:
        distance_fn = build_geodesic_distance(
            mesh,
            cache_all_pairs=cache_all_pairs,
            max_all_pairs=max_all_pairs,
            chunk_size=chunk_size,
        )
    return distance_fn(pred, target)


def build_distance_fn(params) -> callable:
    distance_type = str(getattr(params, "distance_type", "geodesic")).lower()
    if distance_type == "euclidean":
        return euclidean_distance
    if distance_type == "geodesic":
        return geodesic_distance
    raise ValueError("distance_type must be 'geodesic' or 'euclidean'")


def compute_vertex_to_landmark_geodesic_matrix(
    vertices: np.ndarray,
    faces: np.ndarray,
    landmark_vertex_ids: Sequence[int],
    *,
    backend: str = "potpourri3d",
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> np.ndarray:
    """
    Compute geodesic distances from every vertex to every landmark vertex.

    Returns matrix with shape [num_vertices, num_landmarks].
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    lm = np.asarray(list(landmark_vertex_ids), dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if lm.ndim != 1:
        raise ValueError("landmark_vertex_ids must be 1D")
    if lm.size == 0:
        return np.empty((v.shape[0], 0), dtype=np.float32)

    if np.any(lm < 0) or np.any(lm >= v.shape[0]):
        raise ValueError("landmark_vertex_ids contain out-of-range indices")

    backend_name = str(backend).lower()
    if backend_name == "potpourri3d":
        return _compute_vertex_landmark_geodesics_potpourri3d(
            v,
            f,
            lm,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    if backend_name == "pygeodesic":
        return _compute_vertex_landmark_geodesics_pygeodesic(
            v,
            f,
            lm,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    raise ValueError("backend must be 'potpourri3d' or 'pygeodesic'")


def _compute_vertex_landmark_geodesics_potpourri3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    landmark_vertex_ids: np.ndarray,
    *,
    show_progress: bool,
    progress_desc: str | None,
) -> np.ndarray:
    try:
        pp3d = importlib.import_module("potpourri3d")
    except Exception as exc:
        raise ImportError(
            "potpourri3d backend requested, but potpourri3d is not installed"
        ) from exc

    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    out = np.empty((vertices.shape[0], landmark_vertex_ids.shape[0]), dtype=np.float32)
    it = enumerate(landmark_vertex_ids)
    if show_progress:
        it = enumerate(
            tqdm(
                landmark_vertex_ids,
                desc=progress_desc or "Geodesic (heat)",
                ncols=100,
            )
        )
    for j, source_vid in it:
        dist = solver.compute_distance(int(source_vid))
        out[:, j] = np.asarray(dist, dtype=np.float32)
    return out


def _compute_vertex_landmark_geodesics_pygeodesic(
    vertices: np.ndarray,
    faces: np.ndarray,
    landmark_vertex_ids: np.ndarray,
    *,
    show_progress: bool,
    progress_desc: str | None,
) -> np.ndarray:
    try:
        geodesic = importlib.import_module("pygeodesic.geodesic")
    except Exception as exc:
        raise ImportError(
            "pygeodesic backend requested, but pygeodesic is not installed"
        ) from exc

    if not hasattr(geodesic, "PyGeodesicAlgorithmExact"):
        raise RuntimeError("pygeodesic does not expose PyGeodesicAlgorithmExact")

    solver = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
    n_vertices = vertices.shape[0]
    targets = np.arange(n_vertices, dtype=np.int32)
    out = np.empty((n_vertices, landmark_vertex_ids.shape[0]), dtype=np.float32)

    it = enumerate(landmark_vertex_ids)
    if show_progress:
        it = enumerate(
            tqdm(
                landmark_vertex_ids,
                desc=progress_desc or "Geodesic (direct)",
                ncols=100,
            )
        )
    for j, source_vid in it:
        dvec = _pygeodesic_all_distances_from_source(solver, int(source_vid), targets)
        out[:, j] = dvec
    return out


def _pygeodesic_all_distances_from_source(
    solver,
    source_vid: int,
    targets: np.ndarray,
) -> np.ndarray:
    # Try likely pygeodesic interfaces first.
    if hasattr(solver, "geodesicDistances"):
        attempts = [
            lambda: solver.geodesicDistances(np.array([source_vid], dtype=np.int32), targets),
            lambda: solver.geodesicDistances([source_vid], targets),
            lambda: solver.geodesicDistances(source_vid, targets),
        ]
        for fn in attempts:
            try:
                res = fn()
                dist = res[0] if isinstance(res, tuple) else res
                dist = np.asarray(dist, dtype=np.float32).reshape(-1)
                if dist.shape[0] == targets.shape[0]:
                    return dist
            except Exception:
                continue

    # Fallback: query each target individually via geodesicDistance.
    if hasattr(solver, "geodesicDistance"):
        dist = np.empty((targets.shape[0],), dtype=np.float32)
        for i, t in enumerate(targets):
            res = solver.geodesicDistance(int(source_vid), int(t))
            d = res[0] if isinstance(res, tuple) else res
            dist[i] = float(d)
        return dist

    raise RuntimeError(
        "Unable to query distances from pygeodesic solver; unsupported API variant"
    )
