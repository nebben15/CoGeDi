from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GeodesicDescriptorLookup:
	"""
	Cached lookup context for geodesic descriptors.

	- vertices: [V, 3]
	- faces: [F, 3] (vertex IDs)
	- vertex_to_landmark: [V, L] where L=#landmarks
	"""

	vertices: np.ndarray
	faces: np.ndarray
	vertex_to_landmark: np.ndarray
	tree: cKDTree


def build_geodesic_descriptor_lookup(
	*,
	vertices: torch.Tensor | np.ndarray,
	faces: torch.Tensor | np.ndarray,
	vertex_to_landmark: torch.Tensor | np.ndarray,
) -> GeodesicDescriptorLookup:
	"""Build cached descriptor lookup resources (including KD-tree)."""
	verts = _as_numpy(vertices, dtype=np.float64)
	tri = _as_numpy(faces, dtype=np.int64)
	v2l = _as_numpy(vertex_to_landmark, dtype=np.float32)

	if verts.ndim != 2 or verts.shape[1] != 3:
		raise ValueError("vertices must have shape [V, 3]")
	if tri.ndim != 2 or tri.shape[1] != 3:
		raise ValueError("faces must have shape [F, 3]")
	if v2l.ndim != 2:
		raise ValueError("vertex_to_landmark must have shape [V, L]")
	if v2l.shape[0] != verts.shape[0]:
		raise ValueError("vertex_to_landmark first dimension must equal number of vertices")
	if np.any(tri < 0) or np.any(tri >= verts.shape[0]):
		raise ValueError("faces contain out-of-range vertex IDs")

	tree = cKDTree(verts)
	return GeodesicDescriptorLookup(
		vertices=verts,
		faces=tri,
		vertex_to_landmark=v2l,
		tree=tree,
	)


def geodesic_descriptor_from_triangle_barycentric(
	lookup: GeodesicDescriptorLookup,
	*,
	triangle_ids: torch.Tensor | np.ndarray,
	barycentric: torch.Tensor | np.ndarray,
	output_device: torch.device | None = None,
	output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	"""
	Descriptor interpolation from (triangle_id, barycentric) input.

	Inputs support shape [N] or [B, N] for triangle_ids and [..., 3] for barycentric.
	Output shape is [..., L].
	"""
	tri_ids = _as_numpy(triangle_ids, dtype=np.int64)
	bary = _as_numpy(barycentric, dtype=np.float32)

	if bary.shape[-1] != 3:
		raise ValueError("barycentric must have last dimension size 3")
	if tri_ids.shape != bary.shape[:-1]:
		raise ValueError("triangle_ids shape must match barycentric without the last coordinate dim")
	if np.any(tri_ids < 0) or np.any(tri_ids >= lookup.faces.shape[0]):
		raise IndexError("triangle_ids contain out-of-range face indices")

	flat_tri = tri_ids.reshape(-1)
	flat_bary = bary.reshape(-1, 3)

	bary_sum = np.sum(flat_bary, axis=1, keepdims=True)
	nonzero = np.abs(bary_sum[:, 0]) > 1e-12
	flat_bary_norm = flat_bary.copy()
	flat_bary_norm[nonzero] = flat_bary_norm[nonzero] / bary_sum[nonzero]

	tri_vids = lookup.faces[flat_tri]  # [M, 3]
	desc_tri = lookup.vertex_to_landmark[tri_vids]  # [M, 3, L]
	desc = np.einsum("mk,mkl->ml", flat_bary_norm, desc_tri, optimize=True)

	out = desc.reshape(*tri_ids.shape, lookup.vertex_to_landmark.shape[1])
	return _to_torch(out, device=output_device, dtype=output_dtype)


def geodesic_descriptor_from_point_nearest_vertex(
	lookup: GeodesicDescriptorLookup,
	*,
	points: torch.Tensor | np.ndarray,
	output_device: torch.device | None = None,
	output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	"""
	Descriptor lookup from nearest neighbor vertex for each point.

	points: [N, 3] or [B, N, 3]
	output: [N, L] or [B, N, L]
	"""
	pts, prefix_shape = _flatten_points(points)
	_, nn_idx = lookup.tree.query(pts, k=1)
	desc = lookup.vertex_to_landmark[np.asarray(nn_idx, dtype=np.int64)]
	out = desc.reshape(*prefix_shape, lookup.vertex_to_landmark.shape[1])
	return _to_torch(out, device=output_device, dtype=output_dtype)


def geodesic_descriptor_from_point_knn_weighted(
	lookup: GeodesicDescriptorLookup,
	*,
	points: torch.Tensor | np.ndarray,
	n_neighbors: int,
	eps: float = 1e-8,
	output_device: torch.device | None = None,
	output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	"""
	Descriptor lookup from k-NN vertices with normalized inverse-distance weights.

	points: [N, 3] or [B, N, 3]
	output: [N, L] or [B, N, L]
	"""
	if n_neighbors <= 0:
		raise ValueError("n_neighbors must be > 0")

	pts, prefix_shape = _flatten_points(points)
	k = min(int(n_neighbors), lookup.vertices.shape[0])
	dists, idx = lookup.tree.query(pts, k=k)

	if k == 1:
		dists = np.asarray(dists, dtype=np.float32)[:, None]
		idx = np.asarray(idx, dtype=np.int64)[:, None]
	else:
		dists = np.asarray(dists, dtype=np.float32)
		idx = np.asarray(idx, dtype=np.int64)

	desc_knn = lookup.vertex_to_landmark[idx]  # [M, k, L]

	zero_mask = dists <= eps
	any_zero = np.any(zero_mask, axis=1, keepdims=True)

	inv = 1.0 / np.clip(dists, eps, None)
	inv_sum = np.sum(inv, axis=1, keepdims=True)
	w_inv = inv / np.clip(inv_sum, eps, None)

	zero_count = np.sum(zero_mask, axis=1, keepdims=True)
	w_zero = np.where(zero_mask, 1.0 / np.clip(zero_count, 1, None), 0.0)

	weights = np.where(any_zero, w_zero, w_inv).astype(np.float32)
	desc = np.einsum("mk,mkl->ml", weights, desc_knn, optimize=True)

	out = desc.reshape(*prefix_shape, lookup.vertex_to_landmark.shape[1])
	return _to_torch(out, device=output_device, dtype=output_dtype)


def _flatten_points(points: torch.Tensor | np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
	pts = _as_numpy(points, dtype=np.float64)
	if pts.ndim < 2 or pts.shape[-1] != 3:
		raise ValueError("points must have shape [N, 3] or [B, N, 3]")
	prefix = tuple(pts.shape[:-1])
	flat = pts.reshape(-1, 3)
	return flat, prefix


def _as_numpy(x: torch.Tensor | np.ndarray, *, dtype: np.dtype) -> np.ndarray:
	if isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy().astype(dtype, copy=False)
	return np.asarray(x, dtype=dtype)


def _to_torch(x: np.ndarray, *, device: torch.device | None, dtype: torch.dtype) -> torch.Tensor:
	t = torch.from_numpy(np.asarray(x))
	if device is not None:
		t = t.to(device=device)
	return t.to(dtype=dtype)

