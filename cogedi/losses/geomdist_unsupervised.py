from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch

from cogedi.data.base import build_data_source
from cogedi.dtypes import ObservedMask, Sigma, State
from cogedi.losses.base import BaseLoss, LossOutput
from cogedi.utils.descriptors import (
	build_geodesic_descriptor_lookup,
)


class GeomDistUnsupervisedLoss(BaseLoss):
	"""
	diffusion_loss + geodesic descriptor consistency + surface penalty

	total = diffusion + lambda_geo * geo + lambda_surface * surface
	"""

	name = "geomdist_unsupervised"

	def __init__(self, cfg=None, **kwargs):
		params = getattr(cfg, "params", cfg)
		full_cfg = kwargs.get("full_cfg", None)

		self.sigma_data = float(getattr(params, "sigma_data", 1.0))
		self.eps = float(getattr(params, "eps", 1e-12))
		self.lambda_geo = float(getattr(params, "lambda_geo", 0.3))
		self.lambda_surface = float(getattr(params, "lambda_surface", 0.05))
		self.knn_neighbors = int(getattr(params, "knn_neighbors", 4))
		self.surface_chunk_size = int(getattr(params, "surface_chunk_size", 4096))
		self.use_cdist = bool(getattr(params, "use_cdist", False))

		self.descriptor_interpolation = str(
			getattr(
				getattr(getattr(full_cfg, "data", None), "landmark_supervision", None),
				"descriptor_interpolation",
				"barycentric",
			)
		).lower()
		if self.descriptor_interpolation == "knn":
			self.descriptor_interpolation = "knn"
		elif self.descriptor_interpolation == "knn_weighted":
			self.descriptor_interpolation = "knn"
		elif self.descriptor_interpolation == "nearest":
			self.descriptor_interpolation = "nearest_neighbor"

		self.surface_point_mode = str(
			getattr(
				getattr(getattr(full_cfg, "data", None), "landmark_supervision", None),
				"surface_point",
				"projection",
			)
		).strip().lower()
		if self.surface_point_mode not in {"projection", "pull"}:
			raise ValueError("surface_point must be one of: projection, pull")
		if self.surface_point_mode == "pull" and self.descriptor_interpolation == "barycentric":
			raise ValueError(
				"descriptor_interpolation='barycentric' is not supported with surface_point='pull'. "
				"Use nearest_neighbor or knn."
			)

		self._data_source = None
		self._lookups = {}
		self._norm_stats = None
		self._faces_cache: Dict[Tuple[str, str], torch.Tensor] = {}
		self._verts_cache: Dict[Tuple[str, str, str], torch.Tensor] = {}
		self._v2l_cache: Dict[Tuple[str, str, str], torch.Tensor] = {}
		self._surface_vertices_cache: Dict[Tuple[str, str, str], torch.Tensor] = {}
		self._full_cfg = full_cfg

	def _set_data_context(self, data_source) -> None:
		self._data_source = data_source
		self._norm_stats = self._data_source.get_normalization_stats()
		self._lookups = {}
		for mod, geo in self._data_source.vertex_to_landmark_geodesics.items():
			verts = self._data_source.mesh_vertices_by_modality[mod]
			faces = self._data_source.mesh_faces_by_modality[mod]
			self._lookups[mod] = build_geodesic_descriptor_lookup(
				vertices=verts,
				faces=faces,
				vertex_to_landmark=geo.matrix,
			)

		self._faces_cache.clear()
		self._verts_cache.clear()
		self._v2l_cache.clear()
		self._surface_vertices_cache.clear()

	def attach_data_source(self, data_source) -> None:
		"""Attach a pre-built data source to avoid rebuilding expensive geometry artifacts."""
		self._set_data_context(data_source)

	def _ensure_data_context(self) -> None:
		if self._data_source is not None:
			return
		if self._full_cfg is None:
			return
		device = torch.device("cpu")
		data_source = build_data_source(self._full_cfg, device=device)
		self._set_data_context(data_source)

	def __call__(
		self,
		*,
		pred: State,
		target: State,
		sigma: Sigma,
		descriptor=None,
		chosen_modality_idx: Optional[torch.Tensor] = None,
		chosen_point_idx: Optional[torch.Tensor] = None,
		modality_order: Optional[Tuple[str, ...]] = None,
		timing_enabled: bool = False,
		timing_buffer: Optional[Dict[str, float]] = None,
		observed_mask: Optional[ObservedMask] = None,
	) -> LossOutput:
		device = next(iter(pred.values())).device
		self._ensure_data_context()
 
		projection_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

		def _add_time(name: str, dt: float) -> None:
			if not timing_enabled or timing_buffer is None:
				return
			timing_buffer[name] = timing_buffer.get(name, 0.0) + float(dt)

		def _get_projection_for_modality(modality: str, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
			cached = projection_cache.get(modality)
			if cached is not None:
				return cached
			t0 = time.perf_counter()
			proj = self._data_source.project_points_to_surface(modality, points)
			_add_time("loss.geo.projection", time.perf_counter() - t0)
			proj_pts = proj.points.reshape(-1, 3).to(device=points.device, dtype=points.dtype)
			tri_ids = proj.face_ids.reshape(-1).to(device=points.device, dtype=torch.long)
			projection_cache[modality] = (proj_pts, tri_ids)
			return proj_pts, tri_ids

		diffusion_total = torch.zeros((), device=device)
		terms: Dict[str, torch.Tensor] = {}

		active_modalities = [
			m for m in pred.keys()
			if not (observed_mask and observed_mask.get(m, False))
		]
		if modality_order is None:
			modality_order = tuple(pred.keys())
		mod_to_idx = {m: i for i, m in enumerate(modality_order)}

		if descriptor is None or chosen_modality_idx is None:
			raise ValueError(
				"GeomDistUnsupervisedLoss requires `descriptor` and `chosen_modality_idx` batch context"
			)

		anchor_desc = descriptor.data.to(device=device, dtype=next(iter(pred.values())).dtype)
		if anchor_desc.ndim != 2:
			raise ValueError(f"descriptor.data must have shape [B,L], got {tuple(anchor_desc.shape)}")
		B = anchor_desc.shape[0]
		chosen_modality_idx = chosen_modality_idx.to(device=device, dtype=torch.long)
		if chosen_modality_idx.shape != (B,):
			raise ValueError("chosen_modality_idx must have shape [B]")
		if chosen_point_idx is None:
			chosen_point_idx = torch.zeros(B, device=device, dtype=torch.long)
		else:
			chosen_point_idx = chosen_point_idx.to(device=device, dtype=torch.long)
			if chosen_point_idx.shape != (B,):
				raise ValueError("chosen_point_idx must have shape [B]")

		diff_num = torch.zeros((), device=device)
		diff_den = torch.zeros((), device=device)
		for m in active_modalities:
			m_idx = mod_to_idx.get(m, None)
			if m_idx is None:
				continue
			mask = (chosen_modality_idx == m_idx)
			if not bool(mask.any()):
				continue

			s = sigma[m]
			s_safe = torch.clamp(s, min=self.eps)
			w = (s_safe * s_safe + self.sigma_data * self.sigma_data) / (s_safe * self.sigma_data) ** 2
			err = (pred[m] - target[m]).pow(2)
			reduce_dims = tuple(range(1, err.ndim))
			mse_b = err.mean(dim=reduce_dims)
			loss_b = w * mse_b
			loss_m = loss_b[mask].mean()
			diff_num = diff_num + loss_b[mask].sum()
			diff_den = diff_den + mask.sum().to(device=device, dtype=diff_den.dtype)
			terms[f"diffusion/{m}"] = loss_m
		diffusion_total = diff_num / torch.clamp(diff_den, min=1.0)

		geo_total = torch.zeros((), device=device)
		geo_den = torch.zeros((), device=device)
		for m in active_modalities:
			m_idx = mod_to_idx.get(m, None)
			if m_idx is None:
				continue
			if m not in self._lookups:
				continue
			mask = (chosen_modality_idx != m_idx)
			if not bool(mask.any()):
				continue

			t_geo = time.perf_counter()
			m_pts = self._denormalize_if_needed(m, pred[m])
			m_pts_pick = self._select_points_by_index(m_pts, chosen_point_idx)
			precomputed_projection = None
			if self._data_source is not None and self.surface_point_mode == "projection":
				if self.descriptor_interpolation == "barycentric" or (not self.use_cdist):
					precomputed_projection = _get_projection_for_modality(m, m_pts_pick)
			m_desc = self._descriptor_from_points(
				m,
				m_pts_pick.unsqueeze(1),
				timing_enabled=timing_enabled,
				timing_buffer=timing_buffer,
				precomputed_projection=precomputed_projection,
			).squeeze(1)
			_add_time("loss.geo.total", time.perf_counter() - t_geo)

			geo_b = torch.mean((m_desc - anchor_desc).pow(2), dim=-1)
			s_safe = torch.clamp(sigma[m], min=self.eps)
			geo_b = geo_b / (s_safe + self.eps)

			geo_m = geo_b[mask].mean()
			geo_total = geo_total + geo_b[mask].sum()
			geo_den = geo_den + mask.sum().to(device=device, dtype=geo_den.dtype)
			terms[f"geo/anchor_to_{m}"] = geo_m
		if geo_den > 0:
			geo_total = geo_total / geo_den
		else:
			geo_total = torch.zeros((), device=device)

		surf_total = torch.zeros((), device=device)
		surf_den = torch.zeros((), device=device)
		if self._data_source is not None:
			for m in active_modalities:
				m_idx = mod_to_idx.get(m, None)
				if m_idx is None:
					continue
				mask = (chosen_modality_idx != m_idx)
				if not bool(mask.any()):
					continue
				if m not in getattr(self._data_source, "mesh_vertices_by_modality", {}):
					continue
				pts = self._denormalize_if_needed(m, pred[m])
				pts_pick = self._select_points_by_index(pts, chosen_point_idx)
				if self.use_cdist:
					t_surf = time.perf_counter()
					surf_b = self._differentiable_surface_distance_per_sample(modality=m, points=pts_pick)
					_add_time("loss.surface.cdist", time.perf_counter() - t_surf)
				else:
					t_surf = time.perf_counter()
					if self.surface_point_mode == "projection":
						proj_pts, _ = _get_projection_for_modality(m, pts_pick)
						surf_b = torch.linalg.norm(pts_pick - proj_pts, dim=-1)
					else:
						surf_b = self._surface_distance_via_surface_mapping_per_sample(modality=m, points=pts_pick)
					_add_time("loss.surface.map_distance", time.perf_counter() - t_surf)
				surf_m = surf_b[mask].mean()

				surf_total = surf_total + surf_b[mask].sum()
				surf_den = surf_den + mask.sum().to(device=device, dtype=surf_den.dtype)
				terms[f"surface/{m}"] = surf_m
		if surf_den > 0:
			surf_total = surf_total / surf_den
		else:
			surf_total = torch.zeros((), device=device)

		total = diffusion_total + self.lambda_geo * geo_total + self.lambda_surface * surf_total

		terms["diffusion/total"] = diffusion_total
		terms["geo/total"] = self.lambda_geo * geo_total
		terms["surface/total"] = self.lambda_surface * surf_total
		terms["total"] = total
		return LossOutput(loss=total, terms=terms)

	def _denormalize_if_needed(self, modality: str, x: torch.Tensor) -> torch.Tensor:
		if not self._norm_stats or modality not in self._norm_stats:
			return x
		mean, std = self._norm_stats[modality]
		mean = mean.to(device=x.device, dtype=x.dtype)
		std = std.to(device=x.device, dtype=x.dtype)
		return x * std.view(1, 1, -1) + mean.view(1, 1, -1)

	def _descriptor_from_points(
		self,
		modality: str,
		points: torch.Tensor,
		*,
		timing_enabled: bool = False,
		timing_buffer: Optional[Dict[str, float]] = None,
		precomputed_projection: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
	) -> torch.Tensor:
		def _add_time(name: str, dt: float) -> None:
			if not timing_enabled or timing_buffer is None:
				return
			timing_buffer[name] = timing_buffer.get(name, 0.0) + float(dt)

		flat = points.reshape(-1, 3)

		if self._data_source is None:
			t0 = time.perf_counter()
			desc = self._nearest_descriptor_torch(modality=modality, query_points=flat)
			_add_time("loss.geo.distance_calc", time.perf_counter() - t0)
			return desc.reshape(points.shape[0], points.shape[1], -1)

		interp = self.descriptor_interpolation
		if interp == "barycentric":
			if self.surface_point_mode != "projection":
				raise ValueError(
					"descriptor_interpolation='barycentric' requires surface_point='projection'"
				)
			if precomputed_projection is None:
				t0 = time.perf_counter()
				proj = self._data_source.project_points_to_surface(modality, flat)
				_add_time("loss.geo.projection", time.perf_counter() - t0)
				tri = proj.face_ids.reshape(-1).to(device=flat.device, dtype=torch.long)
				proj_pts = proj.points.reshape(-1, 3).to(device=flat.device, dtype=flat.dtype)
			else:
				proj_pts, tri = precomputed_projection
			mapped_st = flat + (proj_pts - flat).detach()
			t0 = time.perf_counter()
			desc = self._barycentric_descriptor_torch(
				modality=modality,
				triangle_ids=tri,
				query_points=mapped_st,
			)
			_add_time("loss.geo.distance_calc", time.perf_counter() - t0)
		elif interp == "knn":
			t0 = time.perf_counter()
			mapped = self._data_source.surface_points(modality, flat)
			_add_time("loss.geo.surface_map", time.perf_counter() - t0)
			mapped = mapped.to(device=flat.device, dtype=flat.dtype)
			mapped_st = flat + (mapped - flat).detach()
			t0 = time.perf_counter()
			desc = self._knn_descriptor_torch(
				modality=modality,
				query_points=mapped_st,
				n_neighbors=self.knn_neighbors,
			)
			_add_time("loss.geo.distance_calc", time.perf_counter() - t0)
		else:
			t0 = time.perf_counter()
			mapped = self._data_source.surface_points(modality, flat)
			_add_time("loss.geo.surface_map", time.perf_counter() - t0)
			mapped = mapped.to(device=flat.device, dtype=flat.dtype)
			mapped_st = flat + (mapped - flat).detach()
			t0 = time.perf_counter()
			desc = self._nearest_descriptor_torch(modality=modality, query_points=mapped_st)
			_add_time("loss.geo.distance_calc", time.perf_counter() - t0)

		return desc.reshape(points.shape[0], points.shape[1], -1)

	def _get_faces_torch(self, *, modality: str, device: torch.device) -> torch.Tensor:
		key = (modality, str(device))
		faces = self._faces_cache.get(key)
		if faces is None:
			if self._data_source is None:
				raise KeyError(f"No data source available for modality '{modality}'")
			faces_np = self._data_source.mesh_faces_by_modality.get(modality)
			if faces_np is None:
				raise KeyError(f"No faces available for modality '{modality}'")
			faces = torch.as_tensor(faces_np, device=device, dtype=torch.long)
			self._faces_cache[key] = faces
		return faces

	def _get_verts_torch(self, *, modality: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		key = (modality, str(device), str(dtype))
		verts = self._verts_cache.get(key)
		if verts is None:
			if self._data_source is None:
				raise KeyError(f"No data source available for modality '{modality}'")
			verts_np = self._data_source.mesh_vertices_by_modality.get(modality)
			if verts_np is None:
				raise KeyError(f"No vertices available for modality '{modality}'")
			verts = torch.as_tensor(verts_np, device=device, dtype=dtype)
			self._verts_cache[key] = verts
		return verts

	def _get_v2l_torch(self, *, modality: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		key = (modality, str(device), str(dtype))
		v2l = self._v2l_cache.get(key)
		if v2l is None:
			geo = None
			if self._data_source is not None:
				geo = self._data_source.vertex_to_landmark_geodesics.get(modality)
			if geo is None:
				raise KeyError(f"No vertex-to-landmark geodesic matrix for modality '{modality}'")
			v2l = geo.matrix.to(device=device, dtype=dtype)
			self._v2l_cache[key] = v2l
		return v2l

	def _nearest_descriptor_torch(self, *, modality: str, query_points: torch.Tensor) -> torch.Tensor:
		verts = self._get_verts_torch(modality=modality, device=query_points.device, dtype=query_points.dtype)
		v2l = self._get_v2l_torch(modality=modality, device=query_points.device, dtype=query_points.dtype)
		d = torch.cdist(query_points, verts)
		idx = torch.argmin(d, dim=1)
		return v2l[idx]

	def _knn_descriptor_torch(
		self,
		*,
		modality: str,
		query_points: torch.Tensor,
		n_neighbors: int,
	) -> torch.Tensor:
		verts = self._get_verts_torch(modality=modality, device=query_points.device, dtype=query_points.dtype)
		v2l = self._get_v2l_torch(modality=modality, device=query_points.device, dtype=query_points.dtype)
		d = torch.cdist(query_points, verts)
		k = max(1, min(int(n_neighbors), verts.shape[0]))
		d_k, idx_k = torch.topk(d, k=k, dim=1, largest=False)
		desc_k = v2l[idx_k]
		weights = 1.0 / (d_k + self.eps)
		weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=self.eps)
		return (weights.unsqueeze(-1) * desc_k).sum(dim=1)

	def _barycentric_descriptor_torch(
		self,
		*,
		modality: str,
		triangle_ids: torch.Tensor,
		query_points: torch.Tensor,
	) -> torch.Tensor:
		faces = self._get_faces_torch(modality=modality, device=query_points.device)
		verts = self._get_verts_torch(modality=modality, device=query_points.device, dtype=query_points.dtype)
		v2l = self._get_v2l_torch(modality=modality, device=query_points.device, dtype=query_points.dtype)

		tri_vids = faces[triangle_ids]  # [B,3]
		tri = verts[tri_vids]  # [B,3,3]
		v0 = tri[:, 0, :]
		v1 = tri[:, 1, :]
		v2 = tri[:, 2, :]

		e0 = v1 - v0
		e1 = v2 - v0
		p = query_points - v0

		d00 = (e0 * e0).sum(dim=1)
		d01 = (e0 * e1).sum(dim=1)
		d11 = (e1 * e1).sum(dim=1)
		d20 = (p * e0).sum(dim=1)
		d21 = (p * e1).sum(dim=1)

		den = torch.clamp(d00 * d11 - d01 * d01, min=self.eps)
		beta = (d11 * d20 - d01 * d21) / den
		gamma = (d00 * d21 - d01 * d20) / den
		alpha = 1.0 - beta - gamma

		bary = torch.stack([alpha, beta, gamma], dim=1)  # [B,3]
		desc_tri = v2l[tri_vids]  # [B,3,L]
		return (bary.unsqueeze(-1) * desc_tri).sum(dim=1)

	def _differentiable_surface_distance_per_sample(self, *, modality: str, points: torch.Tensor) -> torch.Tensor:
		"""
		Differentiable surface penalty proxy: per-sample distance to nearest mesh vertex.

		This avoids numpy/trimesh projection calls in the loss path, preserving gradients
		w.r.t. `points` for backpropagation.
		"""
		if self._data_source is None:
			return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)

		vertices_np = self._data_source.mesh_vertices_by_modality.get(modality)
		if vertices_np is None:
			return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)

		cache_key = (modality, str(points.device), str(points.dtype))
		verts = self._surface_vertices_cache.get(cache_key)
		if verts is None:
			verts = torch.as_tensor(vertices_np, device=points.device, dtype=points.dtype)
			self._surface_vertices_cache[cache_key] = verts

		if points.ndim != 2 or points.shape[-1] != 3:
			raise ValueError(f"surface points must be [B,3], got {tuple(points.shape)}")

		chunk = max(1, self.surface_chunk_size)
		mins = []
		for start in range(0, points.shape[0], chunk):
			end = min(points.shape[0], start + chunk)
			p = points[start:end]
			d = torch.cdist(p, verts)
			mins.append(d.min(dim=1).values)

		all_min = torch.cat(mins, dim=0)
		return all_min

	def _surface_distance_via_surface_mapping_per_sample(self, *, modality: str, points: torch.Tensor) -> torch.Tensor:
		"""
		Per-sample distance to mapped surface point using configured surface_point mode.

		Uses BaseDataSource.surface_points(), which dispatches to projection or pull.
		"""
		if self._data_source is None:
			return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
		if points.ndim != 2 or points.shape[-1] != 3:
			raise ValueError(f"surface points must be [B,3], got {tuple(points.shape)}")

		surface_pts = self._data_source.surface_points(modality, points)
		surface_pts = surface_pts.to(device=points.device, dtype=points.dtype)
		return torch.linalg.norm(points - surface_pts, dim=-1)

	def _select_points_by_index(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
		"""Select one point per batch sample from [B,N,D] using per-sample indices [B]."""
		if points.ndim != 3:
			raise ValueError(f"points must be [B,N,D], got {tuple(points.shape)}")
		B, N, _ = points.shape
		if idx.shape != (B,):
			raise ValueError(f"idx must be [B], got {tuple(idx.shape)}")
		idx_clamped = torch.clamp(idx, min=0, max=max(N - 1, 0))
		b = torch.arange(B, device=points.device)
		return points[b, idx_clamped, :3]
