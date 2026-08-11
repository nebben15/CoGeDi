from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from cogedi.dtypes import State
from cogedi.data.base import BaseDataSource, fit_dim


def _get_attr(obj, path: str, default):
	cur = obj
	for key in path.split("."):
		cur = getattr(cur, key, None)
		if cur is None:
			return default
	return cur


def _sample_sphere(num_points: int, device: torch.device) -> torch.Tensor:
	x = torch.randn(num_points, 3, device=device)
	x = torch.nn.functional.normalize(x, dim=1)
	# match GeomDist scaling
	x = x / (1.0 / 3.0) ** 0.5
	return x


def _sample_ellipsoid(num_points: int, device: torch.device) -> torch.Tensor:
	x = _sample_sphere(num_points, device)
	scales = torch.tensor([1.5, 1.0, 0.7], device=device)
	return x * scales


def _sample_pyramid(num_points: int, device: torch.device) -> torch.Tensor:
	xy = (torch.rand(num_points, 2, device=device) * 2.0) - 1.0
	height = 1.0 - torch.max(xy.abs(), dim=-1, keepdim=True).values
	return torch.cat([xy, height], dim=-1)


def _sample_plane(num_points: int, device: torch.device) -> torch.Tensor:
	samples = torch.rand(num_points, 3, device=device) - 0.5
	samples[:, 2] = 0
	# match GeomDist normalization
	samples = samples / (2 / 9 * 2 * 0.5 ** 3) ** 0.5
	return samples


def _sample_volume(num_points: int, device: torch.device) -> torch.Tensor:
	samples = (torch.rand(num_points, 3, device=device) - 0.5) / (1 / 12) ** 0.5
	return samples


def _sample_gaussian(num_points: int, device: torch.device) -> torch.Tensor:
	return torch.randn(num_points, 3, device=device)


class SyntheticDataSource(BaseDataSource):
	"""Synthetic primitives sampler."""

	name = "synthetic"

	def __init__(self, cfg, *, modality_dims: Dict[str, int], device: torch.device):
		super().__init__(cfg, modality_dims=modality_dims, device=device)
		self.shape_types = list(getattr(cfg, "shape_types", ["sphere", "ellipsoid", "pyramid"]))
		if not self.shape_types:
			self.shape_types = ["sphere"]

	def sample_batch(self, batch_size: int) -> State:
		shape_cycle = list(self.shape_types)
		x0: State = {}
		for i, (mod, dim) in enumerate(self.modality_dims.items()):
			shape = shape_cycle[i % len(shape_cycle)]
			if shape == "sphere":
				pts = _sample_sphere(batch_size, self.device)
			elif shape == "ellipsoid":
				pts = _sample_ellipsoid(batch_size, self.device)
			elif shape == "pyramid":
				pts = _sample_pyramid(batch_size, self.device)
			elif shape == "plane":
				pts = _sample_plane(batch_size, self.device)
			elif shape == "volume":
				pts = _sample_volume(batch_size, self.device)
			elif shape == "gaussian":
				pts = _sample_gaussian(batch_size, self.device)
			else:
				raise ValueError(f"Unknown synthetic shape '{shape}'")

			x0[mod] = fit_dim(pts, dim).unsqueeze(1)
		return x0


def generate_synthetic_batch(
	*,
	modality_dims: Dict[str, int],
	batch_size: int,
	device: torch.device,
	shape_types: Optional[Iterable[str]] = None,
) -> State:
	cfg = type("_Cfg", (), {"shape_types": list(shape_types) if shape_types is not None else None})
	return SyntheticDataSource(cfg, modality_dims=modality_dims, device=device).sample_batch(batch_size)
