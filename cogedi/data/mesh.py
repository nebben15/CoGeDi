from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from cogedi.dtypes import State
from cogedi.data.base import BaseDataSource, fit_dim


class GenericMeshDataSource(BaseDataSource):
    """
    Generic mesh surface sampler (geometry only).

    Each modality is sampled independently from its mesh surface. This does
    NOT couple corresponding points across modalities.
    """

    name = "mesh"

    def __init__(self, cfg, *, modality_dims: Dict[str, int], device: torch.device):
        super().__init__(cfg, modality_dims=modality_dims, device=device)

        meshes_cfg = getattr(cfg, "meshes", None)
        mesh_path = getattr(cfg, "mesh_path", None)

        if meshes_cfg is None and mesh_path is None:
            raise ValueError("MeshDataSource requires data.meshes (dict) or data.mesh_path (single)")

        if meshes_cfg is None:
            meshes_cfg = {m: mesh_path for m in self.modality_dims.keys()}

        self.meshes: Dict[str, trimesh.Trimesh] = {}
        meshes_dict = getattr(meshes_cfg, "__dict__", None)
        if meshes_dict is None:
            meshes_dict = dict(meshes_cfg)
        for mod, path in dict(meshes_dict).items():
            if mod not in self.modality_dims:
                raise KeyError(f"Unknown modality '{mod}' in data.meshes")
            mesh = trimesh.load(path, process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Mesh at '{path}' is not a single trimesh.Trimesh")
            self.meshes[mod] = mesh

        self.normalize = bool(getattr(cfg, "normalize", True))
        self.normalize_samples = int(getattr(cfg, "normalize_samples", 10_000_000))
        self.normalize_chunk = int(getattr(cfg, "normalize_chunk", 1_000_000))
        self._norm_stats: Optional[Dict[str, tuple[torch.Tensor, torch.Tensor]]] = None
        if self.normalize:
            self._norm_stats = self._compute_normalization_stats()

        vertices_by_mod = {
            mod: np.asarray(mesh.vertices, dtype=np.float32)
            for mod, mesh in self.meshes.items()
        }
        faces_by_mod = {
            mod: np.asarray(mesh.faces, dtype=np.int32)
            for mod, mesh in self.meshes.items()
        }
        self.load_landmarks_from_cfg(vertices_by_modality=vertices_by_mod)
        self.maybe_build_surface_point_helpers(
            vertices_by_modality=vertices_by_mod,
            faces_by_modality=faces_by_mod,
        )
        self.maybe_build_landmark_geodesics(
            vertices_by_modality=vertices_by_mod,
            faces_by_modality=faces_by_mod,
        )
        self.apply_landmark_normalization(self._norm_stats)

    def _compute_normalization_stats(self) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        stats: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for mod, mesh in self.meshes.items():
            total = self.normalize_samples
            chunk = max(1, self.normalize_chunk)
            running_sum = torch.zeros(3, dtype=torch.float64)
            running_sumsq = torch.zeros(3, dtype=torch.float64)
            seen = 0

            with tqdm(total=total, desc=f"Normalizing Mesh ({mod})", ncols=100) as pbar:
                while seen < total:
                    n = min(chunk, total - seen)
                    samples, _ = trimesh.sample.sample_surface(mesh, n)
                    pts = torch.from_numpy(samples).double()
                    running_sum += pts.sum(dim=0)
                    running_sumsq += (pts ** 2).sum(dim=0)
                    seen += n
                    pbar.update(n)

            mean = running_sum / float(total)
            var = running_sumsq / float(total) - mean ** 2
            std = torch.sqrt(torch.clamp(var, min=1e-12))
            stats[mod] = (mean.float(), std.float())
        return stats

    def get_normalization_stats(self) -> Optional[Dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        return self._norm_stats

    def sample_batch(self, batch_size: int) -> State:
        x0: State = {}
        for mod, dim in self.modality_dims.items():
            mesh = self.meshes.get(mod)
            if mesh is None:
                raise KeyError(f"Missing mesh for modality '{mod}'")
            samples, _ = trimesh.sample.sample_surface(mesh, batch_size)
            pts = torch.from_numpy(samples).float().to(self.device)
            if self._norm_stats is not None:
                mean, std = self._norm_stats[mod]
                pts = (pts - mean.to(self.device)) / std.to(self.device)
            x0[mod] = fit_dim(pts, dim).unsqueeze(1)
        return x0

    def sample_vertices_batch(self) -> State:
        x0: State = {}
        for mod, dim in self.modality_dims.items():
            mesh = self.meshes.get(mod)
            if mesh is None:
                raise KeyError(f"Missing mesh for modality '{mod}'")
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            pts = torch.from_numpy(vertices).float().to(self.device)
            if self._norm_stats is not None:
                mean, std = self._norm_stats[mod]
                pts = (pts - mean.to(self.device)) / std.to(self.device)
            x0[mod] = fit_dim(pts, dim).unsqueeze(1)
        return x0
