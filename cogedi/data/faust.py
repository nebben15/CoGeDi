from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from cogedi.dtypes import State
from cogedi.data.base import BaseDataSource, fit_dim


class FAUSTDataSource(BaseDataSource):
    """
    FAUST-specific data source with *coupled* surface sampling.

    This assumes all modality meshes share the same topology and vertex order.
    Sampling is performed on a reference mesh to obtain face indices and
    barycentric coordinates, which are then applied to all modalities so that
    corresponding points are aligned across modalities.
    """

    name = "faust"

    def __init__(self, cfg, *, modality_dims: Dict[str, int], device: torch.device):
        super().__init__(cfg, modality_dims=modality_dims, device=device)

        meshes_cfg = getattr(cfg, "meshes", None)
        mesh_path = getattr(cfg, "mesh_path", None)

        if meshes_cfg is None and mesh_path is None:
            raise ValueError("FAUSTDataSource requires data.meshes (dict) or data.mesh_path (single)")

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

        self._validate_shared_topology()

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

    def _validate_shared_topology(self) -> None:
        mods = list(self.meshes.keys())
        if not mods:
            return
        ref = self.meshes[mods[0]]
        ref_faces = ref.faces
        for mod in mods[1:]:
            faces = self.meshes[mod].faces
            if faces.shape != ref_faces.shape or not np.array_equal(faces, ref_faces):
                raise ValueError("FAUSTDataSource requires identical faces across modalities")

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

    def _coupled_sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        ref_mod = next(iter(self.meshes.keys()))
        ref_mesh = self.meshes[ref_mod]

        samples, face_idx = trimesh.sample.sample_surface(ref_mesh, batch_size)
        faces = ref_mesh.faces[face_idx]
        tri = ref_mesh.vertices[faces]

        bary = trimesh.triangles.points_to_barycentric(tri, samples)

        out: Dict[str, np.ndarray] = {}
        for mod, mesh in self.meshes.items():
            tri_m = mesh.vertices[faces]
            pts = (
                bary[:, [0]] * tri_m[:, 0]
                + bary[:, [1]] * tri_m[:, 1]
                + bary[:, [2]] * tri_m[:, 2]
            )
            out[mod] = pts.astype(np.float32)
        return out

    def sample_batch(self, batch_size: int) -> State:
        coupled = self._coupled_sample(batch_size)
        x0: State = {}
        for mod, dim in self.modality_dims.items():
            pts = torch.from_numpy(coupled[mod]).float().to(self.device)
            if self._norm_stats is not None:
                mean, std = self._norm_stats[mod]
                pts = (pts - mean.to(self.device)) / std.to(self.device)
            x0[mod] = fit_dim(pts, dim).unsqueeze(1)
        return x0

    def sample_vertices_batch(self) -> State:
        x0: State = {}
        for mod, dim in self.modality_dims.items():
            vertices = np.asarray(self.meshes[mod].vertices, dtype=np.float32)
            pts = torch.from_numpy(vertices).float().to(self.device)
            if self._norm_stats is not None:
                mean, std = self._norm_stats[mod]
                pts = (pts - mean.to(self.device)) / std.to(self.device)
            x0[mod] = fit_dim(pts, dim).unsqueeze(1)
        return x0
