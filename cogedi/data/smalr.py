from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from cogedi.dtypes import State
from cogedi.data.base import BaseDataSource


class SMALRDataSource(BaseDataSource):
    """
    SMALR data source with coupled surface sampling and texture colors.

    Assumes modality meshes have identical topology and vertex order so a shared
    face + barycentric sample from a reference mesh can be transferred across
    modalities to preserve correspondence.

    Output point features are xyzrgb where:
      - xyz use z-score normalization (if enabled)
      - rgb use GeomDist-style normalization: (rgb - 0.5) / sqrt(1/12)
    """

    name = "smalr"

    def __init__(self, cfg, *, modality_dims: Dict[str, int], device: torch.device):
        super().__init__(cfg, modality_dims=modality_dims, device=device)

        for mod, dim in self.modality_dims.items():
            if int(dim) != 6:
                raise ValueError(f"SMALRDataSource requires modality dim=6 (xyzrgb), got {dim} for '{mod}'")

        meshes_cfg = getattr(cfg, "meshes", None)
        mesh_path = getattr(cfg, "mesh_path", None)

        if meshes_cfg is None and mesh_path is None:
            raise ValueError("SMALRDataSource requires data.meshes (dict) or data.mesh_path (single)")

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
            self._validate_texture_visual(mesh, mod, path)
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

    def _validate_texture_visual(self, mesh: trimesh.Trimesh, mod: str, path: str) -> None:
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None) if visual is not None else None
        material = getattr(visual, "material", None) if visual is not None else None
        image = getattr(material, "image", None) if material is not None else None
        if uv is None:
            raise ValueError(f"SMALRDataSource requires UV coordinates for modality '{mod}' ({path})")
        if image is None:
            raise ValueError(f"SMALRDataSource requires a texture image for modality '{mod}' ({path})")

    def _validate_shared_topology(self) -> None:
        mods = list(self.meshes.keys())
        if not mods:
            return
        ref = self.meshes[mods[0]]
        ref_faces = np.asarray(ref.faces)
        for mod in mods[1:]:
            faces = np.asarray(self.meshes[mod].faces)
            if faces.shape != ref_faces.shape or not np.array_equal(faces, ref_faces):
                raise ValueError("SMALRDataSource requires identical faces across modalities")

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

            mean_xyz = running_sum / float(total)
            var_xyz = running_sumsq / float(total) - mean_xyz ** 2
            std_xyz = torch.sqrt(torch.clamp(var_xyz, min=1e-12))

            mean = torch.tensor([mean_xyz[0], mean_xyz[1], mean_xyz[2], 0.5, 0.5, 0.5], dtype=torch.float32)
            std = torch.tensor(
                [std_xyz[0], std_xyz[1], std_xyz[2], np.sqrt(1.0 / 12.0), np.sqrt(1.0 / 12.0), np.sqrt(1.0 / 12.0)],
                dtype=torch.float32,
            )
            stats[mod] = (mean, std)
        return stats

    def get_normalization_stats(self) -> Optional[Dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        return self._norm_stats

    def _sample_texture_rgb(
        self,
        *,
        mesh: trimesh.Trimesh,
        faces: np.ndarray,
        bary: np.ndarray,
    ) -> np.ndarray:
        uv_all = np.asarray(mesh.visual.uv, dtype=np.float32)
        tri_uv = uv_all[faces]  # [N,3,2]
        uv = np.sum(tri_uv * bary[:, :, None], axis=1)  # [N,2]

        image = mesh.visual.material.image
        rgba_u8 = trimesh.visual.uv_to_color(uv, image)
        rgb_01 = rgba_u8[:, :3].astype(np.float32) / 255.0
        return rgb_01

    def _coupled_sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        ref_mod = next(iter(self.meshes.keys()))
        ref_mesh = self.meshes[ref_mod]

        samples, face_idx = trimesh.sample.sample_surface(ref_mesh, batch_size)
        faces = np.asarray(ref_mesh.faces, dtype=np.int64)[face_idx]
        tri = np.asarray(ref_mesh.vertices, dtype=np.float32)[faces]
        bary = trimesh.triangles.points_to_barycentric(tri, samples).astype(np.float32)

        out: Dict[str, np.ndarray] = {}
        for mod, mesh in self.meshes.items():
            tri_m = np.asarray(mesh.vertices, dtype=np.float32)[faces]
            xyz = (
                bary[:, [0]] * tri_m[:, 0]
                + bary[:, [1]] * tri_m[:, 1]
                + bary[:, [2]] * tri_m[:, 2]
            ).astype(np.float32)

            rgb = self._sample_texture_rgb(mesh=mesh, faces=faces, bary=bary)
            out[mod] = np.concatenate([xyz, rgb.astype(np.float32)], axis=1)
        return out

    def sample_batch(self, batch_size: int) -> State:
        coupled = self._coupled_sample(batch_size)
        x0: State = {}
        for mod in self.modality_dims.keys():
            pts = torch.from_numpy(coupled[mod]).float().to(self.device)
            if self._norm_stats is not None:
                mean, std = self._norm_stats[mod]
                pts = (pts - mean.to(self.device)) / std.to(self.device)
            x0[mod] = pts.unsqueeze(1)
        return x0

    def sample_vertices_batch(self) -> State:
        x0: State = {}
        for mod in self.modality_dims.keys():
            mesh = self.meshes[mod]
            xyz = np.asarray(mesh.vertices, dtype=np.float32)

            uv = np.asarray(mesh.visual.uv, dtype=np.float32)
            rgba_u8 = trimesh.visual.uv_to_color(uv, mesh.visual.material.image)
            rgb = (rgba_u8[:, :3].astype(np.float32) / 255.0)

            pts_np = np.concatenate([xyz, rgb], axis=1)
            pts = torch.from_numpy(pts_np).float().to(self.device)
            if self._norm_stats is not None:
                mean, std = self._norm_stats[mod]
                pts = (pts - mean.to(self.device)) / std.to(self.device)
            x0[mod] = pts.unsqueeze(1)
        return x0