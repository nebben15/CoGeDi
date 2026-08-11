from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from cogedi.data.base import BaseDataSource
from cogedi.dtypes import State


class DFAUSTDataSource(BaseDataSource):
    """
    D-FAUST data source with coupled surface sampling over dynamic sequences.

    Expected config schema:
      data:
        type: dfaust
        modalities:
                    A: 4
                    B: 4
        folder: /path/to/folder/containing/hdf5s
        meshes:
          A:
            name: 50004_chicken_wings
          B:
            name: 50020_hips
        times: true
        normalize: true

    Output features per point are assembled as:
            (x, y, z[, t])

    - x,y,z are optionally z-score normalized if data.normalize=true.
    - t is always in [0, 1].
    """

    name = "dfaust"

    def __init__(self, cfg, *, modality_dims: Dict[str, int], device: torch.device):
        super().__init__(cfg, modality_dims=modality_dims, device=device)

        self.include_time = bool(getattr(cfg, "times", False))

        expected_dim = 3 + (1 if self.include_time else 0)
        for mod, dim in self.modality_dims.items():
            if int(dim) != expected_dim:
                raise ValueError(
                    f"DFAUSTDataSource expected modality '{mod}' dim={expected_dim} "
                    f"from times={self.include_time}, got dim={dim}"
                )

        self.normalize = bool(getattr(cfg, "normalize", True))
        self.normalize_samples = int(getattr(cfg, "normalize_samples", 10_000_000))
        self.normalize_chunk = int(getattr(cfg, "normalize_chunk", 1_000_000))

        self._files = self._resolve_hdf5_files(getattr(cfg, "folder", None))
        self._dataset_index = self._build_dataset_index(self._files)

        meshes_cfg = getattr(cfg, "meshes", None)
        if meshes_cfg is None:
            raise ValueError("DFAUSTDataSource requires data.meshes with per-modality sequence names")

        self.sequence_name_by_modality = self._parse_sequence_names(meshes_cfg)

        self.positions_by_modality: Dict[str, np.ndarray] = {}
        self.frames_by_modality: Dict[str, int] = {}
        self.faces_by_modality: Dict[str, np.ndarray] = {}
        self.meshes: Dict[str, trimesh.Trimesh] = {}

        for mod in self.modality_dims.keys():
            seq_name = self.sequence_name_by_modality.get(mod)
            if seq_name is None:
                raise KeyError(f"Missing sequence name for modality '{mod}' in data.meshes")

            file_path = self._dataset_index.get(seq_name)
            if file_path is None:
                raise KeyError(
                    f"Sequence '{seq_name}' for modality '{mod}' was not found in provided D-FAUST hdf5 files"
                )

            with h5py.File(file_path, "r") as h5:
                pos_raw = np.asarray(h5[seq_name], dtype=np.float32)
                pos_tvc = self._to_tvc(pos_raw)

                if "faces" not in h5:
                    raise KeyError(f"Missing 'faces' dataset in {file_path}")
                faces = np.asarray(h5["faces"], dtype=np.int32)

            self.positions_by_modality[mod] = pos_tvc
            self.frames_by_modality[mod] = int(pos_tvc.shape[0])
            self.faces_by_modality[mod] = faces
            self.meshes[mod] = trimesh.Trimesh(vertices=pos_tvc[0], faces=faces, process=False)

        self._validate_shared_topology()

        self._norm_stats: Optional[Dict[str, tuple[torch.Tensor, torch.Tensor]]] = None
        if self.normalize:
            self._norm_stats = self._compute_normalization_stats()

        vertices_by_mod = {mod: arr[0].astype(np.float32, copy=False) for mod, arr in self.positions_by_modality.items()}
        faces_by_mod = {mod: self.faces_by_modality[mod] for mod in self.modality_dims.keys()}

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

    def _resolve_hdf5_files(self, folder_cfg) -> Sequence[Path]:
        if folder_cfg is None:
            raise ValueError("DFAUSTDataSource requires data.folder pointing to one or more hdf5 files")

        candidates: list[Path] = []
        if isinstance(folder_cfg, (list, tuple)):
            raw_paths = [Path(str(p)).expanduser() for p in folder_cfg]
        else:
            raw_paths = [Path(str(folder_cfg)).expanduser()]

        for path in raw_paths:
            if path.is_dir():
                candidates.extend(sorted(path.glob("*.hdf5")))
                candidates.extend(sorted(path.glob("*.h5")))
            elif path.is_file():
                candidates.append(path)
            else:
                raise FileNotFoundError(f"DFAUST folder/file does not exist: {path}")

        # preserve order, remove duplicates
        uniq: list[Path] = []
        seen: set[Path] = set()
        for p in candidates:
            if p not in seen:
                uniq.append(p)
                seen.add(p)

        if not uniq:
            raise FileNotFoundError("No .hdf5/.h5 files found for D-FAUST data.folder")
        return uniq

    def _build_dataset_index(self, files: Sequence[Path]) -> Dict[str, Path]:
        index: Dict[str, Path] = {}
        for file_path in files:
            with h5py.File(file_path, "r") as h5:
                for key in h5.keys():
                    if key == "faces":
                        continue
                    if key not in index:
                        index[key] = file_path
        return index

    def _parse_sequence_names(self, meshes_cfg) -> Dict[str, str]:
        meshes_dict = getattr(meshes_cfg, "__dict__", None)
        if meshes_dict is None:
            meshes_dict = dict(meshes_cfg)

        names: Dict[str, str] = {}
        for mod, spec in dict(meshes_dict).items():
            if mod not in self.modality_dims:
                continue
            spec_dict = getattr(spec, "__dict__", None)
            if spec_dict is None and isinstance(spec, dict):
                spec_dict = spec
            if spec_dict is None:
                raise ValueError(f"data.meshes.{mod} must be a mapping with field 'name'")

            seq_name = spec_dict.get("name", None)
            if seq_name is None or str(seq_name).strip() == "":
                raise ValueError(f"data.meshes.{mod}.name must be set")
            names[mod] = str(seq_name)
        return names

    def _to_tvc(self, arr: np.ndarray) -> np.ndarray:
        """Convert sequence arrays to [T, V, 3]."""
        if arr.ndim != 3:
            raise ValueError(f"Expected D-FAUST sequence array to be 3D, got shape {arr.shape}")

        # Common D-FAUST layout: [V, 3, T]
        if arr.shape[1] == 3:
            return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)

        # Alternative: [T, V, 3]
        if arr.shape[2] == 3:
            return arr.astype(np.float32, copy=False)

        raise ValueError(f"Could not infer axis order for sequence array with shape {arr.shape}")

    def _validate_shared_topology(self) -> None:
        mods = list(self.modality_dims.keys())
        if not mods:
            return

        ref_mod = mods[0]
        ref_faces = self.faces_by_modality[ref_mod]
        ref_vertices = self.positions_by_modality[ref_mod].shape[1]

        for mod in mods[1:]:
            faces = self.faces_by_modality[mod]
            if faces.shape != ref_faces.shape or not np.array_equal(faces, ref_faces):
                raise ValueError("DFAUSTDataSource requires identical faces across modalities")
            n_vertices = self.positions_by_modality[mod].shape[1]
            if n_vertices != ref_vertices:
                raise ValueError("DFAUSTDataSource requires identical vertex count across modalities")

    def _compute_normalization_stats(self) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        stats: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        total = self.normalize_samples
        chunk = max(1, self.normalize_chunk)

        for mod in self.modality_dims.keys():
            running_sum = torch.zeros(3, dtype=torch.float64)
            running_sumsq = torch.zeros(3, dtype=torch.float64)
            seen = 0

            with tqdm(total=total, desc=f"Normalizing D-FAUST ({mod})", ncols=100) as pbar:
                while seen < total:
                    n = min(chunk, total - seen)
                    geom, _ = self._sample_coupled_arrays(n=n, force_mod=mod)
                    pts = torch.from_numpy(geom[mod]).double()
                    running_sum += pts.sum(dim=0)
                    running_sumsq += (pts ** 2).sum(dim=0)
                    seen += n
                    pbar.update(n)

            mean_xyz = running_sum / float(total)
            var_xyz = running_sumsq / float(total) - mean_xyz ** 2
            std_xyz = torch.sqrt(torch.clamp(var_xyz, min=1e-12))

            dim = int(self.modality_dims[mod])
            mean_full = torch.zeros(dim, dtype=torch.float32)
            std_full = torch.ones(dim, dtype=torch.float32)
            mean_full[:3] = mean_xyz.float()
            std_full[:3] = std_xyz.float()
            stats[mod] = (mean_full, std_full)

        return stats

    def _sample_coupled_arrays(
        self,
        *,
        n: int,
        force_mod: Optional[str] = None,
    ) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Coupled sampling: shared face + barycentric + normalized time per sample.

        Returns:
          geom_by_mod: mod -> [N,3]
          t_vals: [N] in [0,1]
        """
        ref_mod = force_mod if force_mod is not None else next(iter(self.modality_dims.keys()))
        ref_mesh = self.meshes[ref_mod]

        samples, face_idx = trimesh.sample.sample_surface(ref_mesh, n)
        faces = ref_mesh.faces[face_idx]
        tri_ref = ref_mesh.vertices[faces]
        bary = trimesh.triangles.points_to_barycentric(tri_ref, samples).astype(np.float32)

        t_vals = np.random.rand(n).astype(np.float32)

        geom_by_mod: Dict[str, np.ndarray] = {}

        for mod in self.modality_dims.keys():
            pos_tvc = self.positions_by_modality[mod]
            frames = self.frames_by_modality[mod]
            frame_idx = np.clip((t_vals * max(frames - 1, 0)).round().astype(np.int64), 0, max(frames - 1, 0))

            faces_mod = self.faces_by_modality[mod]
            tri_vids = faces_mod[face_idx]  # [N,3]

            tri_pts = pos_tvc[frame_idx[:, None], tri_vids, :]  # [N,3,3]
            pts = (
                bary[:, [0]] * tri_pts[:, 0, :]
                + bary[:, [1]] * tri_pts[:, 1, :]
                + bary[:, [2]] * tri_pts[:, 2, :]
            )
            geom_by_mod[mod] = pts.astype(np.float32, copy=False)

        return geom_by_mod, t_vals

    def _compose_features(
        self,
        *,
        mod: str,
        xyz: np.ndarray,
        t_vals: np.ndarray,
    ) -> torch.Tensor:
        parts: list[np.ndarray] = [xyz.astype(np.float32, copy=False)]

        if self.include_time:
            parts.append(np.clip(t_vals[:, None], 0.0, 1.0).astype(np.float32, copy=False))

        feat = np.concatenate(parts, axis=1)
        return torch.from_numpy(feat).float().to(self.device)

    def get_normalization_stats(self) -> Optional[Dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        return self._norm_stats

    def sample_batch(self, batch_size: int) -> State:
        geom, t_vals = self._sample_coupled_arrays(n=batch_size)

        x0: State = {}
        for mod in self.modality_dims.keys():
            xyz = geom[mod]
            if self._norm_stats is not None:
                mean, std = self._norm_stats[mod]
                xyz_t = torch.from_numpy(xyz).float().to(self.device)
                xyz_t = (xyz_t - mean[:3].to(self.device)) / std[:3].to(self.device)
                xyz = xyz_t.detach().cpu().numpy()

            feat_t = self._compose_features(mod=mod, xyz=xyz, t_vals=t_vals)
            x0[mod] = feat_t.unsqueeze(1)

        return x0

    def sample_vertices_batch(self) -> State:
        # Use all vertices from a randomly sampled normalized time t.
        t_scalar = float(np.random.rand())
        x0: State = {}

        for mod in self.modality_dims.keys():
            pos_tvc = self.positions_by_modality[mod]
            frames = self.frames_by_modality[mod]
            frame_idx = int(np.clip(round(t_scalar * max(frames - 1, 0)), 0, max(frames - 1, 0)))

            xyz = pos_tvc[frame_idx].astype(np.float32, copy=False)
            if self._norm_stats is not None:
                mean, std = self._norm_stats[mod]
                xyz_t = torch.from_numpy(xyz).float().to(self.device)
                xyz_t = (xyz_t - mean[:3].to(self.device)) / std[:3].to(self.device)
                xyz = xyz_t.detach().cpu().numpy()

            t_vals = np.full((xyz.shape[0],), t_scalar, dtype=np.float32)
            feat_t = self._compose_features(mod=mod, xyz=xyz, t_vals=t_vals)
            x0[mod] = feat_t.unsqueeze(1)

        return x0
