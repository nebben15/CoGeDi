from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Type
import importlib

import numpy as np
import torch
import trimesh
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from cogedi.build import DATA_REGISTRY
from cogedi.dtypes import State
from cogedi.utils.distance import compute_vertex_to_landmark_geodesic_matrix


class BaseDataSource(abc.ABC):
    """
    Base class for data sources.

    Implementations must return batches in the standard State format with
    leading batch dimension [B, ...].
    """

    name: str = "base_data_source"

    def __init__(self, cfg, *, modality_dims: Mapping[str, int], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.modality_dims: Dict[str, int] = dict(modality_dims)
        self.landmark_type: str = "id"
        self.landmarks: Dict[str, LandmarkSet] = {}
        self.landmark_label_to_index: Dict[str, Dict[str, int]] = {}
        self.shared_landmark_labels: List[str] = []
        self.mesh_vertices_by_modality: Dict[str, np.ndarray] = {}
        self.mesh_faces_by_modality: Dict[str, np.ndarray] = {}
        self.vertex_to_landmark_geodesics: Dict[str, LandmarkGeodesicMatrix] = {}
        self.surface_point_mode: str = "projection"
        self.surface_point_queries: Dict[str, trimesh.proximity.ProximityQuery] = {}
        self.surface_pull_fields: Dict[str, SDFPullField] = {}

    @abc.abstractmethod
    def sample_batch(self, batch_size: int) -> State:
        """Return a batch of clean samples in normalized space."""

    def sample_vertices_batch(self) -> State:
        """Return all mesh vertices as a batch in normalized space.

        Shape per modality: [N_vertices, 1, D].
        Raises NotImplementedError for data sources that do not expose vertices.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sample_vertices_batch()")

    def get_normalization_stats(self) -> Optional[Dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Return per-modality (mean, std) if available, else None."""
        return None

    def get_landmarks(self, *, normalized: bool = True) -> Dict[str, "LandmarkSet"]:
        """
        Return landmarks by modality.

        If normalized=True, coordinates are returned in normalized space if available.
        """
        if not normalized:
            return self.landmarks
        out: Dict[str, LandmarkSet] = {}
        for mod, lm in self.landmarks.items():
            coords = lm.coords_normalized if lm.coords_normalized is not None else lm.coords
            out[mod] = LandmarkSet(
                coords=coords,
                labels=list(lm.labels),
                vertex_ids=lm.vertex_ids,
                source_path=lm.source_path,
                coords_normalized=lm.coords_normalized,
            )
        return out

    def maybe_build_landmark_geodesics(
        self,
        *,
        vertices_by_modality: Dict[str, np.ndarray],
        faces_by_modality: Dict[str, np.ndarray],
    ) -> None:
        """
        Precompute vertex-to-landmark geodesic matrices once at data source init.

        This runs only when supervision mode is set to landmarks.
        """
        self.mesh_vertices_by_modality = {
            mod: np.asarray(v, dtype=np.float64) for mod, v in vertices_by_modality.items()
        }
        self.mesh_faces_by_modality = {
            mod: np.asarray(f, dtype=np.int32) for mod, f in faces_by_modality.items()
        }

        supervision_mode = str(
            getattr(self.cfg, "_supervision", getattr(self.cfg, "supervision", "full"))
        ).lower()
        if supervision_mode != "landmarks":
            self.vertex_to_landmark_geodesics = {}
            return
        if not self.landmarks:
            self.vertex_to_landmark_geodesics = {}
            return

        landmark_cfg = getattr(self.cfg, "landmark_supervision", None)
        backend_raw = str(getattr(landmark_cfg, "geodesic_backend", "heat")).lower()
        backend = self._resolve_geodesic_backend_name(backend_raw)

        aligned = self.get_landmark_alignment(normalized=False, require_all_modalities=True)
        if not aligned.labels:
            self.vertex_to_landmark_geodesics = {}
            return

        matrices: Dict[str, LandmarkGeodesicMatrix] = {}
        mods = list(self.modality_dims.keys())
        for mod in tqdm(mods, desc="Precompute geodesics", ncols=100):
            vertices = self.mesh_vertices_by_modality.get(mod)
            faces = self.mesh_faces_by_modality.get(mod)
            if vertices is None or faces is None:
                continue

            lm_vertex_ids = aligned.vertex_ids_by_modality.get(mod)
            if lm_vertex_ids is None:
                continue

            lm_vertex_ids_np = self._resolve_landmark_vertex_ids(mod, lm_vertex_ids)
            mat_np = compute_vertex_to_landmark_geodesic_matrix(
                vertices=vertices,
                faces=faces,
                landmark_vertex_ids=lm_vertex_ids_np,
                backend=backend,
                show_progress=True,
                progress_desc=f"{mod}: landmarks",
            )
            matrices[mod] = LandmarkGeodesicMatrix(
                matrix=torch.from_numpy(mat_np.astype(np.float32)),
                labels=list(aligned.labels),
                landmark_vertex_ids=torch.from_numpy(lm_vertex_ids_np.astype(np.int64)),
                backend=backend,
            )

        self.vertex_to_landmark_geodesics = matrices

    def maybe_build_surface_point_helpers(
        self,
        *,
        vertices_by_modality: Dict[str, np.ndarray],
        faces_by_modality: Dict[str, np.ndarray],
    ) -> None:
        """
        Precompute helpers for mapping arbitrary points to the mesh surface.

        - projection: build trimesh proximity queries (BVH-backed)
        - pull: build SDF grid + gradient interpolators
        """
        supervision_mode = str(
            getattr(self.cfg, "_supervision", getattr(self.cfg, "supervision", "full"))
        ).lower()
        if supervision_mode != "landmarks":
            self.surface_point_queries = {}
            self.surface_pull_fields = {}
            return

        build_surface_helpers = bool(getattr(self.cfg, "_build_surface_helpers", True))
        if not build_surface_helpers:
            self.surface_point_queries = {}
            self.surface_pull_fields = {}
            return

        landmark_cfg = getattr(self.cfg, "landmark_supervision", None)
        mode_raw = str(getattr(landmark_cfg, "surface_point", "projection")).lower()
        if mode_raw not in {"projection", "pull"}:
            raise ValueError("surface_point must be 'projection' or 'pull'")

        self.surface_point_mode = mode_raw
        self.surface_point_queries = {}
        self.surface_pull_fields = {}

        resolution = int(getattr(landmark_cfg, "SDF_resolution", 128)) if landmark_cfg is not None else 128

        for mod in self.modality_dims.keys():
            
            vertices = np.asarray(vertices_by_modality.get(mod), dtype=np.float64)
            faces = np.asarray(faces_by_modality.get(mod), dtype=np.int32)
            if vertices.size == 0 or faces.size == 0:
                continue
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            with tqdm(total=2 if self.surface_point_mode == "pull" else 1, desc=f"Precompute surface ({mod})", ncols=100) as pbar:
                self.surface_point_queries[mod] = trimesh.proximity.ProximityQuery(mesh)
                pbar.set_postfix_str("BVH")
                pbar.update(1)

                if self.surface_point_mode == "pull":
                    self.surface_pull_fields[mod] = self._build_sdf_pull_field(mesh, resolution=resolution)
                    pbar.set_postfix_str("SDF+grad")
                    pbar.update(1)

    def project_points_to_surface(
        self,
        modality: str,
        points: torch.Tensor | np.ndarray,
    ) -> "SurfaceProjectionResult":
        """
        Project points to closest mesh points using a precomputed proximity query.

        Supports input shape [N,3] or [B,N,3].
        """
        if modality not in self.surface_point_queries:
            raise KeyError(f"No surface projection helper for modality '{modality}'")

        pts_t = torch.as_tensor(points)
        if pts_t.ndim not in (2, 3) or pts_t.shape[-1] != 3:
            raise ValueError("points must have shape [N,3] or [B,N,3]")

        src_device = pts_t.device
        src_dtype = pts_t.dtype if pts_t.is_floating_point() else torch.float32

        flat = pts_t.detach().cpu().to(torch.float32).reshape(-1, 3).numpy()
        query = self.surface_point_queries[modality]
        closest, distances, tri_ids = query.on_surface(flat)

        faces = self.mesh_faces_by_modality.get(modality)
        vertices = self.mesh_vertices_by_modality.get(modality)
        if faces is None or vertices is None:
            raise KeyError(f"No mesh geometry cached for modality '{modality}'")
        tri_vids = faces[np.asarray(tri_ids, dtype=np.int64)]
        tri_pts = vertices[tri_vids]
        bary = trimesh.triangles.points_to_barycentric(tri_pts, closest)

        if pts_t.ndim == 2:
            out_shape_pts = (pts_t.shape[0], 3)
            out_shape_scal = (pts_t.shape[0],)
        else:
            out_shape_pts = (pts_t.shape[0], pts_t.shape[1], 3)
            out_shape_scal = (pts_t.shape[0], pts_t.shape[1])

        return SurfaceProjectionResult(
            points=torch.from_numpy(closest.astype(np.float32)).to(device=src_device, dtype=src_dtype).reshape(out_shape_pts),
            face_ids=torch.from_numpy(np.asarray(tri_ids, dtype=np.int64)).to(device=src_device).reshape(out_shape_scal),
            barycentric=torch.from_numpy(np.asarray(bary, dtype=np.float32)).to(device=src_device, dtype=src_dtype).reshape(out_shape_pts),
            distances=torch.from_numpy(np.asarray(distances, dtype=np.float32)).to(device=src_device, dtype=src_dtype).reshape(out_shape_scal),
        )

    def pull_points_to_surface(
        self,
        modality: str,
        points: torch.Tensor | np.ndarray,
        *,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Pull points to the zero-level set via x <- x - sdf(x) * n(x).

        Supports input shape [N,3] or [B,N,3].
        """
        if modality not in self.surface_pull_fields:
            raise KeyError(
                f"No pull field for modality '{modality}'. "
                "Set landmark_supervision.surface_point=pull."
            )

        pts_t = torch.as_tensor(points)
        if pts_t.ndim not in (2, 3) or pts_t.shape[-1] != 3:
            raise ValueError("points must have shape [N,3] or [B,N,3]")

        src_device = pts_t.device
        src_dtype = pts_t.dtype if pts_t.is_floating_point() else torch.float32
        x = pts_t.detach().cpu().to(torch.float32).reshape(-1, 3).numpy()

        field = self.surface_pull_fields[modality]
        steps = max(1, int(num_steps))
        for _ in range(steps):
            sdf = np.asarray(field.sdf_interp(x), dtype=np.float32)
            gx = np.asarray(field.grad_interp_x(x), dtype=np.float32)
            gy = np.asarray(field.grad_interp_y(x), dtype=np.float32)
            gz = np.asarray(field.grad_interp_z(x), dtype=np.float32)
            grad = np.stack([gx, gy, gz], axis=1)

            valid = np.isfinite(sdf) & np.isfinite(grad).all(axis=1)
            if np.any(valid):
                norm = np.linalg.norm(grad[valid], axis=1, keepdims=True)
                normal = grad[valid] / np.clip(norm, 1e-8, None)
                x[valid] = x[valid] - sdf[valid, None] * normal

            if np.any(~valid):
                proj = self.project_points_to_surface(modality, torch.from_numpy(x[~valid]))
                x[~valid] = proj.points.detach().cpu().numpy().reshape(-1, 3)

        pulled = torch.from_numpy(x).to(device=src_device, dtype=src_dtype)
        return pulled.reshape(pts_t.shape)

    def surface_points(
        self,
        modality: str,
        points: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Return surface points according to configured surface_point mode."""
        if self.surface_point_mode == "pull":
            return self.pull_points_to_surface(modality, points)
        return self.project_points_to_surface(modality, points).points

    def _build_sdf_pull_field(self, mesh: trimesh.Trimesh, *, resolution: int) -> "SDFPullField":
        try:
            mesh_to_sdf = importlib.import_module("mesh_to_sdf")
        except Exception as exc:
            raise ImportError(
                "surface_point='pull' requires mesh-to-sdf. Install with: pip install mesh-to-sdf"
            ) from exc

        if resolution < 8:
            raise ValueError("SDF_resolution must be >= 8")

        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        extent = bounds[1] - bounds[0]
        pad = 0.05 * max(float(np.max(extent)), 1e-6)
        lo = bounds[0] - pad
        hi = bounds[1] + pad

        with tqdm(total=3, desc=f"SDF field ({resolution}^3)", ncols=100) as pbar:
            axes = [np.linspace(lo[i], hi[i], resolution, dtype=np.float64) for i in range(3)]
            gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
            query = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
            pbar.set_postfix_str("grid")
            pbar.update(1)

            sdf_flat = np.asarray(mesh_to_sdf.mesh_to_sdf(mesh, query), dtype=np.float32)
            sdf_grid = sdf_flat.reshape(resolution, resolution, resolution)
            pbar.set_postfix_str("sdf")
            pbar.update(1)

            dx = float(axes[0][1] - axes[0][0])
            dy = float(axes[1][1] - axes[1][0])
            dz = float(axes[2][1] - axes[2][0])
            grad_x, grad_y, grad_z = np.gradient(sdf_grid, dx, dy, dz, edge_order=1)
            pbar.set_postfix_str("grad")
            pbar.update(1)

        grid_axes = (axes[0], axes[1], axes[2])
        return SDFPullField(
            sdf_interp=RegularGridInterpolator(grid_axes, sdf_grid, bounds_error=False, fill_value=np.nan),
            grad_interp_x=RegularGridInterpolator(grid_axes, grad_x.astype(np.float32), bounds_error=False, fill_value=np.nan),
            grad_interp_y=RegularGridInterpolator(grid_axes, grad_y.astype(np.float32), bounds_error=False, fill_value=np.nan),
            grad_interp_z=RegularGridInterpolator(grid_axes, grad_z.astype(np.float32), bounds_error=False, fill_value=np.nan),
        )

    def vertex_id_landmark_distance(self, modality: str, vertex_id: int, landmark: int | str) -> float:
        """Return geodesic distance between a mesh vertex ID and a landmark."""
        geo = self._get_landmark_geodesic_matrix(modality)
        if vertex_id < 0 or vertex_id >= geo.matrix.shape[0]:
            raise IndexError(
                f"vertex_id {vertex_id} out of range for modality '{modality}' (num_vertices={geo.matrix.shape[0]})"
            )
        landmark_idx = self._landmark_to_index(geo, landmark)
        return float(geo.matrix[vertex_id, landmark_idx].item())

    def point_landmark_distance_barycentric(
        self,
        modality: str,
        face_id: int,
        barycentric: torch.Tensor | np.ndarray | List[float],
        landmark: int | str,
    ) -> float:
        """
        Return geodesic distance for a point on mesh via barycentric interpolation.

        The distance is interpolated from the three triangle-vertex distances.
        """
        geo = self._get_landmark_geodesic_matrix(modality)
        faces = self.mesh_faces_by_modality.get(modality)
        if faces is None:
            raise KeyError(f"No faces available for modality '{modality}'")
        if face_id < 0 or face_id >= faces.shape[0]:
            raise IndexError(f"face_id {face_id} out of range for modality '{modality}'")

        bary = torch.as_tensor(barycentric, dtype=torch.float32).view(-1)
        if bary.numel() != 3:
            raise ValueError("barycentric must contain exactly 3 weights")
        bary_sum = float(bary.sum().item())
        if abs(bary_sum - 1.0) > 1e-3:
            bary = bary / max(bary_sum, 1e-12)

        landmark_idx = self._landmark_to_index(geo, landmark)
        tri = faces[face_id]
        d = geo.matrix[tri, landmark_idx].to(dtype=torch.float32)
        return float((bary * d).sum().item())

    def get_landmark_alignment(
        self,
        *,
        normalized: bool = True,
        labels: Optional[List[str]] = None,
        require_all_modalities: bool = True,
    ) -> "LandmarkAlignment":
        """
        Return landmarks aligned across modalities by label.

        - If labels is None and require_all_modalities=True: uses labels shared by all modalities.
        - If labels is None and require_all_modalities=False: uses union of labels (missing labels per
          modality are encoded with NaN coords and vertex_id/index = -1).
        - If labels is provided: returns those labels in given order.
        """
        source = self.get_landmarks(normalized=normalized)
        if not source:
            return LandmarkAlignment(
                labels=[],
                coords_by_modality={},
                vertex_ids_by_modality={},
                indices_by_modality={},
            )

        modalities = list(source.keys())

        if labels is None:
            if require_all_modalities:
                labels_use = list(self.shared_landmark_labels)
            else:
                ordered: List[str] = []
                seen = set()
                for mod in modalities:
                    for label in source[mod].labels:
                        if label not in seen:
                            seen.add(label)
                            ordered.append(label)
                labels_use = ordered
        else:
            labels_use = list(labels)

        coords_by_modality: Dict[str, torch.Tensor] = {}
        vertex_ids_by_modality: Dict[str, torch.Tensor] = {}
        indices_by_modality: Dict[str, torch.Tensor] = {}

        for mod in modalities:
            lm = source[mod]
            label_to_idx = {label: i for i, label in enumerate(lm.labels)}

            coords_rows: List[torch.Tensor] = []
            vids: List[int] = []
            idxs: List[int] = []

            for label in labels_use:
                i = label_to_idx.get(label, None)
                if i is None:
                    if require_all_modalities:
                        raise ValueError(
                            f"Missing landmark label '{label}' for modality '{mod}'"
                        )
                    coords_rows.append(torch.full((3,), float("nan"), dtype=lm.coords.dtype))
                    vids.append(-1)
                    idxs.append(-1)
                    continue

                coords_rows.append(lm.coords[i])
                vids.append(int(lm.vertex_ids[i].item()))
                idxs.append(i)

            coords_by_modality[mod] = (
                torch.stack(coords_rows, dim=0)
                if coords_rows
                else torch.empty((0, 3), dtype=lm.coords.dtype)
            )
            vertex_ids_by_modality[mod] = torch.tensor(vids, dtype=torch.long)
            indices_by_modality[mod] = torch.tensor(idxs, dtype=torch.long)

        return LandmarkAlignment(
            labels=labels_use,
            coords_by_modality=coords_by_modality,
            vertex_ids_by_modality=vertex_ids_by_modality,
            indices_by_modality=indices_by_modality,
        )

    def load_landmarks_from_cfg(self, *, vertices_by_modality: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Load landmark files from data.landmarks config into self.landmarks."""
        landmarks_cfg = getattr(self.cfg, "landmarks", None)
        if landmarks_cfg is None:
            self.landmarks = {}
            return

        self.landmark_type = str(getattr(landmarks_cfg, "type", "id")).lower()

        landmarks_dict = getattr(landmarks_cfg, "__dict__", None)
        if landmarks_dict is None:
            landmarks_dict = dict(landmarks_cfg)

        loaded: Dict[str, LandmarkSet] = {}
        for mod, value in dict(landmarks_dict).items():
            if mod == "type":
                continue
            if mod not in self.modality_dims:
                continue
            if value is None:
                continue
            path = str(value).strip()
            if not path:
                continue
            loaded[mod] = _parse_landmark_file(path)

            if vertices_by_modality is not None and mod in vertices_by_modality:
                vertices = vertices_by_modality[mod]
                n_verts = int(vertices.shape[0])
                valid = (loaded[mod].vertex_ids == -1) | (
                    (loaded[mod].vertex_ids >= 0) & (loaded[mod].vertex_ids < n_verts)
                )
                if not bool(torch.all(valid)):
                    raise ValueError(
                        f"Landmark vertex IDs out of range for modality '{mod}' in '{path}'"
                    )

                if self.landmark_type == "id":
                    ids = loaded[mod].vertex_ids
                    use_id = ids >= 0
                    if bool(torch.any(use_id)):
                        coords = loaded[mod].coords.clone()
                        idx = ids[use_id].long().detach().cpu().numpy()
                        coords_from_mesh = torch.from_numpy(vertices[idx]).to(coords.dtype)
                        coords[use_id] = coords_from_mesh
                        loaded[mod] = LandmarkSet(
                            coords=coords,
                            labels=loaded[mod].labels,
                            vertex_ids=loaded[mod].vertex_ids,
                            source_path=loaded[mod].source_path,
                            coords_normalized=loaded[mod].coords_normalized,
                        )

        self.landmarks = loaded
        self._build_landmark_index()

    def apply_landmark_normalization(self, stats: Optional[Dict[str, tuple[torch.Tensor, torch.Tensor]]]) -> None:
        """Apply per-modality normalization stats to landmark coordinates."""
        if not self.landmarks:
            return
        if stats is None:
            return
        for mod, lm in list(self.landmarks.items()):
            if mod not in stats:
                continue
            mean, std = stats[mod]
            mean = mean.detach().cpu()
            std = std.detach().cpu()
            if mean.numel() < 3 or std.numel() < 3:
                continue
            coords_norm = (lm.coords - mean[:3]) / std[:3]
            self.landmarks[mod] = LandmarkSet(
                coords=lm.coords,
                labels=lm.labels,
                vertex_ids=lm.vertex_ids,
                source_path=lm.source_path,
                coords_normalized=coords_norm,
            )

    def _build_landmark_index(self) -> None:
        self.landmark_label_to_index = {}
        self.shared_landmark_labels = []
        if not self.landmarks:
            return

        modalities = list(self.landmarks.keys())
        for mod, lm in self.landmarks.items():
            lookup: Dict[str, int] = {}
            for idx, label in enumerate(lm.labels):
                if label in lookup:
                    raise ValueError(
                        f"Duplicate landmark label '{label}' in modality '{mod}' ({lm.source_path})"
                    )
                lookup[label] = idx
            self.landmark_label_to_index[mod] = lookup

        ref_mod = modalities[0]
        ref_labels = self.landmarks[ref_mod].labels
        shared = []
        for label in ref_labels:
            if all(label in self.landmark_label_to_index[m] for m in modalities[1:]):
                shared.append(label)
        self.shared_landmark_labels = shared

    def _resolve_landmark_vertex_ids(self, modality: str, landmark_vertex_ids: torch.Tensor) -> np.ndarray:
        vertices = self.mesh_vertices_by_modality.get(modality)
        if vertices is None:
            raise KeyError(f"No mesh vertices available for modality '{modality}'")
        out = landmark_vertex_ids.detach().cpu().numpy().astype(np.int64).copy()
        need_resolve = out < 0
        if not np.any(need_resolve):
            return out

        coords = self.landmarks[modality].coords.detach().cpu().numpy().astype(np.float64)
        for i in np.where(need_resolve)[0]:
            diff = vertices - coords[i][None, :]
            out[i] = int(np.argmin(np.sum(diff * diff, axis=1)))
        return out

    @staticmethod
    def _landmark_to_index(geo: "LandmarkGeodesicMatrix", landmark: int | str) -> int:
        if isinstance(landmark, int):
            idx = landmark
        else:
            if landmark not in geo.label_to_index:
                raise KeyError(f"Unknown landmark label '{landmark}'. Known: {geo.labels}")
            idx = geo.label_to_index[landmark]
        if idx < 0 or idx >= len(geo.labels):
            raise IndexError(f"landmark index {idx} out of range [0, {len(geo.labels)})")
        return idx

    def _get_landmark_geodesic_matrix(self, modality: str) -> "LandmarkGeodesicMatrix":
        if modality not in self.vertex_to_landmark_geodesics:
            raise KeyError(
                f"No landmark geodesic matrix for modality '{modality}'. "
                "Ensure supervision=landmarks and matrix precomputation is enabled."
            )
        return self.vertex_to_landmark_geodesics[modality]

    @staticmethod
    def _resolve_geodesic_backend_name(name: str) -> str:
        aliases = {
            "heat": "potpourri3d",
            "potpourri3d": "potpourri3d",
            "direct": "pygeodesic",
            "pygeodesic": "pygeodesic",
        }
        key = str(name).lower().strip()
        if key not in aliases:
            known = ", ".join(sorted(aliases.keys()))
            raise ValueError(
                f"Unknown geodesic_backend '{name}'. Supported values: {known}"
            )
        return aliases[key]


@dataclass(frozen=True)
class LandmarkSet:
    coords: torch.Tensor
    labels: List[str]
    vertex_ids: torch.Tensor
    source_path: str
    coords_normalized: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class LandmarkAlignment:
    labels: List[str]
    coords_by_modality: Dict[str, torch.Tensor]
    vertex_ids_by_modality: Dict[str, torch.Tensor]
    indices_by_modality: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class LandmarkGeodesicMatrix:
    matrix: torch.Tensor
    labels: List[str]
    landmark_vertex_ids: torch.Tensor
    backend: str

    @property
    def label_to_index(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}


@dataclass(frozen=True)
class SurfaceProjectionResult:
    points: torch.Tensor
    face_ids: torch.Tensor
    barycentric: torch.Tensor
    distances: torch.Tensor


@dataclass(frozen=True)
class SDFPullField:
    sdf_interp: RegularGridInterpolator
    grad_interp_x: RegularGridInterpolator
    grad_interp_y: RegularGridInterpolator
    grad_interp_z: RegularGridInterpolator


def _parse_landmark_file(path: str) -> LandmarkSet:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Landmark file does not exist: {path}")

    coords_list: List[List[float]] = []
    labels: List[str] = []
    vertex_ids: List[int] = []

    with p.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid landmark row in {path}:{line_idx}. Expected: X Y Z label [vertex_id]"
                )
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid XYZ values in {path}:{line_idx}"
                ) from exc

            if len(parts) == 4:
                label = parts[3]
                vid = -1
            else:
                label = " ".join(parts[3:-1])
                try:
                    vid = int(parts[-1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid vertex_id in {path}:{line_idx}. Last token must be integer."
                    ) from exc

            coords_list.append([x, y, z])
            labels.append(label)
            vertex_ids.append(vid)

    if not coords_list:
        raise ValueError(f"Landmark file is empty: {path}")

    coords = torch.tensor(coords_list, dtype=torch.float32)
    vids = torch.tensor(vertex_ids, dtype=torch.long)
    return LandmarkSet(coords=coords, labels=labels, vertex_ids=vids, source_path=str(p))


def _get(reg: Dict[str, Type], key: str, kind: str):
    if key not in reg:
        known = ", ".join(sorted(reg.keys())) if reg else "(empty)"
        raise KeyError(f"Unknown {kind} '{key}'. Known: {known}")
    return reg[key]


def fit_dim(points: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 3:
        return points
    if dim < 3:
        return points[..., :dim]
    pad = torch.zeros(points.shape[0], points.shape[1], dim - 3, device=points.device, dtype=points.dtype)
    return torch.cat([points, pad], dim=-1)


def build_data_source(cfg, *, device: torch.device) -> BaseDataSource:
    data_cfg = cfg.data
    data_type = str(getattr(data_cfg, "type", "synthetic"))
    modality_dims = dict(data_cfg.modalities.__dict__)

    run_cfg = getattr(cfg, "run", None)
    supervision = getattr(run_cfg, "supervision", None) if run_cfg is not None else None
    if supervision is not None:
        setattr(data_cfg, "_supervision", supervision)

    cls = _get(DATA_REGISTRY, data_type, "data source")
    return cls(data_cfg, modality_dims=modality_dims, device=device)
