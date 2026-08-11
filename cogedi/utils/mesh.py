from __future__ import annotations

from typing import Dict


def _get_meshes_dict(meshes_cfg) -> dict:
    meshes_dict = getattr(meshes_cfg, "__dict__", None)
    if meshes_dict is None:
        return dict(meshes_cfg)
    return dict(meshes_dict)


def load_mesh_paths_from_cfg(full_cfg) -> Dict[str, str]:
    if full_cfg is None:
        raise ValueError("full_cfg is required to load mesh paths")

    data_cfg = getattr(full_cfg, "data", None)
    mesh_path = getattr(data_cfg, "mesh_path", None) if data_cfg is not None else None
    meshes_cfg = getattr(data_cfg, "meshes", None) if data_cfg is not None else None
    modality_cfg = getattr(data_cfg, "modalities", None) if data_cfg is not None else None
    modality_dict = getattr(modality_cfg, "__dict__", None) if modality_cfg is not None else None

    if meshes_cfg is None and mesh_path is None:
        raise ValueError("data.meshes or data.mesh_path must be provided for geodesic distance")

    if meshes_cfg is None:
        if not modality_dict:
            raise ValueError("data.modalities must be set to map mesh_path")
        return {mod: str(mesh_path) for mod in modality_dict.keys()}

    meshes_dict = _get_meshes_dict(meshes_cfg)
    if not meshes_dict:
        raise ValueError("data.meshes must include at least one modality")

    return {str(mod): str(path) for mod, path in meshes_dict.items()}
