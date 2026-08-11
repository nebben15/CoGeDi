from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import numpy as np
import open3d as o3d
import trimesh

from cogedi.build import build_sampling, build_denoise
from cogedi.dtypes import Descriptor, ObservedMask, State
from cogedi.orch.denoise import GuidanceConfig
from cogedi.data.base import build_data_source
from cogedi.utils.descriptors import (
    GeodesicDescriptorLookup,
    build_geodesic_descriptor_lookup,
    geodesic_descriptor_from_point_knn_weighted,
    geodesic_descriptor_from_point_nearest_vertex,
    geodesic_descriptor_from_triangle_barycentric,
)


def _get_attr(obj, path: str, default):
    cur = obj
    for key in path.split("."):
        cur = getattr(cur, key, None)
        if cur is None:
            return default
    return cur


def _get_ckpt_epoch_tag(ckpt_path: str) -> str:
    name = os.path.basename(ckpt_path)
    match = re.search(r"checkpoint[-_](?:epoch[-_])?(\d+)", name)
    return match.group(1) if match else "unknown"


def _resolve_output_dir(cfg) -> Optional[str]:
    paths_cfg = getattr(cfg, "paths", None)
    samples_base = getattr(paths_cfg, "samples", None) if paths_cfg is not None else None
    selector_keys = {"local", "slurm", "default"}
    if samples_base is not None and (isinstance(samples_base, dict) or hasattr(samples_base, "__dict__")):
        obj = samples_base if isinstance(samples_base, dict) else getattr(samples_base, "__dict__", {})
        if obj and set(obj.keys()).issubset(selector_keys):
            env = getattr(cfg.run, "env", None) or getattr(cfg.run, "environment", None) or "local"
            env = str(env).lower()
            if env in obj:
                samples_base = obj[env]
            elif "default" in obj:
                samples_base = obj["default"]
            else:
                samples_base = None
    exp_name = getattr(cfg.run, "experiment_name", None)
    if samples_base and exp_name:
        return os.path.join(samples_base, exp_name)
    return samples_base


def _resolve_checkpoint_path(cfg) -> str:
    # Resolve checkpoint from config name/path + environment-specific base directory.
    sample_cfg = getattr(cfg, "sample", None)
    ckpt = getattr(sample_cfg, "checkpoint", None) if sample_cfg is not None else None
    if ckpt:
        if os.path.isabs(ckpt):
            return ckpt

        paths_cfg = getattr(cfg, "paths", None)
        ckpt_base = getattr(paths_cfg, "checkpoints", None) if paths_cfg is not None else None
        selector_keys = {"local", "slurm", "default"}
        if ckpt_base is not None and (isinstance(ckpt_base, dict) or hasattr(ckpt_base, "__dict__")):
            obj = ckpt_base if isinstance(ckpt_base, dict) else getattr(ckpt_base, "__dict__", {})
            if obj and set(obj.keys()).issubset(selector_keys):
                env = getattr(cfg.run, "env", None) or getattr(cfg.run, "environment", None) or "local"
                env = str(env).lower()
                if env in obj:
                    ckpt_base = obj[env]
                elif "default" in obj:
                    ckpt_base = obj["default"]
                else:
                    ckpt_base = None
        exp_name = getattr(cfg.run, "experiment_name", None)
        if not ckpt_base:
            raise ValueError("paths.checkpoints is required when sample.checkpoint is a name")
        if exp_name:
            ckpt_base = os.path.join(ckpt_base, exp_name)

        ckpt_name = ckpt if ckpt.endswith(".pth") else f"{ckpt}.pth"
        return os.path.join(ckpt_base, ckpt_name)

    paths_cfg = getattr(cfg, "paths", None)
    ckpt_base = getattr(paths_cfg, "checkpoints", None) if paths_cfg is not None else None
    selector_keys = {"local", "slurm", "default"}
    if ckpt_base is not None and (isinstance(ckpt_base, dict) or hasattr(ckpt_base, "__dict__")):
        obj = ckpt_base if isinstance(ckpt_base, dict) else getattr(ckpt_base, "__dict__", {})
        if obj and set(obj.keys()).issubset(selector_keys):
            env = getattr(cfg.run, "env", None) or getattr(cfg.run, "environment", None) or "local"
            env = str(env).lower()
            if env in obj:
                ckpt_base = obj[env]
            elif "default" in obj:
                ckpt_base = obj["default"]
            else:
                ckpt_base = None
    exp_name = getattr(cfg.run, "experiment_name", None)
    if not ckpt_base:
        raise ValueError("Missing sample.checkpoint and paths.checkpoints; cannot resolve checkpoint")
    if exp_name:
        ckpt_base = os.path.join(ckpt_base, exp_name)

    raise ValueError("sample.checkpoint is required to select the checkpoint name")


def _init_noise(
    modality_dims: Dict[str, int],
    batch_size: int,
    num_points: int,
    device: torch.device,
    target: str,
    noise_mesh: Optional[str],
) -> State:
    target = str(target).lower()

    def _noise_for_dim(dim: int) -> torch.Tensor:
        if target == "gaussian":
            return torch.randn(batch_size, num_points, dim, device=device)
        if target == "uniform":
            return (torch.rand(batch_size, num_points, dim, device=device) - 0.5) / np.sqrt(1 / 12)
        if target == "sphere":
            n = torch.randn(batch_size, num_points, dim, device=device)
            n = torch.nn.functional.normalize(n, dim=-1)
            return n / np.sqrt(1 / 3)
        if target == "mesh":
            if noise_mesh is None:
                raise ValueError("sample.noise_mesh is required when target='mesh'")
            if dim != 3:
                raise ValueError("mesh target only supports 3D geometry")
            mesh = trimesh.load(noise_mesh)
            pts, _ = trimesh.sample.sample_surface(mesh, batch_size * num_points)
            pts = torch.from_numpy(pts).float().to(device)
            return pts.view(batch_size, num_points, 3)
        raise ValueError("sample.target must be one of: gaussian, uniform, sphere, mesh")

    return {m: _noise_for_dim(dim) for m, dim in modality_dims.items()}


def _select_output_modalities(modality_dims: Dict[str, int], cfg, mode: str) -> Tuple[str, ...]:
    sample_cfg = getattr(cfg, "sample", None)
    prefer = getattr(sample_cfg, "modality", None) if sample_cfg is not None else None

    if mode == "joint":
        return tuple(modality_dims.keys())

    if prefer:
        if prefer not in modality_dims:
            raise KeyError(f"Unknown modality '{prefer}' for sampling. Known: {list(modality_dims)}")
        return (prefer,)

    if len(modality_dims) == 1:
        return (next(iter(modality_dims.keys())),)

    return tuple(modality_dims.keys())


def _build_guidance_cfg(cfg, *, schedule, mode: str) -> GuidanceConfig:
    # Guidance is optional; disabled unless explicitly enabled with non-zero scale.
    sample_cfg = getattr(cfg, "sample", None)
    g_cfg = getattr(sample_cfg, "guidance", None) if sample_cfg is not None else None

    if g_cfg is None:
        return GuidanceConfig(scale=0.0, mode=mode, sigma_max=float(schedule.cfg.sigma_max), enabled=False)

    scale = float(getattr(g_cfg, "scale", 0.0))
    enabled = bool(getattr(g_cfg, "enabled", True)) and scale != 0.0
    sigma_max = float(getattr(g_cfg, "sigma_max", getattr(schedule.cfg, "sigma_max", 1.0)))
    g_mode = str(getattr(g_cfg, "mode", mode)).lower()
    return GuidanceConfig(scale=scale, mode=g_mode, sigma_max=sigma_max, enabled=enabled)


def _denormalize_state_scaled(state: State, model, sig_max, sig) -> State:
    out: State = {}
    for m, v in state.items():
        try:
            s = 1 - (sig[m][0]/sig_max[m][0])
            out[m] = model.denormalize(v, m, s)
        except RuntimeError:
            out[m] = v
    return out

def _denormalize_state(state: State, model) -> State:
    out: State = {}
    for m, v in state.items():
        try:
            out[m] = model.denormalize(v, m)
        except RuntimeError:
            out[m] = v
    return out


def _build_conditional_from_cfg(
    cfg,
    *,
    batch_size: int,
    num_points: int,
    modality_dims: Dict[str, int],
    model,
) -> Tuple[Optional[ObservedMask], Optional[State], Optional[str], Optional[Dict[str, str]]]:
    sample_cfg = getattr(cfg, "sample", None)
    cond = getattr(sample_cfg, "conditional", None) if sample_cfg is not None else None
    if cond is None:
        return None, None, None, None

    cond_dict = getattr(cond, "__dict__", None)
    if cond_dict is None:
        cond_dict = dict(cond)

    observed_mask: ObservedMask = {}
    observed: State = {}
    cond_name = None
    cond_names: Optional[Dict[str, str]] = None

    normalize_input = not bool(getattr(sample_cfg, "conditional_normalized", False))

    model_device = next(model.parameters()).device

    for mod, value in dict(cond_dict).items():
        if mod == "name":
            cond_name = str(value)
            continue
        if mod == "names":
            if isinstance(value, dict):
                cond_names = {str(k): str(v) for k, v in value.items()}
            continue
        if mod not in modality_dims:
            raise KeyError(f"Unknown modality '{mod}' in sample.conditional")
        dim = modality_dims[mod]

        arr = torch.as_tensor(value, dtype=torch.float32, device=model_device)
        if arr.ndim == 1:
            if arr.shape[0] != dim:
                raise ValueError(f"sample.conditional['{mod}'] must have length {dim}")
            arr = arr.view(1, 1, dim).repeat(batch_size, num_points, 1)
        elif arr.ndim == 2:
            if arr.shape[1] != dim:
                raise ValueError(f"sample.conditional['{mod}'] rows must have length {dim}")
            if arr.shape[0] == 1:
                arr = arr.view(1, 1, dim).repeat(batch_size, num_points, 1)
            elif arr.shape[0] == batch_size:
                arr = arr.view(batch_size, 1, dim).repeat(1, num_points, 1)
            else:
                raise ValueError(f"sample.conditional['{mod}'] must have 1 or {batch_size} rows")
        else:
            raise ValueError(f"sample.conditional['{mod}'] must be a 1D or 2D array")

        if normalize_input:
            try:
                arr = model.normalize(arr, mod)
            except RuntimeError:
                pass

        observed_mask[mod] = True
        observed[mod] = arr

    return observed_mask, observed, cond_name, cond_names


def _write_point_clouds(
    x: torch.Tensor,
    out_dir: str,
    base_name: str,
    *,
    ascii: bool,
    colors: Optional[np.ndarray] = None,
    correspondence_colors: Optional[np.ndarray] = None,
    texture_from_points: bool = False,
    time_from_points: bool = False,
    time_channel: Optional[int] = None,
    suffix: Optional[str] = None,
) -> str:
    x_np = x.detach().cpu().numpy().astype(np.float32)
    pts = x_np.reshape(-1, x_np.shape[-1])
    if pts.shape[1] < 3:
        raise ValueError("Point cloud must have at least 3 channels (x,y,z)")

    xyz = pts[:, :3].astype(np.float32, copy=False)

    rgb_u8: Optional[np.ndarray] = None
    if texture_from_points:
        if pts.shape[1] < 6:
            raise ValueError("sample.texture=true requires point features to contain rgb at channels 3:6")
        rgb = np.clip(pts[:, 3:6], 0.0, 1.0)
        rgb_u8 = np.clip(np.rint(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
    elif colors is not None:
        if len(colors) != len(pts):
            raise ValueError("colors must match number of points")
        c = np.clip(colors.astype(np.float32, copy=False), 0.0, 1.0)
        rgb_u8 = np.clip(np.rint(c * 255.0), 0.0, 255.0).astype(np.uint8)

    corr_rgb_u8: Optional[np.ndarray] = None
    if correspondence_colors is not None:
        if len(correspondence_colors) != len(pts):
            raise ValueError("correspondence_colors must match number of points")
        c_corr = np.clip(correspondence_colors.astype(np.float32, copy=False), 0.0, 1.0)
        corr_rgb_u8 = np.clip(np.rint(c_corr * 255.0), 0.0, 255.0).astype(np.uint8)

    t_vals: Optional[np.ndarray] = None
    if time_from_points:
        if time_channel is None:
            if pts.shape[1] >= 7:
                idx = 6
            elif pts.shape[1] >= 4:
                idx = 3
            else:
                raise ValueError("time_from_points=true requires point features to contain t at channel 3 or 6")
        else:
            idx = int(time_channel)
            if idx < 0 or idx >= pts.shape[1]:
                raise ValueError(f"time_channel={idx} is out of range for feature dimension {pts.shape[1]}")
        t_vals = pts[:, idx].astype(np.float32, copy=False)

    name = base_name
    if suffix:
        name = f"{name}_{suffix}"
    out_path = os.path.join(out_dir, f"{name}.ply")

    with open(out_path, "wb") as f:
        header = [
            "ply",
            "format ascii 1.0" if bool(ascii) else "format binary_little_endian 1.0",
            f"element vertex {xyz.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if rgb_u8 is not None:
            header.extend([
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ])
        if corr_rgb_u8 is not None:
            header.extend([
                "property uchar corr_red",
                "property uchar corr_green",
                "property uchar corr_blue",
            ])
        if t_vals is not None:
            header.append("property float t")
        header.append("end_header")
        header_text = "\n".join(header) + "\n"
        f.write(header_text.encode("ascii"))

        if bool(ascii):
            cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
            fmts = ["%.6f", "%.6f", "%.6f"]
            if rgb_u8 is not None:
                cols.extend([rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]])
                fmts.extend(["%d", "%d", "%d"])
            if corr_rgb_u8 is not None:
                cols.extend([corr_rgb_u8[:, 0], corr_rgb_u8[:, 1], corr_rgb_u8[:, 2]])
                fmts.extend(["%d", "%d", "%d"])
            if t_vals is not None:
                cols.append(t_vals)
                fmts.append("%.6f")
            arr = np.column_stack(cols)
            np.savetxt(f, arr, fmt=" ".join(fmts))
        else:
            fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if rgb_u8 is not None:
                fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
            if corr_rgb_u8 is not None:
                fields.extend([("corr_red", "u1"), ("corr_green", "u1"), ("corr_blue", "u1")])
            if t_vals is not None:
                fields.append(("t", "<f4"))

            out = np.empty(xyz.shape[0], dtype=np.dtype(fields))
            out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            if rgb_u8 is not None:
                out["red"], out["green"], out["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
            if corr_rgb_u8 is not None:
                out["corr_red"], out["corr_green"], out["corr_blue"] = corr_rgb_u8[:, 0], corr_rgb_u8[:, 1], corr_rgb_u8[:, 2]
            if t_vals is not None:
                out["t"] = t_vals
            out.tofile(f)

    return out_path


def _compute_xyz_colors(points: np.ndarray) -> np.ndarray:
    pts = points[:, :3]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    den = np.maximum(maxs - mins, 1e-12)
    colors = (pts - mins) / den
    return np.clip(colors, 0.0, 1.0)


def _swap_texture_rgb_in_state(state: State, *, modality_a: str, modality_b: str) -> State:
    if modality_a not in state:
        raise KeyError(f"Unknown modality '{modality_a}' for swap_texture_colors")
    if modality_b not in state:
        raise KeyError(f"Unknown modality '{modality_b}' for swap_texture_colors")

    a = state[modality_a]
    b = state[modality_b]
    if a.shape[-1] < 6 or b.shape[-1] < 6:
        raise ValueError(
            "swap_texture_colors requires both modalities to have at least 6 channels (xyz + rgb)"
        )

    out: State = {m: v for m, v in state.items()}
    a_out = a.clone()
    b_out = b.clone()
    a_rgb = a[:, :, 3:6].clone()
    b_rgb = b[:, :, 3:6].clone()
    a_out[:, :, 3:6] = b_rgb
    b_out[:, :, 3:6] = a_rgb
    out[modality_a] = a_out
    out[modality_b] = b_out
    return out


def _sanitize_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")


def _build_descriptor_lookups(data_source) -> Dict[str, GeodesicDescriptorLookup]:
    lookups: Dict[str, GeodesicDescriptorLookup] = {}
    for mod, geo in getattr(data_source, "vertex_to_landmark_geodesics", {}).items():
        verts = data_source.mesh_vertices_by_modality.get(mod)
        faces = data_source.mesh_faces_by_modality.get(mod)
        if verts is None or faces is None:
            continue
        lookups[mod] = build_geodesic_descriptor_lookup(
            vertices=verts,
            faces=faces,
            vertex_to_landmark=geo.matrix,
        )
    return lookups


def _normalize_descriptor_interpolation(raw: str) -> str:
    interp = str(raw).strip().lower()
    if interp == "knn" or interp == "knn_weighted":
        return "knn"
    if interp == "nearest":
        return "nearest_neighbor"
    return interp


def _build_null_descriptor(*, batch_size: int, desc_dim: int, device: torch.device, dtype: torch.dtype) -> Descriptor:
    data = torch.full((batch_size, desc_dim), float("nan"), device=device, dtype=dtype)
    return Descriptor(type="geodesic", data=data)


def _build_conditional_descriptor(
    *,
    observed_mask: Optional[ObservedMask],
    observed: Optional[State],
    model,
    data_source,
    lookups: Dict[str, GeodesicDescriptorLookup],
    descriptor_interpolation: str,
    knn_neighbors: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Descriptor:
    if observed_mask is None or observed is None:
        raise ValueError("Landmarks conditional sampling requires observed_mask and observed state")

    observed_mods = [m for m, is_obs in observed_mask.items() if is_obs and m in observed]
    if len(observed_mods) != 1:
        raise ValueError(
            "Landmarks conditional sampling currently requires exactly one observed modality in sample.conditional"
        )

    modality = observed_mods[0]
    if modality not in lookups:
        raise KeyError(f"No descriptor lookup for conditioned modality '{modality}'")

    cond = observed[modality]
    if cond.ndim != 3 or cond.shape[1] < 1 or cond.shape[-1] < 3:
        raise ValueError(
            f"Expected observed['{modality}'] to have shape [B,N,D] with N>=1 and D>=3, got {tuple(cond.shape)}"
        )

    points = cond[:, 0, :3].to(device=device, dtype=dtype)
    try:
        points = model.denormalize(points, modality)
    except RuntimeError:
        pass

    lookup = lookups[modality]
    if descriptor_interpolation == "barycentric":
        proj = data_source.project_points_to_surface(modality, points)
        desc = geodesic_descriptor_from_triangle_barycentric(
            lookup,
            triangle_ids=proj.face_ids.reshape(-1),
            barycentric=proj.barycentric.reshape(-1, 3),
            output_device=device,
            output_dtype=dtype,
        )
    elif descriptor_interpolation == "nearest_neighbor":
        desc = geodesic_descriptor_from_point_nearest_vertex(
            lookup,
            points=points,
            output_device=device,
            output_dtype=dtype,
        )
    elif descriptor_interpolation == "knn":
        desc = geodesic_descriptor_from_point_knn_weighted(
            lookup,
            points=points,
            n_neighbors=knn_neighbors,
            output_device=device,
            output_dtype=dtype,
        )
    else:
        raise ValueError("descriptor_interpolation must be one of: barycentric, nearest_neighbor, knn")

    return Descriptor(type="geodesic", data=desc)


def _sample_with_batches(
    *,
    art,
    device: torch.device,
    modality_dims: Dict[str, int],
    total_points: int,
    steps: int,
    target: str,
    noise_mesh: Optional[str],
    observed_mask: Optional[ObservedMask],
    observed: Optional[State],
    sampling_descriptor: Optional[Descriptor],
    guidance: GuidanceConfig,
    num_batches: int = 1,
    return_trajectory: bool = False,
    trajectory_space: str = "denoised",
) -> tuple[State, Optional[Sequence], Optional[np.ndarray]]:
    """Sample points in sub-batches and concatenate results.
    
    Returns:
        (x0_hat_final, traj, None)
    """
    ctx, denoise_fn = build_denoise(
        model=art.model,
        conditioning=art.conditioning,
        observed_mask=observed_mask,
        observed=observed,
        guidance=guidance,
        descriptor=sampling_descriptor,
    )

    def _scaled_latent_for_traj(x_state: State, sigma_state: Dict[str, torch.Tensor], *, initial: bool) -> State:
        out: State = {}
        for m, x_m in x_state.items():
            s = sigma_state[m]
            while s.ndim < x_m.ndim:
                s = s.unsqueeze(-1)
            if initial:
                # Match original GeomDist export: initial latent is divided by sigma_0.
                denom = torch.clamp(s, min=1e-12)
            else:
                # Match original GeomDist export: intermediate states use x / sqrt(1 + sigma^2).
                denom = torch.sqrt(1.0 + s * s)
            out[m] = x_m / denom
        return out

    traj_mode = str(trajectory_space).lower().strip()
    # Two trajectory representations are supported:
    # - denoised: model x0 prediction at each step
    # - legacy_scaled_latent: GeomDist-style solver-state export
    if traj_mode not in {"denoised", "legacy_scaled_latent"}:
        raise ValueError("sample.trajectory_space must be one of: denoised, legacy_scaled_latent")
    
    points_per_batch = max(1, total_points // num_batches)
    remainder = total_points % num_batches
    batch_sizes = [points_per_batch + (1 if i < remainder else 0) for i in range(num_batches)]
    
    x0_hat_batches: Dict[str, list] = {m: [] for m in modality_dims.keys()}
    traj_batches: Optional[list] = [[] for _ in range(num_batches)] if return_trajectory else None
    
    for batch_idx, batch_size in enumerate(batch_sizes):
        if num_batches > 1:
            print(f"  Sampling batch {batch_idx + 1}/{num_batches} ({batch_size} points)")
        
        with torch.no_grad():
            # Build EDM sigma schedule once per batch.
            sigmas = art.schedule.sampling_sigmas(steps, batch_size, observed_mask=observed_mask)
            
            x_init = _init_noise(
                modality_dims,
                batch_size,
                num_points=1,
                device=device,
                target=target,
                noise_mesh=noise_mesh,
            )
            if observed is not None:
                for m, v in observed.items():
                    if m in x_init:
                        x_init[m] = v.to(device)
            
            sigma0 = sigmas[0]
            # Match EDM initialization: x ~ N(0, I) scaled by sigma_0.
            for m in x_init.keys():
                if observed_mask and observed_mask.get(m, False):
                    continue
                s0 = sigma0[m]
                while s0.ndim < x_init[m].ndim:
                    s0 = s0.unsqueeze(-1)
                x_init[m] = x_init[m] * s0
            x = x_init
            if return_trajectory:
                if traj_mode == "denoised":
                    traj_local = [_denormalize_state_scaled(_scaled_latent_for_traj(x, sigmas[0], initial=True), art.model, sigmas[0], sigmas[0])]
                    # traj_local = [denoise_fn(x, sigmas[0], observed_mask)]
                else:
                    traj_local = [_scaled_latent_for_traj(x, sigmas[0], initial=True)]
            else:
                traj_local = None
            
            for i in range(len(sigmas) - 1):
                sigma_cur = sigmas[i]
                parts = []
                for m in sorted(sigma_cur.keys()):
                    s = sigma_cur[m]
                    s_val = s.mean().item() if s.numel() > 0 else float("nan")
                    parts.append(f"{m}={s_val:.6f}")
                print(f"sigma[{i}]: " + ", ".join(parts))
                
                x = art.solver.step(
                    x=x,
                    sigma=sigmas[i],
                    sigma_next=sigmas[i + 1],
                    denoise_fn=denoise_fn,
                    observed_mask=observed_mask,
                )
                # Re-enforce hard conditioning after each solver step.
                x = art.conditioning.clamp(x, ctx)
                if return_trajectory:
                    if traj_mode == "denoised":
                        traj_local.append(_denormalize_state_scaled(_scaled_latent_for_traj(x, sigmas[i + 1], initial=False), art.model, sigmas[0], sigma_cur))
                        # traj_local.append(denoise_fn(x, sigmas[i + 1], observed_mask))
                    else:
                        print("trigger")
                        traj_local.append(_scaled_latent_for_traj(x, sigmas[i + 1], initial=False))
            
            x0_hat = denoise_fn(x, sigmas[-1], observed_mask)
        
        x0_hat = _denormalize_state(x0_hat, art.model)
        for m, v in x0_hat.items():
            x0_hat_batches[m].append(v.detach().cpu())
        
        if return_trajectory and traj_local is not None:
            traj_batches[batch_idx].extend(traj_local)
    
    # Concatenate batches
    x0_hat_final: State = {}
    for m, batch_list in x0_hat_batches.items():
        x0_hat_final[m] = torch.cat(batch_list, dim=1).to(device)
    
    # Format trajectory
    traj_formatted: Optional[Sequence] = None
    if return_trajectory and traj_batches is not None:
        # Convert batch-local lists into per-step full tensors for export.
        num_steps = len(traj_batches[0])
        traj_formatted = tuple(
            {m: torch.cat([traj_batches[b][t][m].cpu() for b in range(num_batches)], dim=1) for m in modality_dims}
            for t in range(num_steps)
        )
    
    return x0_hat_final, traj_formatted, None


def run(cfg) -> None:
    # 1) Build runtime artifacts and load normalization stats used for output conversion.
    art = build_sampling(cfg)

    device = art.device
    seed = int(getattr(cfg.run, "seed", 0))
    torch.manual_seed(seed)

    data_source = build_data_source(cfg, device=device)
    stats = data_source.get_normalization_stats()
    if stats:
        for mod, (mean, std) in stats.items():
            art.model.set_normalization_stats(mod, mean, std)

    modality_dims = dict(cfg.data.modalities.__dict__)

    total_points = int(_get_attr(cfg, "sample.num_points", 100000))
    batch_size = total_points
    num_points = 1
    steps = int(_get_attr(cfg, "sample.steps", getattr(art.solver.cfg, "steps", 18)))
    return_trajectory = bool(_get_attr(cfg, "sample.return_trajectory", False))
    trajectory_space = str(_get_attr(cfg, "sample.trajectory_space", "denoised"))
    target = _get_attr(cfg, "sample.target", "gaussian")
    noise_mesh = _get_attr(cfg, "sample.noise_mesh", None)
    ascii = bool(_get_attr(cfg, "sample.ascii", False))
    mode = str(_get_attr(cfg, "sample.mode", "joint" if len(modality_dims) > 1 else "single")).lower()
    if mode not in {"joint", "conditional", "single"}:
        raise ValueError("sample.mode must be one of: joint, conditional, single")
    guidance = _build_guidance_cfg(cfg, schedule=art.schedule, mode=mode)

    ckpt_path = _resolve_checkpoint_path(cfg)
    # 2) Load checkpoint weights (with compatibility filter for legacy keys).
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Backward compatibility: old checkpoints may contain removed sigma_embedder.proj params.
    model_state = ckpt.get("model", {})
    model_state.pop("sigma_embedder.proj.weight", None)
    model_state.pop("sigma_embedder.proj.bias", None)
    art.model.load_state_dict(model_state, strict=True)
    art.model.eval()

    observed_mask: Optional[ObservedMask] = _get_attr(cfg, "sample.observed_mask", None)
    observed: Optional[State] = _get_attr(cfg, "sample.observed", None)
    cond_name = _get_attr(cfg, "sample.conditional_name", None)
    cond_names = _get_attr(cfg, "sample.conditional_names", None)
    cond_names_dict = getattr(cond_names, "__dict__", None) if cond_names is not None else None
    if cond_names_dict is None and isinstance(cond_names, dict):
        cond_names_dict = dict(cond_names)
    if mode == "conditional":
        # 3) Build conditional inputs from config if conditional mode is requested.
        cond_mask, cond_state, cond_name_cfg, cond_names_cfg = _build_conditional_from_cfg(
            cfg,
            batch_size=batch_size,
            num_points=num_points,
            modality_dims=modality_dims,
            model=art.model,
        )
        if cond_mask is not None:
            observed_mask = cond_mask
            observed = cond_state
            if cond_name_cfg is not None:
                cond_name = cond_name_cfg
            if cond_names_cfg is not None:
                cond_names_dict = cond_names_cfg
        if observed_mask is None or observed is None:
            raise ValueError("Conditional sampling requires sample.observed_mask or sample.conditional")

    supervision_mode = str(getattr(getattr(cfg, "run", None), "supervision", "full")).lower()
    descriptor_sampling_enabled = (
        supervision_mode == "landmarks"
        and getattr(art.model, "descriptor_embedder", None) is not None
    )
    sampling_descriptor: Optional[Descriptor] = None
    if descriptor_sampling_enabled:
        # 4) Optional descriptor construction for landmarks supervision.
        model_dtype = next(art.model.parameters()).dtype

        def _descriptor_dim_from_model() -> int:
            emb = getattr(art.model, "descriptor_embedder", None)
            if emb is None:
                raise ValueError("Descriptor embedder is required for landmarks sampling")
            net = getattr(emb, "net", None)
            if net is not None and len(net) > 0:
                first = net[0]
                in_features = getattr(first, "in_features", None)
                if in_features is not None and int(in_features) > 0:
                    return int(in_features)
                weight = getattr(first, "weight", None)
                if weight is not None and weight.ndim == 2:
                    return int(weight.shape[1])
            raise ValueError("Could not infer descriptor input dimension from model checkpoint")

        landmark_cfg = getattr(cfg.data, "landmark_supervision", None)
        descriptor_interpolation = _normalize_descriptor_interpolation(
            getattr(landmark_cfg, "descriptor_interpolation", "barycentric")
        )
        surface_point_mode = str(getattr(landmark_cfg, "surface_point", "projection")).strip().lower()
        if surface_point_mode == "pull" and descriptor_interpolation == "barycentric":
            raise ValueError(
                "descriptor_interpolation='barycentric' is not supported with data.landmark_supervision.surface_point='pull'. "
                "Use nearest_neighbor or knn."
            )
        knn_neighbors = int(getattr(landmark_cfg, "knn_neighbors", getattr(landmark_cfg, "knn_k", 4)))

        if mode == "conditional":
            need_surface_helpers = descriptor_interpolation == "barycentric"
            prev_normalize = getattr(cfg.data, "normalize", None)
            prev_build_surface_helpers = getattr(cfg.data, "_build_surface_helpers", None)
            setattr(cfg.data, "normalize", False)
            setattr(cfg.data, "_build_surface_helpers", bool(need_surface_helpers))
            try:
                data_source = build_data_source(cfg, device=device)
            finally:
                if prev_normalize is None:
                    try:
                        delattr(cfg.data, "normalize")
                    except Exception:
                        pass
                else:
                    setattr(cfg.data, "normalize", prev_normalize)
                if prev_build_surface_helpers is None:
                    try:
                        delattr(cfg.data, "_build_surface_helpers")
                    except Exception:
                        pass
                else:
                    setattr(cfg.data, "_build_surface_helpers", prev_build_surface_helpers)

            descriptor_lookups = _build_descriptor_lookups(data_source)
            if not descriptor_lookups:
                raise ValueError("Landmarks sampling requires precomputed geodesic descriptor lookups")

            sampling_descriptor = _build_conditional_descriptor(
                observed_mask=observed_mask,
                observed=observed,
                model=art.model,
                data_source=data_source,
                lookups=descriptor_lookups,
                descriptor_interpolation=descriptor_interpolation,
                knn_neighbors=knn_neighbors,
                device=device,
                dtype=model_dtype,
            )
        else:
            desc_dim = _descriptor_dim_from_model()
            sampling_descriptor = _build_null_descriptor(
                batch_size=batch_size,
                desc_dim=desc_dim,
                device=device,
                dtype=model_dtype,
            )

    out_dir = _resolve_output_dir(cfg)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Sampling")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  target: {target}")
    print(f"  steps: {steps}")
    print(f"  num_points: {total_points}")
    print(f"  mode: {mode}")
    if guidance.enabled:
        print(f"  guidance: scale={guidance.scale}, mode={guidance.mode}, sigma_max={guidance.sigma_max}")
    print(f"  output_dir: {out_dir}")

    if out_dir:
        log_lines = [
            "=== Sampling Log ===",
            f"checkpoint: {ckpt_path}",
            f"target: {target}",
            f"steps: {steps}",
            f"num_points: {total_points}",
            f"mode: {mode}",
        ]
        if guidance.enabled:
            log_lines.append(
                f"guidance: scale={guidance.scale}, mode={guidance.mode}, sigma_max={guidance.sigma_max}"
            )

        if mode == "conditional":
            cond_mods = sorted(observed_mask.keys()) if observed_mask else []
            log_lines.append(f"conditioned_modalities: {cond_mods}")
            if cond_name is not None:
                log_lines.append(f"conditioned_name: {cond_name}")
            elif cond_names_dict:
                log_lines.append(f"conditioned_names: {cond_names_dict}")
            if observed is not None:
                for m, v in observed.items():
                    v_stats = (float(v.mean().item()), float(v.std().item()))
                    log_lines.append(f"conditioned_{m}_mean_std: {v_stats}")
        log_path = os.path.join(out_dir, "logs.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

    # Attempt sampling with OOM recovery via batching
    num_batches = 1
    x0_hat_final = None
    traj = None
    while x0_hat_final is None:
        try:
            print(f"Attempting sampling with num_batches={num_batches}")
            x0_hat_final, traj, _ = _sample_with_batches(
                art=art,
                device=device,
                modality_dims=modality_dims,
                total_points=total_points,
                steps=steps,
                target=target,
                noise_mesh=noise_mesh,
                observed_mask=observed_mask,
                observed=observed,
                sampling_descriptor=sampling_descriptor,
                guidance=guidance,
                num_batches=num_batches,
                return_trajectory=return_trajectory,
                trajectory_space=trajectory_space,
            )
            break
        except RuntimeError as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda out of memory" in err_str:
                print(f"CUDA OOM detected. Retrying with num_batches={num_batches * 2}")
                torch.cuda.empty_cache()
                num_batches *= 2
                x0_hat_final = None
            else:
                raise

    print("Sampled batch:")
    for m, v in x0_hat_final.items():
        print(f"  {m}: shape={tuple(v.shape)}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")

    out_modalities = _select_output_modalities(modality_dims, cfg, mode=mode)
    if out_dir:
        # 5) Export final outputs and (optionally) trajectory frames.
        sample_cfg = getattr(cfg, "sample", None)
        texture_raw = getattr(sample_cfg, "texture", None) if sample_cfg is not None else None
        textures_raw = getattr(sample_cfg, "textures", None) if sample_cfg is not None else None
        if texture_raw is None and textures_raw is None:
            # SMALR stores RGB in point channels; default to writing textured output.
            texture_mode = str(_get_attr(cfg, "data.type", "")).lower() == "smalr"
        else:
            texture_mode = bool(bool(texture_raw) or bool(textures_raw))
        swap_cfg = getattr(sample_cfg, "swap_texture_colors", None) if sample_cfg is not None else None
        if isinstance(swap_cfg, bool):
            swap_enabled = bool(swap_cfg)
            swap_mod_a = "A"
            swap_mod_b = "B"
        else:
            swap_enabled = bool(getattr(swap_cfg, "enabled", False)) if swap_cfg is not None else False
            swap_mod_a = str(getattr(swap_cfg, "modality_a", "A")) if swap_cfg is not None else "A"
            swap_mod_b = str(getattr(swap_cfg, "modality_b", "B")) if swap_cfg is not None else "B"
        if swap_enabled and not texture_mode:
            raise ValueError("sample.swap_texture_colors requires texture output mode")
        if swap_enabled:
            x0_hat_export = _swap_texture_rgb_in_state(
                x0_hat_final,
                modality_a=swap_mod_a,
                modality_b=swap_mod_b,
            )
        else:
            x0_hat_export = x0_hat_final
        times_mode = bool(_get_attr(cfg, "data.times", False))
        color_cfg = _get_attr(cfg, "sample.colorize_joint", None)
        color_enabled = bool(_get_attr(color_cfg, "enabled", False)) if color_cfg is not None else False
        color_source = str(_get_attr(color_cfg, "source_modality", "A")) if color_cfg is not None else "A"
        color_map = None
        if color_enabled and mode == "joint":
            if color_source not in x0_hat_export:
                raise KeyError(f"Unknown color source modality '{color_source}'")
            src = x0_hat_export[color_source].detach().cpu().numpy().astype(np.float32)
            if src.shape[-1] < 3:
                raise ValueError("colorize_joint requires at least 3D points")
            color_map = _compute_xyz_colors(src.reshape(-1, src.shape[-1]))

        epoch_tag = _get_ckpt_epoch_tag(ckpt_path)
        cond_tag = ""
        if mode == "conditional" and observed_mask:
            cond_mods = sorted([m for m, is_obs in observed_mask.items() if is_obs])
            if cond_mods:
                name_parts = []
                for m in cond_mods:
                    label = None
                    if cond_name is not None:
                        label = cond_name
                    elif cond_names_dict and m in cond_names_dict:
                        label = cond_names_dict[m]
                    if label:
                        name_parts.append(f"{m}-{_sanitize_tag(str(label))}")
                    else:
                        name_parts.append(m)
                cond_tag = f"_cond-{'-'.join(name_parts)}"
        elif mode == "conditional":
            cond_tag = "_cond"
        guidance_tag = ""
        if guidance.enabled:
            g_scale = str(guidance.scale).replace(".", "p")
            guidance_tag = f"_guidance-{g_scale}"
        swap_tag = f"_swap-{_sanitize_tag(swap_mod_a)}-{_sanitize_tag(swap_mod_b)}" if swap_enabled else ""
        color_tag = "_colored" if color_map is not None else ""
        base_name = f"e{epoch_tag}_n{total_points}{cond_tag}{guidance_tag}{swap_tag}{color_tag}"
        name_suffix = _get_attr(cfg, "sample.name_suffix", None)
        if name_suffix:
            base_name = f"{base_name}_{_sanitize_tag(str(name_suffix))}"
        for m in out_modalities:
            name = f"shape-{m}_{base_name}"
            out_path = _write_point_clouds(
                x0_hat_export[m],
                out_dir,
                name,
                ascii=ascii,
                colors=None if texture_mode else color_map,
                correspondence_colors=color_map if texture_mode else None,
                texture_from_points=texture_mode,
                time_from_points=times_mode,
            )
            print(f"Saved: {out_path}")

        if return_trajectory and traj is not None:
            for idx, x_i in enumerate(traj):
                if str(trajectory_space).lower().strip() == "denoised":
                    x_i_out = _denormalize_state(x_i, art.model)
                else:
                    # Keep legacy_scaled_latent in solver space to match original GeomDist exports.
                    x_i_out = x_i

                if swap_enabled:
                    x_i_export = _swap_texture_rgb_in_state(
                        x_i_out,
                        modality_a=swap_mod_a,
                        modality_b=swap_mod_b,
                    )
                else:
                    x_i_export = x_i_out

                step_color_map = color_map
                if color_enabled and mode == "joint":
                    if color_source not in x_i_export:
                        raise KeyError(f"Unknown color source modality '{color_source}'")
                    src_i = x_i_export[color_source].detach().cpu().numpy().astype(np.float32)
                    if src_i.shape[-1] < 3:
                        raise ValueError("colorize_joint requires at least 3D points")
                    step_color_map = _compute_xyz_colors(src_i.reshape(-1, src_i.shape[-1]))

                for m in out_modalities:
                    out_path = _write_point_clouds(
                        x_i_export[m],
                        out_dir,
                        f"{base_name}_{m}",
                        ascii=ascii,
                        colors=None if texture_mode else step_color_map,
                        correspondence_colors=step_color_map if texture_mode else None,
                        texture_from_points=texture_mode,
                        time_from_points=times_mode,
                        suffix=f"i{idx:03d}",
                    )
                    print(f"Saved: {out_path}")

    if return_trajectory:
        print(f"Trajectory length: {len(traj)}")
