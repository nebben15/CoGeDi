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
from cogedi.orch.sample import _compute_xyz_colors, _write_point_clouds
from cogedi.data.base import build_data_source


def _get_attr(obj, path: str, default):
    cur = obj
    for key in path.split("."):
        cur = getattr(cur, key, None)
        if cur is None:
            return default
    return cur


def _resolve_checkpoint_path(cfg) -> str:
    """Resolve checkpoint from config name/path + environment-specific base directory."""
    inverse_cfg = getattr(cfg, "inverse", None)
    ckpt = getattr(inverse_cfg, "checkpoint", None) if inverse_cfg is not None else None
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
            raise ValueError("paths.checkpoints is required when inverse.checkpoint is a name")
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
        raise ValueError("Missing inverse.checkpoint and paths.checkpoints; cannot resolve checkpoint")
    if exp_name:
        ckpt_base = os.path.join(ckpt_base, exp_name)

    raise ValueError("inverse.checkpoint is required to select the checkpoint name")


def _load_single_cloud(path: str) -> torch.Tensor:
    """Load a single point cloud from PLY or OBJ file."""
    if path.lower().endswith('.ply'):
        pcd = o3d.io.read_point_cloud(path)
        pts = torch.from_numpy(np.asarray(pcd.points)).float()
    elif path.lower().endswith('.obj'):
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Trimesh):
            pts = torch.from_numpy(mesh.vertices).float()
        else:
            raise ValueError(f"Cannot extract vertices from {path}")
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return pts


def _load_input_state(
    input_config,
    modality_dims: Dict[str, int],
    device: torch.device,
) -> State:
    """Load point clouds for inverse sampling.

    Preferred format is a dict that maps each modality name to a file path:

        inverse:
          input_per_modality:
            A: /path/to/a.ply
            B: /path/to/b.ply

    A single path is still accepted as a fallback and is shared across modalities.
    """
    state: State = {}

    if isinstance(input_config, dict) or hasattr(input_config, "__dict__"):
        config_dict = input_config if isinstance(input_config, dict) else dict(getattr(input_config, "__dict__", {}))
        unknown = [key for key in config_dict.keys() if key not in modality_dims]
        if unknown:
            raise ValueError(f"inverse.input_per_modality contains unknown modalities: {unknown}")

        print("Loading per-modality input files...")
        point_count: Optional[int] = None
        for modality, path in config_dict.items():
            pts = _load_single_cloud(str(path))
            dim = modality_dims[modality]
            if pts.shape[1] < dim:
                raise ValueError(
                    f"Cloud {path} has {pts.shape[1]} channels but modality '{modality}' expects {dim}"
                )

            pts_mod = pts[:, :dim]
            if point_count is None:
                point_count = int(pts_mod.shape[0])
            elif int(pts_mod.shape[0]) != point_count:
                raise ValueError(
                    "All inverse input point clouds must have the same number of points "
                    "so correspondence colors can be shared across modalities"
                )

            # In this model, each point is treated as a batch item and the
            # embedder expects N=1, so convert [N, D] -> [N, 1, D].
            state[modality] = pts_mod.unsqueeze(1).to(device)
            print(f"  Loaded '{modality}' from {path}: {tuple(state[modality].shape)}")

        return state

    input_str = str(input_config)
    print(f"Loading shared inverse input from {input_str}")
    cloud = _load_single_cloud(input_str)

    for modality, dim in modality_dims.items():
        if cloud.shape[1] < dim:
            raise ValueError(
                f"Point cloud has {cloud.shape[1]} features but modality '{modality}' expects {dim}"
            )
        state[modality] = cloud[:, :dim].unsqueeze(1).to(device)
        print(f"  Modality '{modality}': {tuple(state[modality].shape)}")

    return state


def _denormalize_state(state: State, model) -> State:
    """Denormalize every modality in a state dict if stats are available."""
    out: State = {}
    for modality, tensor in state.items():
        try:
            out[modality] = model.denormalize(tensor, modality)
        except RuntimeError:
            out[modality] = tensor
    return out


def _inverse_with_batches(
    *,
    art,
    device: torch.device,
    input_state: State,
    steps: int,
    return_trajectory: bool = False,
    trajectory_space: str = "denoised",
) -> tuple[State, Optional[Sequence]]:
    """Invert the full multimodal state jointly while keeping batch items independent."""
    clamp_ctx, denoise_fn = build_denoise(
        model=art.model,
        conditioning=art.conditioning,
        observed_mask=None,
        observed=None,
        guidance=GuidanceConfig(enabled=False, scale=0.0, mode="joint", sigma_max=80.0),
        descriptor=None,
    )

    final_state: State = {}
    trajectories_by_modality: Dict[str, list[torch.Tensor]] = {}

    print(f"Running inverse diffusion for {steps} steps")
    print("Purpose: encode clean data -> noise space")
    traj_mode = str(trajectory_space).lower().strip()
    if traj_mode not in {"denoised", "latent"}:
        raise ValueError("inverse.trajectory_space must be one of: denoised, latent")

    x_normalized: State = {}
    batch_size: Optional[int] = None
    for modality, pts in input_state.items():
        if pts.ndim != 3 or pts.shape[1] != 1:
            raise ValueError(
                f"inverse inputs must have shape [B, 1, D] per modality, got {tuple(pts.shape)} for '{modality}'"
            )
        if batch_size is None:
            batch_size = int(pts.shape[0])
        elif int(pts.shape[0]) != batch_size:
            raise ValueError("All inverse modalities must have the same batch size")

        print(f"Inverting modality '{modality}'")
        try:
            x_normalized[modality] = art.model.normalize(pts, modality)
            print(f"  Normalized modality '{modality}'")
        except RuntimeError as exc:
            print(f"  Warning: could not normalize '{modality}': {exc}. Using raw input.")
            x_normalized[modality] = pts

    assert batch_size is not None
    forward_sigmas = art.schedule.sampling_sigmas(steps, batch_size, observed_mask=None)
    reverse_sigmas = list(reversed(forward_sigmas))

    x = x_normalized
    traj_local: Optional[list[State]] = [
        {modality: tensor.detach().clone() for modality, tensor in x.items()}
    ] if return_trajectory else None

    with torch.no_grad():
        for sigma_cur, sigma_next in zip(reverse_sigmas[:-1], reverse_sigmas[1:]):
            x = art.solver.step(
                x=x,
                sigma=sigma_cur,
                sigma_next=sigma_next,
                denoise_fn=denoise_fn,
                observed_mask=None,
            )
            x = art.conditioning.clamp(x, clamp_ctx)
            if traj_local is not None:
                if traj_mode == "denoised":
                    x_vis = denoise_fn(x, sigma_next, None)
                    traj_local.append({modality: tensor.detach().clone() for modality, tensor in x_vis.items()})
                else:
                    traj_local.append({modality: tensor.detach().clone() for modality, tensor in x.items()})

    final_state = x
    if return_trajectory and traj_local is not None:
        for modality in input_state.keys():
            trajectories_by_modality[modality] = [step_state[modality] for step_state in traj_local]

    traj_formatted: Optional[Sequence] = None
    if return_trajectory and trajectories_by_modality:
        modalities = list(input_state.keys())
        num_steps = len(next(iter(trajectories_by_modality.values())))
        traj_formatted = tuple(
            {mod: trajectories_by_modality[mod][step_idx] for mod in modalities}
            for step_idx in range(num_steps)
        )

    return final_state, traj_formatted


def _save_latent_state(x: State, out_dir: str, base_name: str) -> Dict[str, str]:
    """Save the final inverse latents as per-modality NPZ files."""
    os.makedirs(out_dir, exist_ok=True)
    output_paths: Dict[str, str] = {}
    for modality, tensor in x.items():
        out_path = os.path.join(out_dir, f"{base_name}_{modality}.npz")
        np.savez_compressed(out_path, latent=tensor.detach().cpu().numpy())
        output_paths[modality] = out_path
        print(f"Saved latent for '{modality}' to {out_path}")
    return output_paths


def _save_trajectory(
    traj: Sequence[State],
    out_dir: str,
    base_name: str,
    *,
    model,
    color_source_modality: str,
    ascii: bool,
    denormalize_trajectory: bool,
) -> Dict[str, str]:
    """Save trajectory as colored point clouds and NPZ latents."""
    os.makedirs(out_dir, exist_ok=True)
    output_paths: Dict[str, str] = {}

    for step_idx, step_state in enumerate(traj):
        step_dir = os.path.join(out_dir, f"step_{step_idx:03d}")
        os.makedirs(step_dir, exist_ok=True)

        step_state_out = _denormalize_state(step_state, model) if denormalize_trajectory else step_state
        if color_source_modality not in step_state_out:
            raise KeyError(f"Unknown color source modality '{color_source_modality}'")

        ref = step_state_out[color_source_modality].detach().cpu().numpy().astype(np.float32)
        if ref.shape[-1] < 3:
            raise ValueError("Correspondence coloring requires at least 3D geometry")
        color_map = _compute_xyz_colors(ref.reshape(-1, ref.shape[-1]))

        for modality, tensor in step_state.items():
            npz_path = os.path.join(step_dir, f"{base_name}_{modality}.npz")
            np.savez_compressed(npz_path, latent=tensor.detach().cpu().numpy())
            output_paths[f"step_{step_idx}_{modality}_npz"] = npz_path

            ply_path = _write_point_clouds(
                step_state_out[modality],
                step_dir,
                f"{base_name}_{modality}",
                ascii=ascii,
                colors=color_map,
                correspondence_colors=None,
                texture_from_points=False,
                time_from_points=False,
                suffix=f"i{step_idx:03d}",
            )
            output_paths[f"step_{step_idx}_{modality}_ply"] = ply_path

    print(f"Saved trajectory with {len(traj)} steps to {out_dir}")
    return output_paths


def run(cfg) -> None:
    """Run inverse sampling given normalized input."""
    # 1) Build runtime artifacts and load normalization stats
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

    # 2) Load input point cloud(s)
    # Support both inverse.input (single/directory) and inverse.input_per_modality (dict)
    input_per_modality = _get_attr(cfg, "inverse.input_per_modality", None)
    input_single = _get_attr(cfg, "inverse.input", None)
    
    if not input_per_modality and not input_single:
        raise ValueError("Either inverse.input or inverse.input_per_modality is required")
    
    input_config = input_per_modality if input_per_modality else input_single

    input_state = _load_input_state(
        input_config,
        modality_dims,
        device=device,
    )
    batch_size = input_state[next(iter(input_state.keys()))].shape[0]

    # 3) Setup inverse parameters
    steps = int(_get_attr(cfg, "inverse.steps", getattr(art.solver.cfg, "steps", 18)))
    return_trajectory = bool(_get_attr(cfg, "inverse.return_trajectory", False))
    trajectory_space = str(_get_attr(cfg, "inverse.trajectory_space", "denoised"))
    color_source_modality = str(
        _get_attr(cfg, "inverse.color_source_modality", next(iter(modality_dims.keys())))
    )
    ascii = bool(_get_attr(cfg, "inverse.ascii", False))

    # 4) Load checkpoint
    ckpt_path = _resolve_checkpoint_path(cfg)
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_state = ckpt.get("model", {})
    model_state.pop("sigma_embedder.proj.weight", None)
    model_state.pop("sigma_embedder.proj.bias", None)
    art.model.load_state_dict(model_state, strict=True)
    art.model.eval()
    print("Model loaded and set to eval mode")

    # 5) Run inverse sampling
    print(f"\n=== INVERSE SAMPLING ===")
    input_desc = "per-modality files" if input_per_modality else f"file/directory: {input_single}"
    print(f"Input: {input_desc}")
    print(f"Color source modality: {color_source_modality}")
    print(f"Steps: {steps}")
    print(f"Return trajectory: {return_trajectory}")
    print(f"Trajectory space: {trajectory_space}")

    latent_final, traj = _inverse_with_batches(
        art=art,
        device=device,
        input_state=input_state,
        steps=steps,
        return_trajectory=return_trajectory,
        trajectory_space=trajectory_space,
    )

    # 6) Save outputs
    out_dir = _get_attr(cfg, "inverse.output_dir", None)
    if out_dir is None:
        out_base = getattr(cfg.run, "experiment_name", "inverse_output")
        paths_cfg = getattr(cfg, "paths", None)
        samples_base = getattr(paths_cfg, "samples", None) if paths_cfg is not None else None
        if samples_base:
            out_dir = os.path.join(samples_base, out_base)
        else:
            out_dir = os.path.join("./outputs", out_base)

    os.makedirs(out_dir, exist_ok=True)
    base_name = "latent"

    # Save final latent state and colorized denormalized point clouds.
    output_paths = _save_latent_state(latent_final, out_dir, base_name)
    print(f"\nLatent output paths: {output_paths}")

    denorm_final = _denormalize_state(latent_final, art.model)
    if color_source_modality not in denorm_final:
        raise KeyError(f"Unknown color source modality '{color_source_modality}'")
    ref_final = denorm_final[color_source_modality].detach().cpu().numpy().astype(np.float32)
    if ref_final.shape[-1] < 3:
        raise ValueError("Correspondence coloring requires at least 3D geometry")
    final_color_map = _compute_xyz_colors(ref_final.reshape(-1, ref_final.shape[-1]))

    for modality, tensor in denorm_final.items():
        out_path = _write_point_clouds(
            tensor,
            out_dir,
            f"shape-{modality}_{base_name}",
            ascii=ascii,
            colors=final_color_map,
            correspondence_colors=None,
            texture_from_points=False,
            time_from_points=False,
        )
        print(f"Saved colored final output: {out_path}")

    # Save trajectory if requested
    if return_trajectory and traj is not None:
        traj_paths = _save_trajectory(
            traj,
            os.path.join(out_dir, "trajectory"),
            base_name,
            model=art.model,
            color_source_modality=color_source_modality,
            ascii=ascii,
            denormalize_trajectory=str(trajectory_space).lower().strip() == "denoised",
        )
        print(f"Trajectory paths: {traj_paths}")

    print(f"\n=== INVERSE SAMPLING COMPLETE ===")
    print(f"Output directory: {out_dir}")
