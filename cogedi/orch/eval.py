from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from cogedi.build import build_denoise, build_eval
from cogedi.data.base import build_data_source
from cogedi.dtypes import ObservedMask, State
from cogedi.orch.denoise import GuidanceConfig
from cogedi.metrics.geometry import (
	chamfer_distance_l2,
	earth_movers_distance,
	f_score,
	hausdorff_distance,
	point_to_surface_distance,
)
from cogedi.metrics.correspondence import correspondence_metrics


def _get_attr(obj, path: str, default):
	cur = obj
	for key in path.split("."):
		cur = getattr(cur, key, None)
		if cur is None:
			return default
	return cur


def _resolve_checkpoint_path(cfg) -> str:
	eval_cfg = getattr(cfg, "eval", None)
	ckpt = getattr(eval_cfg, "checkpoint", None) if eval_cfg is not None else None
	if not ckpt:
		raise ValueError("eval.checkpoint is required")

	if os.path.isabs(ckpt):
		return ckpt

	paths_cfg = getattr(cfg, "paths", None)
	ckpt_base = getattr(paths_cfg, "checkpoints", None) if paths_cfg is not None else None
	exp_name = getattr(cfg.run, "experiment_name", None)
	if not ckpt_base:
		raise ValueError("paths.checkpoints is required when eval.checkpoint is a name")
	if exp_name:
		ckpt_base = os.path.join(ckpt_base, exp_name)

	ckpt_name = ckpt if ckpt.endswith(".pth") else f"{ckpt}.pth"
	return os.path.join(ckpt_base, ckpt_name)


def _resolve_output_dir(cfg) -> Optional[str]:
	paths_cfg = getattr(cfg, "paths", None)
	eval_base = getattr(paths_cfg, "eval", None) if paths_cfg is not None else None
	# If eval_base is a selector structure represented as a SimpleNamespace or dict,
	# resolve it using the run env if available (this makes eval robust to configs
	# that were not fully resolved earlier).
	selector_keys = {"local", "slurm", "default"}
	if eval_base is not None and (isinstance(eval_base, dict) or hasattr(eval_base, "__dict__")):
		if isinstance(eval_base, dict):
			obj = eval_base
		else:
			obj = getattr(eval_base, "__dict__", {})
		if obj and set(obj.keys()).issubset(selector_keys):
			env = getattr(cfg.run, "env", None) or getattr(cfg.run, "environment", None) or "local"
			env = str(env).lower()
			if env in obj:
				eval_base = obj[env]
			elif "default" in obj:
				eval_base = obj["default"]
			else:
				eval_base = None

	exp_name = getattr(cfg.run, "experiment_name", None)
	if eval_base and exp_name:
		return os.path.join(eval_base, exp_name)
	return eval_base


def _denormalize_state(state: State, model) -> State:
	out: State = {}
	for m, v in state.items():
		try:
			out[m] = model.denormalize(v, m)
		except RuntimeError:
			out[m] = v
	return out


def _parse_fscore_thresholds(geometry_cfg) -> list[float]:
	raw_multi = _get_attr(geometry_cfg, "fscore_thresholds", None)
	if raw_multi is None:
		raw_single = _get_attr(geometry_cfg, "fscore_threshold", 0.01)
		vals = [float(raw_single)]
	elif isinstance(raw_multi, (list, tuple)):
		vals = [float(v) for v in raw_multi]
	else:
		vals = [float(raw_multi)]

	if len(vals) == 0:
		raise ValueError("eval.geometry.fscore_thresholds must contain at least one value")

	clean: list[float] = []
	for v in vals:
		if v < 0:
			raise ValueError("F-score thresholds must be >= 0")
		if v not in clean:
			clean.append(v)
	return clean


def _mean_finite(values: list[float]) -> float:
	if len(values) == 0:
		return float("nan")
	a = np.asarray(values, dtype=np.float64)
	finite = np.isfinite(a)
	if not np.any(finite):
		return float("nan")
	return float(a[finite].mean())


def _sanitize_tag(value: str) -> str:
	clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
	clean = re.sub(r"-+", "-", clean).strip("-._")
	return clean or "tag"


def _format_report_value(value: float) -> str:
	value = float(value)
	return f"{value:.3e}"


@contextmanager
def _temporary_numpy_seed(seed: Optional[int]):
	if seed is None:
		yield
		return
	state = np.random.get_state()
	try:
		np.random.seed(int(seed))
		yield
	finally:
		np.random.set_state(state)


def _build_guidance_cfg(cfg, *, schedule, mode: str, section: str) -> GuidanceConfig:
	eval_cfg = getattr(cfg, "eval", None)
	section_cfg = getattr(eval_cfg, section, None) if eval_cfg is not None else None
	g_cfg = getattr(section_cfg, "guidance", None) if section_cfg is not None else None
	if g_cfg is None:
		g_cfg = getattr(eval_cfg, "guidance", None) if eval_cfg is not None else None

	if g_cfg is None:
		return GuidanceConfig(scale=0.0, mode=mode, sigma_max=float(schedule.cfg.sigma_max), enabled=False)

	scale = float(getattr(g_cfg, "scale", 0.0))
	enabled = bool(getattr(g_cfg, "enabled", True)) and scale != 0.0
	sigma_max = float(getattr(g_cfg, "sigma_max", getattr(schedule.cfg, "sigma_max", 1.0)))
	g_mode = str(getattr(g_cfg, "mode", mode)).lower()
	return GuidanceConfig(scale=scale, mode=g_mode, sigma_max=sigma_max, enabled=enabled)


def _state_to_points(state: State, modality: str) -> np.ndarray:
	x = state[modality].detach().cpu().numpy().astype(np.float32)
	return x.reshape(-1, x.shape[-1])


def _project_to_mesh_barycentric(mesh: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	if points.shape[-1] != 3:
		raise ValueError("barycentric projection requires 3D points")
	closest, _, face_idx = trimesh.proximity.closest_point(mesh, points)
	faces = mesh.faces[face_idx]
	tri = mesh.vertices[faces]
	bary = trimesh.triangles.points_to_barycentric(tri, closest)
	return face_idx, bary


def _barycentric_points(mesh: trimesh.Trimesh, face_idx: np.ndarray, bary: np.ndarray) -> np.ndarray:
	faces = mesh.faces[face_idx]
	tri = mesh.vertices[faces]
	return (
		bary[:, [0]] * tri[:, 0]
		+ bary[:, [1]] * tri[:, 1]
		+ bary[:, [2]] * tri[:, 2]
	)


def _sample_model_points(
	art,
	*,
	modality_dims: Dict[str, int],
	num_points: int,
	steps: int,
	target: str,
	noise_mesh: Optional[str],
	observed_mask: Optional[ObservedMask],
	observed: Optional[State],
	guidance: GuidanceConfig,
	progress_desc: Optional[str],
	num_batches: int = 1,
) -> State:
	if num_batches < 1:
		raise ValueError("num_batches must be >= 1")

	num_points_per = 1
	points_per_batch = max(1, num_points // num_batches)
	remainder = num_points % num_batches
	batch_sizes = [points_per_batch + (1 if i < remainder else 0) for i in range(num_batches)]
	batch_sizes = [s for s in batch_sizes if s > 0]

	x0_hat_batches: Dict[str, list[torch.Tensor]] = {m: [] for m in modality_dims}
	offset = 0

	for batch_idx, batch_size in enumerate(batch_sizes):
		if num_batches > 1:
			print(f"  Eval sampling batch {batch_idx + 1}/{num_batches} ({batch_size} points)")

		sigmas = art.schedule.sampling_sigmas(steps, batch_size, observed_mask=observed_mask)

		def _init_noise(dim: int) -> torch.Tensor:
			if target == "gaussian":
				return torch.randn(batch_size, num_points_per, dim, device=art.device)
			if target == "uniform":
				return (torch.rand(batch_size, num_points_per, dim, device=art.device) - 0.5) / np.sqrt(1 / 12)
			if target == "sphere":
				n = torch.randn(batch_size, num_points_per, dim, device=art.device)
				n = torch.nn.functional.normalize(n, dim=-1)
				return n / np.sqrt(1 / 3)
			if target == "mesh":
				if noise_mesh is None:
					raise ValueError("eval.noise_mesh is required when target='mesh'")
				if dim != 3:
					raise ValueError("mesh target only supports 3D geometry")
				mesh = trimesh.load(noise_mesh)
				pts, _ = trimesh.sample.sample_surface(mesh, batch_size * num_points_per)
				pts = torch.from_numpy(pts).float().to(art.device)
				return pts.view(batch_size, num_points_per, 3)
			raise ValueError("eval.target must be one of: gaussian, uniform, sphere, mesh")

		x_init: State = {m: _init_noise(dim) for m, dim in modality_dims.items()}
		if observed is not None:
			for m, v in observed.items():
				if m in x_init:
					v_dev = v.to(art.device)
					if v_dev.shape[0] == batch_size:
						x_init[m] = v_dev
					elif v_dev.shape[0] == num_points:
						x_init[m] = v_dev[offset:offset + batch_size]
					elif v_dev.shape[0] == 1:
						x_init[m] = v_dev.repeat(batch_size, 1, 1)
					else:
						raise ValueError(
							f"Observed batch for modality '{m}' has incompatible shape {tuple(v_dev.shape)} "
							f"for num_points={num_points}, batch_size={batch_size}"
						)

		sigma0 = sigmas[0]
		for m in x_init.keys():
			if observed_mask and observed_mask.get(m, False):
				continue
			s0 = sigma0[m]
			while s0.ndim < x_init[m].ndim:
				s0 = s0.unsqueeze(-1)
			x_init[m] = x_init[m] * s0

		ctx, denoise_fn = build_denoise(
			model=art.model,
			conditioning=art.conditioning,
			observed_mask=observed_mask,
			observed=observed,
			guidance=guidance,
		)

		x = x_init
		iter_steps = range(len(sigmas) - 1)
		if progress_desc:
			if num_batches > 1:
				desc = f"{progress_desc} [{batch_idx + 1}/{num_batches}]"
			else:
				desc = progress_desc
			iter_steps = tqdm(iter_steps, desc=desc, ncols=100)
		for i in iter_steps:
			x = art.solver.step(
				x=x,
				sigma=sigmas[i],
				sigma_next=sigmas[i + 1],
				denoise_fn=denoise_fn,
				observed_mask=observed_mask,
			)
			x = art.conditioning.clamp(x, ctx)

		x0_hat_batch = denoise_fn(x, sigmas[-1], observed_mask)
		for m, v in x0_hat_batch.items():
			x0_hat_batches[m].append(v)
		offset += batch_size

	x0_hat_final: State = {}
	for m, batch_list in x0_hat_batches.items():
		x0_hat_final[m] = torch.cat(batch_list, dim=0)
	return x0_hat_final


def _sample_model_points_with_oom_recovery(
	art,
	*,
	modality_dims: Dict[str, int],
	num_points: int,
	steps: int,
	target: str,
	noise_mesh: Optional[str],
	observed_mask: Optional[ObservedMask],
	observed: Optional[State],
	guidance: GuidanceConfig,
	progress_desc: Optional[str],
) -> State:
	num_batches = 1
	while True:
		try:
			if num_batches > 1:
				print(f"Retrying eval sampling with num_batches={num_batches}")
			return _sample_model_points(
				art,
				modality_dims=modality_dims,
				num_points=num_points,
				steps=steps,
				target=target,
				noise_mesh=noise_mesh,
				observed_mask=observed_mask,
				observed=observed,
				guidance=guidance,
				progress_desc=progress_desc,
				num_batches=num_batches,
			)
		except RuntimeError as e:
			err_str = str(e).lower()
			if "out of memory" not in err_str and "cuda out of memory" not in err_str:
				raise
			if num_batches >= num_points:
				raise
			next_batches = min(num_points, num_batches * 2)
			print(f"CUDA OOM detected during eval. Increasing num_batches from {num_batches} to {next_batches}")
			torch.cuda.empty_cache()
			num_batches = next_batches


def _build_correspondence_conditional(
	cfg,
	*,
	model,
	batch_size: int,
	modality_dims: Dict[str, int],
) -> tuple[ObservedMask, State, str, np.ndarray]:
	eval_cfg = getattr(cfg, "eval", None)
	corr_cfg = getattr(eval_cfg, "correspondence", None) if eval_cfg is not None else None
	if corr_cfg is None:
		raise ValueError("eval.correspondence is required when correspondence is enabled")

	observed = getattr(corr_cfg, "observed", None)
	target = getattr(corr_cfg, "target", None)

	obs_mod = getattr(corr_cfg, "observed_modality", None)
	obs_point = getattr(corr_cfg, "observed_point", None)
	tgt_mod = getattr(corr_cfg, "target_modality", None)
	tgt_point = getattr(corr_cfg, "target_point", None)

	if observed is None and obs_mod is not None and obs_point is not None:
		observed = {str(obs_mod): obs_point}
	if target is None and tgt_mod is not None and tgt_point is not None:
		target = {str(tgt_mod): tgt_point}

	if not isinstance(observed, dict) or not isinstance(target, dict):
		raise ValueError("eval.correspondence requires observed/target points")

	obs_mods = list(observed.keys())
	tgt_mods = list(target.keys())
	if len(obs_mods) != 1 or len(tgt_mods) != 1:
		raise ValueError("eval.correspondence observed/target must each contain exactly one modality")

	obs_mod = obs_mods[0]
	tgt_mod = tgt_mods[0]
	if obs_mod not in modality_dims or tgt_mod not in modality_dims:
		raise KeyError("Unknown modality in eval.correspondence")

	obs_point = np.asarray(observed[obs_mod], dtype=np.float32)
	tgt_point = np.asarray(target[tgt_mod], dtype=np.float32)
	if obs_point.shape != (modality_dims[obs_mod],):
		raise ValueError("Observed point shape mismatch for correspondence")
	if tgt_point.shape != (modality_dims[tgt_mod],):
		raise ValueError("Target point shape mismatch for correspondence")

	conditional_normalized = bool(getattr(corr_cfg, "conditional_normalized", False))
	obs_tensor = torch.as_tensor(obs_point, dtype=torch.float32, device=next(model.parameters()).device)
	if not conditional_normalized:
		obs_tensor = model.normalize(obs_tensor, obs_mod)

	obs_tensor = obs_tensor.view(1, 1, -1).repeat(batch_size, 1, 1)
	observed_mask: ObservedMask = {obs_mod: True}
	observed_state: State = {obs_mod: obs_tensor}

	return observed_mask, observed_state, tgt_mod, tgt_point


def _build_correspondence_from_points(
	*,
	model,
	modality_dims: Dict[str, int],
	batch_size: int,
	observed_mod: str,
	observed_point: np.ndarray,
	target_mod: str,
	target_point: np.ndarray,
	conditional_normalized: bool,
) -> tuple[ObservedMask, State, str, np.ndarray]:
	if observed_mod not in modality_dims or target_mod not in modality_dims:
		raise KeyError("Unknown modality in correspondence points")

	obs_point = np.asarray(observed_point, dtype=np.float32)
	tgt_point = np.asarray(target_point, dtype=np.float32)
	if obs_point.shape != (modality_dims[observed_mod],):
		raise ValueError("Observed point shape mismatch for correspondence")
	if tgt_point.shape != (modality_dims[target_mod],):
		raise ValueError("Target point shape mismatch for correspondence")

	obs_tensor = torch.as_tensor(obs_point, dtype=torch.float32, device=next(model.parameters()).device)
	if not conditional_normalized:
		obs_tensor = model.normalize(obs_tensor, observed_mod)

	obs_tensor = obs_tensor.view(1, 1, -1).repeat(batch_size, 1, 1)
	observed_mask: ObservedMask = {observed_mod: True}
	observed_state: State = {observed_mod: obs_tensor}

	return observed_mask, observed_state, target_mod, tgt_point


def _sample_correspondence_pairs(
	data_source,
	*,
	modality_dims: Dict[str, int],
	num_pairs: int,
	seed: Optional[int] = None,
) -> list[tuple[str, np.ndarray, str, np.ndarray]]:
	if not hasattr(data_source, "_coupled_sample"):
		raise ValueError("Data source does not support coupled correspondence sampling")

	with _temporary_numpy_seed(seed):
		coupled = data_source._coupled_sample(num_pairs)
	mods = list(modality_dims.keys())
	if len(mods) < 2:
		raise ValueError("Need at least two modalities for correspondence evaluation")

	obs_mod = mods[0]
	tgt_mod = mods[1]
	pairs = []
	for i in range(num_pairs):
		pairs.append((obs_mod, coupled[obs_mod][i], tgt_mod, coupled[tgt_mod][i]))
	return pairs


def _eval_correspondence_pair(
	*,
	art,
	modality_dims: Dict[str, int],
	observed_mod: str,
	observed_point: np.ndarray,
	target_mod: str,
	target_point: np.ndarray,
	corr_points: int,
	corr_steps: int,
	corr_target: str,
	corr_noise_mesh: Optional[str],
	corr_guidance: GuidanceConfig,
	conditional_normalized: bool,
) -> Dict[str, float]:
	mask, observed, tgt_mod, tgt_point = _build_correspondence_from_points(
		model=art.model,
		modality_dims=modality_dims,
		batch_size=corr_points,
		observed_mod=observed_mod,
		observed_point=observed_point,
		target_mod=target_mod,
		target_point=target_point,
		conditional_normalized=conditional_normalized,
	)

	with torch.no_grad():
			corr_state = _sample_model_points_with_oom_recovery(
			art,
			modality_dims=modality_dims,
			num_points=corr_points,
			steps=corr_steps,
			target=corr_target,
			noise_mesh=corr_noise_mesh,
			observed_mask=mask,
			observed=observed,
			guidance=corr_guidance,
				progress_desc=None,
		)
	corr_state = _denormalize_state(corr_state, art.model)
	corr_cloud = _state_to_points(corr_state, tgt_mod)
	return correspondence_metrics(corr_cloud, tgt_point)


def run(cfg) -> None:
	art = build_eval(cfg)
	device = art.device

	data_source = build_data_source(cfg, device=device)
	stats = data_source.get_normalization_stats()
	if stats:
		for mod, (mean, std) in stats.items():
			art.model.set_normalization_stats(mod, mean, std)

	ckpt_path = _resolve_checkpoint_path(cfg)
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	# Backward compatibility: old checkpoints may contain removed sigma_embedder.proj params.
	model_state = ckpt.get("model", {})
	model_state.pop("sigma_embedder.proj.weight", None)
	model_state.pop("sigma_embedder.proj.bias", None)
	art.model.load_state_dict(model_state, strict=True)
	art.model.eval()

	modality_dims = dict(cfg.data.modalities.__dict__)
	eval_cfg = getattr(cfg, "eval", None)

	if eval_cfg is None:
		raise ValueError("eval section is required")

	geometry_cfg = getattr(eval_cfg, "geometry", None)
	if geometry_cfg is None:
		raise ValueError("eval.geometry is required")
	geometry_enabled = bool(getattr(geometry_cfg, "enabled", True))

	num_points = int(_get_attr(geometry_cfg, "num_points", 10000))
	steps = int(_get_attr(geometry_cfg, "steps", _get_attr(eval_cfg, "steps", 18)))
	target = str(_get_attr(geometry_cfg, "target", _get_attr(eval_cfg, "target", "gaussian"))).lower()
	noise_mesh = _get_attr(geometry_cfg, "noise_mesh", None)
	fscore_thresholds = _parse_fscore_thresholds(geometry_cfg)
	fscore_thresh = float(fscore_thresholds[0])
	emd_max = int(_get_attr(geometry_cfg, "emd_max_points", 2048))
	ref_samples = int(_get_attr(geometry_cfg, "ref_samples", num_points))
	mode = str(_get_attr(geometry_cfg, "mode", "joint" if len(modality_dims) > 1 else "single")).lower()
	if mode not in {"joint", "conditional", "single"}:
		raise ValueError("eval.mode must be one of: joint, conditional, single")
	guidance = _build_guidance_cfg(cfg, schedule=art.schedule, mode=mode, section="geometry")

	observed_mask: Optional[ObservedMask] = _get_attr(geometry_cfg, "observed_mask", None)
	observed: Optional[State] = _get_attr(geometry_cfg, "observed", None)

	correspondence_cfg = getattr(eval_cfg, "correspondence", None)
	correspondence_enabled = bool(getattr(correspondence_cfg, "enabled", False)) if correspondence_cfg is not None else False

	joint_cfg = getattr(eval_cfg, "joint_correspondence", None)
	joint_enabled = bool(getattr(joint_cfg, "enabled", False)) if joint_cfg is not None else False

	out_dir = _resolve_output_dir(cfg)
	if out_dir:
		Path(out_dir).mkdir(parents=True, exist_ok=True)

	lines = []
	results: Dict[str, Dict[str, float]] = {}
	joint_metrics: Dict[str, float] = {}

	if geometry_enabled:
		with torch.no_grad():
			pred_state = _sample_model_points_with_oom_recovery(
				art,
				modality_dims=modality_dims,
				num_points=num_points,
				steps=steps,
				target=target,
				noise_mesh=noise_mesh,
				observed_mask=observed_mask,
				observed=observed,
				guidance=guidance,
				progress_desc="Evaluating Geometry",
			)
		pred_state = _denormalize_state(pred_state, art.model)

		data_source = build_data_source(cfg, device=device)

		for mod in modality_dims.keys():
			if hasattr(data_source, "meshes") and mod in getattr(data_source, "meshes"):
				mesh = data_source.meshes[mod]
				ref_samples_pts, _ = trimesh.sample.sample_surface(mesh, ref_samples)
				ref_points = ref_samples_pts.astype(np.float32)
			else:
				ref_state = data_source.sample_batch(ref_samples)
				ref_state = _denormalize_state(ref_state, art.model)
				ref_points = _state_to_points(ref_state, mod)

			pred_points = _state_to_points(pred_state, mod)

			cd = chamfer_distance_l2(pred_points, ref_points)
			hd = hausdorff_distance(pred_points, ref_points)
			emd = earth_movers_distance(pred_points, ref_points, max_points=emd_max)
			fscore_map: Dict[float, float] = {}
			precision_map: Dict[float, float] = {}
			recall_map: Dict[float, float] = {}
			for thr in fscore_thresholds:
				f, precision, recall = f_score(pred_points, ref_points, threshold=thr)
				fscore_map[thr] = f
				precision_map[thr] = precision
				recall_map[thr] = recall

			f = fscore_map[fscore_thresh]
			precision = precision_map[fscore_thresh]
			recall = recall_map[fscore_thresh]

			p2s = None
			if hasattr(data_source, "meshes") and mod in getattr(data_source, "meshes"):
				try:
					p2s = point_to_surface_distance(pred_points, data_source.meshes[mod])
				except Exception:
					p2s = None

			results[mod] = {
				"chamfer_L2": cd,
				"earth_movers": emd,
				"hausdorff": hd,
				"point_to_surface": float(p2s) if p2s is not None else float("nan"),
				"f_score": f,
				"precision": precision,
				"recall": recall,
				"f_score_by_threshold": {str(k): v for k, v in fscore_map.items()},
				"precision_by_threshold": {str(k): v for k, v in precision_map.items()},
				"recall_by_threshold": {str(k): v for k, v in recall_map.items()},
			}

		lines.append("=== Geometry Evaluation ===")
		lines.append(f"checkpoint: {ckpt_path}")
		lines.append(f"num_points: {num_points}")
		lines.append(f"ref_samples: {ref_samples}")
		lines.append(f"steps: {steps}")
		lines.append(f"target: {target}")
		if guidance.enabled:
			lines.append(f"guidance: scale={guidance.scale}, mode={guidance.mode}, sigma_max={guidance.sigma_max}")
		if len(fscore_thresholds) == 1:
			lines.append(f"fscore_threshold: {fscore_thresh}")
		else:
			lines.append(f"fscore_thresholds: {fscore_thresholds}")
		lines.append(f"emd_max_points: {emd_max}")
		lines.append("")

		for mod, metrics in results.items():
			lines.append(f"[Modality: {mod}]")
			lines.append(f"  Chamfer (L2): {_format_report_value(metrics['chamfer_L2'])}")
			lines.append(f"  EMD (Wasserstein-1): {_format_report_value(metrics['earth_movers'])}")
			lines.append(f"  Hausdorff: {_format_report_value(metrics['hausdorff'])}")
			lines.append(f"  Point-to-Surface: {_format_report_value(metrics['point_to_surface'])}")
			if len(fscore_thresholds) == 1:
				lines.append(f"  F-Score: {_format_report_value(metrics['f_score'])}")
				lines.append(f"    Precision: {_format_report_value(metrics['precision'])}")
				lines.append(f"    Recall: {_format_report_value(metrics['recall'])}")
			else:
				lines.append("  F-Score (multi-threshold):")
				for thr in fscore_thresholds:
					thr_key = str(thr)
					thr_str = _format_report_value(thr)
					lines.append(
						f"    @ {thr_str}: F={_format_report_value(metrics['f_score_by_threshold'][thr_key])}, "
						f"P={_format_report_value(metrics['precision_by_threshold'][thr_key])}, "
						f"R={_format_report_value(metrics['recall_by_threshold'][thr_key])}"
					)
			lines.append("")

		if results:
			avg_cd = _mean_finite([float(m["chamfer_L2"]) for m in results.values()])
			avg_emd = _mean_finite([float(m["earth_movers"]) for m in results.values()])
			avg_hd = _mean_finite([float(m["hausdorff"]) for m in results.values()])
			avg_p2s = _mean_finite([float(m["point_to_surface"]) for m in results.values()])
			avg_f = _mean_finite([float(m["f_score"]) for m in results.values()])
			avg_precision = _mean_finite([float(m["precision"]) for m in results.values()])
			avg_recall = _mean_finite([float(m["recall"]) for m in results.values()])

			lines.append("[Geometry Average: all_shapes]")
			lines.append(f"  Chamfer (L2): {_format_report_value(avg_cd)}")
			lines.append(f"  EMD (Wasserstein-1): {_format_report_value(avg_emd)}")
			lines.append(f"  Hausdorff: {_format_report_value(avg_hd)}")
			lines.append(f"  Point-to-Surface: {_format_report_value(avg_p2s)}")
			if len(fscore_thresholds) == 1:
				lines.append(f"  F-Score: {_format_report_value(avg_f)}")
				lines.append(f"    Precision: {_format_report_value(avg_precision)}")
				lines.append(f"    Recall: {_format_report_value(avg_recall)}")
			else:
				lines.append("  F-Score (multi-threshold):")
				for thr in fscore_thresholds:
					thr_key = str(thr)
					avg_f_thr = _mean_finite([
						float(m["f_score_by_threshold"][thr_key]) for m in results.values()
					])
					avg_p_thr = _mean_finite([
						float(m["precision_by_threshold"][thr_key]) for m in results.values()
					])
					avg_r_thr = _mean_finite([
						float(m["recall_by_threshold"][thr_key]) for m in results.values()
					])
					thr_str = _format_report_value(thr)
					lines.append(
						f"    @ {thr_str}: F={_format_report_value(avg_f_thr)}, "
						f"P={_format_report_value(avg_p_thr)}, R={_format_report_value(avg_r_thr)}"
					)
			lines.append("")


	if joint_enabled:
		joint_num_points = int(_get_attr(joint_cfg, "num_points", num_points)) if joint_cfg is not None else num_points
		joint_steps = int(_get_attr(joint_cfg, "steps", _get_attr(eval_cfg, "steps", 18)))
		joint_target = str(_get_attr(joint_cfg, "target", _get_attr(eval_cfg, "target", "gaussian"))).lower()
		joint_noise_mesh = _get_attr(joint_cfg, "noise_mesh", None)
		joint_source = str(getattr(joint_cfg, "source_modality", "A")) if joint_cfg is not None else "A"
		joint_guidance = _build_guidance_cfg(cfg, schedule=art.schedule, mode="joint", section="joint_correspondence")

		if data_source is None:
			data_source = build_data_source(cfg, device=device)
		if not hasattr(data_source, "meshes"):
			raise ValueError("joint_correspondence requires data_source.meshes")
		meshes = data_source.meshes
		if joint_source not in meshes:
			raise KeyError(f"Unknown source modality '{joint_source}' for joint_correspondence")

		with torch.no_grad():
			joint_state = _sample_model_points_with_oom_recovery(
				art,
				modality_dims=modality_dims,
				num_points=joint_num_points,
				steps=joint_steps,
				target=joint_target,
				noise_mesh=joint_noise_mesh,
				observed_mask=None,
				observed=None,
				guidance=joint_guidance,
				progress_desc="Evaluating Joint Correspondence",
			)
		joint_state = _denormalize_state(joint_state, art.model)

		src_points = _state_to_points(joint_state, joint_source)
		src_mesh = meshes[joint_source]
		face_idx, bary = _project_to_mesh_barycentric(src_mesh, src_points)
		for mod, mesh in meshes.items():
			if mod == joint_source:
				continue
			if mesh.faces.shape != src_mesh.faces.shape or not np.array_equal(mesh.faces, src_mesh.faces):
				raise ValueError("joint_correspondence requires identical mesh faces across modalities")
			gt_points = _barycentric_points(mesh, face_idx, bary)
			pred_points = _state_to_points(joint_state, mod)
			if pred_points.shape != gt_points.shape:
				raise ValueError("joint_correspondence requires matching point counts")
			diff = pred_points - gt_points
			l2 = np.linalg.norm(diff, axis=1).mean()
			joint_metrics[f"joint_correspondence_L2/{mod}"] = float(l2)

		lines.append("=== Joint Correspondence ===")
		lines.append(f"num_points: {joint_num_points}")
		lines.append(f"steps: {joint_steps}")
		lines.append(f"target: {joint_target}")
		lines.append(f"source_modality: {joint_source}")
		if joint_guidance.enabled:
			lines.append(
				f"guidance: scale={joint_guidance.scale}, mode={joint_guidance.mode}, sigma_max={joint_guidance.sigma_max}"
			)
		for key, value in joint_metrics.items():
			lines.append(f"{key}: {_format_report_value(value)}")
		if joint_metrics:
			joint_avg = _mean_finite([float(v) for v in joint_metrics.values()])
			joint_metrics["joint_correspondence_L2/avg_all"] = joint_avg
			lines.append(f"joint_correspondence_L2/avg_all: {_format_report_value(joint_avg)}")
		lines.append("")

	if correspondence_enabled:
		corr_points = int(_get_attr(correspondence_cfg, "num_points", num_points))
		corr_steps = int(_get_attr(correspondence_cfg, "steps", steps))
		corr_target = str(_get_attr(correspondence_cfg, "target", target)).lower()
		corr_noise_mesh = _get_attr(correspondence_cfg, "noise_mesh", noise_mesh)
		corr_mode = str(_get_attr(correspondence_cfg, "mode", "conditional")).lower()
		if corr_mode not in {"joint", "conditional", "single"}:
			raise ValueError("eval.correspondence.mode must be one of: joint, conditional, single")
		corr_guidance = _build_guidance_cfg(cfg, schedule=art.schedule, mode=corr_mode, section="correspondence")
		corr_pairs = int(_get_attr(correspondence_cfg, "num_pairs", 1))
		cond_norm = bool(_get_attr(correspondence_cfg, "conditional_normalized", False))
		corr_seed = _get_attr(correspondence_cfg, "seed", _get_attr(cfg, "run.seed", None))
		corr_seed = int(corr_seed) if corr_seed is not None else None

		if data_source is None:
			data_source = build_data_source(cfg, device=device)
		pairs = _sample_correspondence_pairs(
			data_source,
			modality_dims=modality_dims,
			num_pairs=corr_pairs,
			seed=corr_seed,
		)

		agg: Dict[str, float] = {
			"mean_dist": 0.0,
			"median_dist": 0.0,
			"var_mean": 0.0,
			"var_trace": 0.0,
			"mahalanobis": 0.0,
			"gaussian_loglik": 0.0,
			"kde_loglik": 0.0,
		}
		pair_iter = tqdm(pairs, desc="Evaluating Correspondence", ncols=100)
		for obs_mod, obs_point, tgt_mod, tgt_point in pair_iter:
			metrics = _eval_correspondence_pair(
				art=art,
				modality_dims=modality_dims,
				observed_mod=obs_mod,
				observed_point=obs_point,
				target_mod=tgt_mod,
				target_point=tgt_point,
				corr_points=corr_points,
				corr_steps=corr_steps,
				corr_target=corr_target,
				corr_noise_mesh=corr_noise_mesh,
				corr_guidance=corr_guidance,
				conditional_normalized=cond_norm,
			)
			for k in agg.keys():
				agg[k] += metrics[k]
		for k in agg.keys():
			agg[k] /= max(corr_pairs, 1)

		lines.append("=== Correspondence Evaluation (Average) ===")
		lines.append(f"num_pairs: {corr_pairs}")
		if corr_seed is not None:
			lines.append(f"pair_seed: {corr_seed}")
		lines.append(f"num_points: {corr_points}")
		lines.append(f"steps: {corr_steps}")
		lines.append(f"target: {corr_target}")
		if corr_guidance.enabled:
			lines.append(f"guidance: scale={corr_guidance.scale}, mode={corr_guidance.mode}, sigma_max={corr_guidance.sigma_max}")
		lines.append(f"mean_dist: {_format_report_value(agg['mean_dist'])}")
		lines.append(f"median_dist: {_format_report_value(agg['median_dist'])}")
		lines.append(f"var_mean: {_format_report_value(agg['var_mean'])}")
		lines.append(f"var_trace: {_format_report_value(agg['var_trace'])}")
		lines.append(f"mahalanobis: {_format_report_value(agg['mahalanobis'])}")
		lines.append(f"gaussian_loglik: {_format_report_value(agg['gaussian_loglik'])}")
		lines.append(f"kde_loglik: {_format_report_value(agg['kde_loglik'])}")
		lines.append("")

		specific_cfg = getattr(correspondence_cfg, "specific_pair", None)
		if specific_cfg is not None and bool(getattr(specific_cfg, "enable", False)):
			name = getattr(specific_cfg, "name", None)
			obs_mod = getattr(specific_cfg, "observed_modality", None)
			obs_point = getattr(specific_cfg, "observed_point", None)
			tgt_mod = getattr(specific_cfg, "target_modality", None)
			tgt_point = getattr(specific_cfg, "target_point", None)
			if obs_mod is None or obs_point is None or tgt_mod is None or tgt_point is None:
				raise ValueError("specific_pair requires observed_modality/observed_point/target_modality/target_point")

			metrics = _eval_correspondence_pair(
				art=art,
				modality_dims=modality_dims,
				observed_mod=str(obs_mod),
				observed_point=np.asarray(obs_point, dtype=np.float32),
				target_mod=str(tgt_mod),
				target_point=np.asarray(tgt_point, dtype=np.float32),
				corr_points=corr_points,
				corr_steps=corr_steps,
				corr_target=corr_target,
				corr_noise_mesh=corr_noise_mesh,
				corr_guidance=corr_guidance,
				conditional_normalized=cond_norm,
			)

			lines.append("=== Correspondence Evaluation (Specific Pair) ===")
			if name:
				lines.append(f"name: {name}")
			lines.append(f"observed_modality: {obs_mod}")
			lines.append(f"target_modality: {tgt_mod}")
			lines.append(f"mean_dist: {_format_report_value(metrics['mean_dist'])}")
			lines.append(f"median_dist: {_format_report_value(metrics['median_dist'])}")
			lines.append(f"var_mean: {_format_report_value(metrics['var_mean'])}")
			lines.append(f"var_trace: {_format_report_value(metrics['var_trace'])}")
			lines.append(f"mahalanobis: {_format_report_value(metrics['mahalanobis'])}")
			lines.append(f"gaussian_loglik: {_format_report_value(metrics['gaussian_loglik'])}")
			lines.append(f"kde_loglik: {_format_report_value(metrics['kde_loglik'])}")
			lines.append("")

	report = "\n".join(lines)
	print(report)

	if out_dir:
		name_suffix = _get_attr(cfg, "eval.name_suffix", None)
		report_name = "eval_report"
		if name_suffix:
			report_name = f"{report_name}_{_sanitize_tag(str(name_suffix))}"
		out_path = os.path.join(out_dir, f"{report_name}.txt")
		with open(out_path, "w", encoding="utf-8") as f:
			f.write(report)
		print(f"Saved: {out_path}")
