from __future__ import annotations

import os
import re
from glob import glob
from typing import Dict

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from cogedi.build import build_training
from cogedi.dtypes import Descriptor, Sigma, State
from cogedi.utils.descriptors import (
	GeodesicDescriptorLookup,
	build_geodesic_descriptor_lookup,
	geodesic_descriptor_from_point_knn_weighted,
	geodesic_descriptor_from_point_nearest_vertex,
	geodesic_descriptor_from_triangle_barycentric,
)
from cogedi.orch.train_step import train_step
from cogedi.orch.logging import MetricLogger, SmoothedValue


def _get_dist_info() -> tuple[bool, int, int, int]:
	if dist.is_available() and dist.is_initialized():
		rank = dist.get_rank()
		world_size = dist.get_world_size()
		local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
		return True, rank, world_size, local_rank

	if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
		rank = int(os.environ["RANK"])
		world_size = int(os.environ["WORLD_SIZE"])
		local_rank = int(os.environ.get("LOCAL_RANK", 0))
		return world_size > 1, rank, world_size, local_rank

	if "SLURM_PROCID" in os.environ:
		rank = int(os.environ["SLURM_PROCID"])
		world_size = int(os.environ.get("SLURM_NTASKS", "1"))
		local_rank = int(os.environ.get("SLURM_LOCALID", rank % max(1, torch.cuda.device_count())))
		return world_size > 1, rank, world_size, local_rank

	return False, 0, 1, 0


def _init_distributed(cfg) -> tuple[bool, int, int, int]:
	use_dist, rank, world_size, local_rank = _get_dist_info()
	setattr(cfg.run, "rank", rank)
	setattr(cfg.run, "world_size", world_size)
	setattr(cfg.run, "local_rank", local_rank)

	if not use_dist:
		return False, rank, world_size, local_rank

	if not torch.cuda.is_available():
		raise RuntimeError("Distributed training requires CUDA devices")

	torch.cuda.set_device(local_rank)
	os.environ.setdefault("LOCAL_RANK", str(local_rank))
	os.environ.setdefault("RANK", str(rank))
	os.environ.setdefault("WORLD_SIZE", str(world_size))
	os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
	os.environ.setdefault("MASTER_PORT", "29500")

	if not dist.is_initialized():
		dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
		dist.barrier()

	return True, rank, world_size, local_rank


def _cleanup_distributed() -> None:
	if dist.is_available() and dist.is_initialized():
		dist.barrier()
		dist.destroy_process_group()


def _is_main_process() -> bool:
	return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _reduce_logs(logs: Dict[str, float], *, device: torch.device) -> Dict[str, float]:
	if (not dist.is_available()) or (not dist.is_initialized()):
		return logs

	world_size = dist.get_world_size()
	if world_size <= 1:
		return logs

	keys = sorted(logs.keys())
	vals = torch.tensor([float(logs[k]) for k in keys], device=device, dtype=torch.float64)
	dist.all_reduce(vals, op=dist.ReduceOp.SUM)
	vals /= float(world_size)
	return {k: float(v) for k, v in zip(keys, vals.tolist())}


def _materialize_lazy_modules(
	model,
	*,
	data_source,
	schedule,
	sample_mode: str,
	descriptor_training_enabled: bool,
	descriptor_lookups: Dict[str, GeodesicDescriptorLookup],
	descriptor_interpolation: str,
	knn_neighbors: int,
	modality_order: tuple[str, ...],
	sigma_mode: str,
	clamp_prob: float,
	apply_prob: float,
) -> None:
	model_module = model.module if hasattr(model, "module") else model
	was_training = model_module.training
	model_module.eval()
	with torch.no_grad():
		# Use a tiny batch to initialize all lazy layers before DDP inspects parameters.
		x_dummy = data_source.sample_batch(1)
		dummy_bs = next(iter(x_dummy.values())).shape[0]
		sigma_dummy = schedule.sample_training_sigma(
			batch_size=dummy_bs,
			sigma_mode=sigma_mode,
			clamp_prob=clamp_prob,
			apply_prob=apply_prob,
		)
		descriptor = None
		chosen_modality_idx = None
		chosen_point_idx = None
		if descriptor_training_enabled:
			descriptor, chosen_modality_idx, chosen_point_idx = _build_unsupervised_descriptor(
				x0=x_dummy,
				data_source=data_source,
				lookups=descriptor_lookups,
				descriptor_interpolation=descriptor_interpolation,
				knn_neighbors=knn_neighbors,
				modality_order=modality_order,
			)
		# Call the model once so LazyLinear layers materialize their weights.
		_ = model_module(
			x_dummy,
			sigma_dummy,
			descriptor,
			observed_mask=None,
			observed=None,
		)
	if was_training:
		model_module.train()


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


def _build_unsupervised_descriptor(
	*,
	x0: State,
	data_source,
	lookups: Dict[str, GeodesicDescriptorLookup],
	descriptor_interpolation: str,
	knn_neighbors: int,
	modality_order: tuple[str, ...],
) -> tuple[Descriptor, torch.Tensor, torch.Tensor]:
	first = next(iter(x0.values()))
	device = first.device
	dtype = first.dtype
	B = first.shape[0]
	M = len(modality_order)

	if not lookups:
		raise ValueError("Landmarks mode requires precomputed geodesic descriptor lookups")

	ref_lookup = next(iter(lookups.values()))
	desc_dim = int(ref_lookup.vertex_to_landmark.shape[1])
	desc = torch.empty((B, desc_dim), device=device, dtype=dtype)
	norm_stats = data_source.get_normalization_stats()
	chosen_modality_idx = torch.randint(0, M, (B,), device=device)
	chosen_point_idx = torch.zeros(B, dtype=torch.long, device=device)

	for mod_idx, mod in enumerate(modality_order):
		idx = torch.where(chosen_modality_idx == mod_idx)[0]
		if idx.numel() == 0:
			continue

		if mod not in lookups:
			raise KeyError(f"No descriptor lookup for conditioned modality '{mod}'")
		lookup = lookups[mod]

		n_pts = int(x0[mod].shape[1])
		point_idx = torch.randint(0, n_pts, (idx.numel(),), device=device)
		chosen_point_idx[idx] = point_idx

		pts_all = x0[mod][idx, :, :3]
		pts = pts_all[torch.arange(idx.numel(), device=device), point_idx]
		if pts.shape[-1] != 3:
			raise ValueError(f"Descriptor extraction expects 3D points, got shape {tuple(pts.shape)} for modality '{mod}'")

		if norm_stats is not None and mod in norm_stats:
			mean, std = norm_stats[mod]
			mean = mean.to(device=device, dtype=dtype)
			std = std.to(device=device, dtype=dtype)
			pts = pts * std[:3] + mean[:3]

		interp = descriptor_interpolation
		if interp == "barycentric":
			proj = data_source.project_points_to_surface(mod, pts)
			desc_mod = geodesic_descriptor_from_triangle_barycentric(
				lookup,
				triangle_ids=proj.face_ids.reshape(-1),
				barycentric=proj.barycentric.reshape(-1, 3),
				output_device=device,
				output_dtype=dtype,
			)
		elif interp == "nearest_neighbor":
			desc_mod = geodesic_descriptor_from_point_nearest_vertex(
				lookup,
				points=pts,
				output_device=device,
				output_dtype=dtype,
			)
		elif interp == "knn":
			desc_mod = geodesic_descriptor_from_point_knn_weighted(
				lookup,
				points=pts,
				n_neighbors=knn_neighbors,
				output_device=device,
				output_dtype=dtype,
			)
		else:
			raise ValueError(
				"descriptor_interpolation must be one of: barycentric, nearest_neighbor, kNN"
			)

		desc[idx] = desc_mod

	return Descriptor(type="geodesic", data=desc), chosen_modality_idx, chosen_point_idx

def run(cfg) -> None:
	is_distributed, rank, _, local_rank = _init_distributed(cfg)
	try:
		art = build_training(cfg)
		device = art.device
		seed = int(getattr(cfg.run, "seed", 0)) + rank
		torch.manual_seed(seed)

		data_source = art.data_source
		stats = data_source.get_normalization_stats()
		if stats:
			for mod, (mean, std) in stats.items():
				art.model.set_normalization_stats(mod, mean, std)
		if hasattr(art.loss, "attach_data_source"):
			art.loss.attach_data_source(data_source)

		supervision_mode = str(getattr(cfg.run, "supervision", "full")).lower()
		descriptor_training_enabled = (
			supervision_mode == "landmarks"
			and getattr(art.model, "descriptor_embedder", None) is not None
		)
		if descriptor_training_enabled and getattr(art.loss, "name", "") != "geomdist_unsupervised":
			raise ValueError(
				"run.supervision='landmarks' with descriptor conditioning requires loss.type='geomdist_unsupervised'"
			)

		descriptor_lookups: Dict[str, GeodesicDescriptorLookup] = {}
		descriptor_interpolation = "barycentric"
		knn_neighbors = 4
		if descriptor_training_enabled:
			descriptor_lookups = _build_descriptor_lookups(data_source)
			landmark_cfg = getattr(cfg.data, "landmark_supervision", None)
			surface_point_mode = str(getattr(landmark_cfg, "surface_point", "projection")).strip().lower()
			descriptor_interpolation = str(
				getattr(landmark_cfg, "descriptor_interpolation", "barycentric")
			).strip().lower()
			if descriptor_interpolation in {"knn", "knn_weighted"}:
				descriptor_interpolation = "knn"
			elif descriptor_interpolation == "nearest":
				descriptor_interpolation = "nearest_neighbor"
			if surface_point_mode == "pull" and descriptor_interpolation == "barycentric":
				raise ValueError(
					"descriptor_interpolation='barycentric' is not supported with data.landmark_supervision.surface_point='pull'. "
					"Use nearest_neighbor or knn."
				)
			knn_neighbors = int(getattr(landmark_cfg, "knn_neighbors", getattr(landmark_cfg, "knn_k", 4)))

		modality_order = tuple(art.model.modalities)
		model = art.model

		sample_cfg = getattr(cfg.train, "sample", None)
		if sample_cfg is not None:
			sample_mode = str(getattr(sample_cfg, "mode", "surface")).lower()
			batch_size = int(getattr(sample_cfg, "batch_size", getattr(cfg.train, "batch_size", 1)))
			resample = str(getattr(sample_cfg, "resample", "step")).lower()
		else:
			sample_mode = "surface"
			batch_size = int(getattr(cfg.train, "batch_size", 1))
			resample = str(getattr(cfg.train, "resample", "step")).lower()

		if sample_mode not in {"surface", "vertices"}:
			raise ValueError("train.sample.mode must be 'surface' or 'vertices'")
		if resample not in {"step", "epoch", "never"}:
			raise ValueError("train.sample.resample must be one of: step, epoch, never")

		max_epochs = int(getattr(cfg.train, "max_epochs", getattr(cfg.train, "epochs", 1)))
		log_every = int(getattr(cfg.train, "log_every", 20))
		progress_hook = getattr(cfg.train, "_progress_hook", None)
		exp_name = getattr(cfg.run, "experiment_name", None)
		paths_cfg = getattr(cfg, "paths", None)
		log_base = getattr(paths_cfg, "logs", None) if paths_cfg is not None else None
		ckpt_base = getattr(paths_cfg, "checkpoints", None) if paths_cfg is not None else None
		log_dir = getattr(cfg.train, "log_dir", None) or (os.path.join(log_base, exp_name) if log_base and exp_name else None)
		ckpt_dir = (
			getattr(cfg.train, "checkpoint_dir", None)
			or getattr(cfg.train, "output_dir", None)
			or (os.path.join(ckpt_base, exp_name) if ckpt_base and exp_name else None)
			or log_dir
		)
		ckpt_every = int(getattr(cfg.train, "checkpoint_every", 0))
		steps_per_epoch = max(int(getattr(cfg.train, "steps_per_epoch", 1)), 1)
		timings_enabled = bool(getattr(cfg.train, "timings", False)) and _is_main_process()
		resume = getattr(cfg.train, "resume", None)
		auto_resume = bool(getattr(cfg.train, "auto_resume", False))

		hpo_mode = bool(getattr(cfg.run, "_hpo", False))
		writer = None
		if log_dir and not hpo_mode and _is_main_process():
			os.makedirs(log_dir, exist_ok=True)
			writer = SummaryWriter(log_dir=log_dir)

		if ckpt_dir and _is_main_process():
			os.makedirs(ckpt_dir, exist_ok=True)

		def _find_latest_checkpoint(path: str) -> str | None:
			candidates = []
			latest_path = os.path.join(path, "checkpoint-latest.pth")
			if os.path.exists(latest_path):
				return latest_path
			for p in glob(os.path.join(path, "checkpoint-epoch-*.pth")):
				m = re.search(r"checkpoint-epoch-(\d+)\.pth$", p)
				if m:
					candidates.append((int(m.group(1)), p))
			if not candidates:
				return None
			candidates.sort(key=lambda x: x[0])
			return candidates[-1][1]

		start_epoch = 0
		ckpt_path = None
		if resume:
			if str(resume).lower() == "latest":
				if not ckpt_dir:
					raise ValueError("Cannot resolve checkpoint directory for resume; set paths.checkpoints and run.experiment_name")
				ckpt_path = _find_latest_checkpoint(ckpt_dir)
				if ckpt_path is None:
					raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
			else:
				ckpt_name = resume if str(resume).endswith(".pth") else f"{resume}.pth"
				if not ckpt_dir:
					raise ValueError("Cannot resolve checkpoint directory for resume; set paths.checkpoints and run.experiment_name")
				ckpt_path = os.path.join(ckpt_dir, ckpt_name)
		elif auto_resume and ckpt_dir:
			ckpt_path = _find_latest_checkpoint(ckpt_dir)

		if ckpt_path:
			if _is_main_process():
				print(f"Resuming from checkpoint: {ckpt_path}")
			ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
			# Backward compatibility: old checkpoints may contain removed sigma_embedder.proj params.
			model_state = ckpt.get("model", {})
			model_state.pop("sigma_embedder.proj.weight", None)
			model_state.pop("sigma_embedder.proj.bias", None)
			model.load_state_dict(model_state, strict=True)
			if "optimizer" in ckpt:
				art.optimizer.load_state_dict(ckpt["optimizer"])
			start_epoch = int(ckpt.get("epoch", 0)) + 1
			if _is_main_process():
				print("Continuing Training:")
		elif _is_main_process():
			print("Starting Training:")

		cached_x0 = None
		print_freq = log_every if _is_main_process() else (steps_per_epoch + 1)
		sigma_cfg = getattr(cfg.train, "sigma_sampling", None)
		if sigma_cfg is None:
			clamp_prob = float(getattr(cfg.train, "clamp_prob", 0.0))
			apply_prob = 1.0
			sigma_mode = "default"
		else:
			sigma_mode = str(getattr(sigma_cfg, "sigma_mode", "default")).lower()
			clamp_prob = float(getattr(sigma_cfg, "clamp_prob", 0.0))
			apply_prob = float(getattr(sigma_cfg, "apply_prob", 1.0))

		_materialize_lazy_modules(
			model,
			data_source=data_source,
			schedule=art.schedule,
			sample_mode=sample_mode,
			descriptor_training_enabled=descriptor_training_enabled,
			descriptor_lookups=descriptor_lookups,
			descriptor_interpolation=descriptor_interpolation,
			knn_neighbors=knn_neighbors,
			modality_order=modality_order,
			sigma_mode=sigma_mode,
			clamp_prob=clamp_prob,
			apply_prob=apply_prob,
		)

		if is_distributed:
			model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

		for epoch_idx in range(start_epoch, max_epochs):
			metric_logger = MetricLogger(delimiter="  ")
			metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6e}"))
			header = f"Epoch: [{epoch_idx}/{max_epochs}]"
			for step_in_epoch in metric_logger.log_every(range(steps_per_epoch), print_freq, header=header):
				step = epoch_idx * steps_per_epoch + step_in_epoch
				epoch = epoch_idx + step_in_epoch / steps_per_epoch
				if art.lr_schedule is not None:
					art.lr_schedule.step(art.optimizer, epoch, cfg)

				if sample_mode == "vertices":
					if cached_x0 is None:
						cached_x0 = data_source.sample_vertices_batch()
				else:
					need_resample = (
						resample == "step"
						or cached_x0 is None
						or (resample == "epoch" and step_in_epoch == 0)
					)
					if need_resample:
						cached_x0 = data_source.sample_batch(batch_size)

				x0 = cached_x0
				descriptor = None
				sigma_override = None
				chosen_modality_idx = None
				chosen_point_idx = None
				if descriptor_training_enabled:
					sigma_override = art.schedule.sample_training_sigma(
						batch_size=batch_size,
						sigma_mode=sigma_mode,
						clamp_prob=clamp_prob,
						apply_prob=apply_prob,
					)
					descriptor, chosen_modality_idx, chosen_point_idx = _build_unsupervised_descriptor(
						x0=x0,
						data_source=data_source,
						lookups=descriptor_lookups,
						descriptor_interpolation=descriptor_interpolation,
						knn_neighbors=knn_neighbors,
						modality_order=modality_order,
					)

				if writer is not None:
					logs, sigma = train_step(
						model=model,
						forward=art.forward,
						parametrization=art.parametrization,
						schedule=art.schedule,
						loss_fn=art.loss,
						optimizer=art.optimizer,
						x0=x0,
						sigma_override=sigma_override,
						descriptor=descriptor,
						chosen_modality_idx=chosen_modality_idx,
						chosen_point_idx=chosen_point_idx,
						modality_order=modality_order,
						clamp_prob=clamp_prob,
						apply_prob=apply_prob,
						sigma_mode=sigma_mode,
						enable_timings=timings_enabled,
						return_sigma=True,
					)
				else:
					logs = train_step(
						model=model,
						forward=art.forward,
						parametrization=art.parametrization,
						schedule=art.schedule,
						loss_fn=art.loss,
						optimizer=art.optimizer,
						x0=x0,
						sigma_override=sigma_override,
						descriptor=descriptor,
						chosen_modality_idx=chosen_modality_idx,
						chosen_point_idx=chosen_point_idx,
						modality_order=modality_order,
						clamp_prob=clamp_prob,
						apply_prob=apply_prob,
						sigma_mode=sigma_mode,
						enable_timings=timings_enabled,
					)

				logs = _reduce_logs(logs, device=device)
				if timings_enabled and _is_main_process():
					timing_items = sorted((k, v) for k, v in logs.items() if k.startswith("time/"))
					if timing_items:
						msg = " | ".join(f"{k[5:]}={v*1000.0:.2f}ms" for k, v in timing_items)
						print(f"[timings] step={step} {msg}")

				loss_val = logs.get("loss", 0.0)
				lr_actual = float(art.optimizer.param_groups[0]["lr"])
				if writer is not None:
					writer.add_scalar("loss/loss_step", loss_val, step)
					writer.add_scalar("loss/loss_epoch", loss_val, epoch)
					writer.add_scalar("learning_rate/lr_step", lr_actual, step)
					writer.add_scalar("learning_rate/lr_epoch", lr_actual, epoch)
					if print_freq > 0 and (step % print_freq == 0):
						for mod, s in sigma.items():
							s_cpu = s.detach().cpu().numpy()
							fig, ax = plt.subplots(figsize=(4.5, 3.0))
							ax.hist(s_cpu, bins=60, alpha=0.8)
							ax.set_yscale("log")
							ax.set_xlabel("sigma")
							ax.set_ylabel("count")
							ax.set_title(f"sigma (log y) / {mod}")
							writer.add_figure(f"sigma_hist_logy/{mod}", fig, step)
							plt.close(fig)

				metric_logger.update(loss=loss_val, lr=lr_actual)

			should_save = (epoch_idx == 0) or (epoch_idx == max_epochs - 1)
			if ckpt_every > 0 and (epoch_idx % ckpt_every == 0):
				should_save = True

			if ckpt_dir and should_save and _is_main_process():
				model_to_save = model.module if hasattr(model, "module") else model
				state = {
					"model": model_to_save.state_dict(),
					"optimizer": art.optimizer.state_dict(),
					"epoch": epoch_idx,
					"step": (epoch_idx + 1) * steps_per_epoch - 1,
				}
				epoch_path = os.path.join(ckpt_dir, f"checkpoint-epoch-{epoch_idx:04d}.pth")
				torch.save(state, epoch_path)
				latest_path = os.path.join(ckpt_dir, "checkpoint-latest.pth")
				torch.save(state, latest_path)

			if progress_hook is not None and _is_main_process():
				try:
					progress_hook(epoch_idx + 1, max_epochs)
				except Exception:
					pass

		if writer is not None:
			writer.flush()
			writer.close()

		if _is_main_process():
			print("Training Ended.")
	finally:
		_cleanup_distributed()
