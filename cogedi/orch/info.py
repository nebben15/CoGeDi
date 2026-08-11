from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from cogedi.build import (
    build_training,
    build_conditioning,
    build_device,
    build_forward,
    build_model,
    build_parametrization,
)
from cogedi.orch.train_step import train_step


def _count_params(model: torch.nn.Module) -> Tuple[int, int, int]:
    total = 0
    trainable = 0
    skipped = 0
    for p in model.parameters():
        if isinstance(p, UninitializedParameter):
            skipped += 1
            continue
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable, skipped


def _bytes_of_params(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if isinstance(p, UninitializedParameter):
            continue
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return int(total)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _resolve_output_path(cfg) -> str | None:
    info_cfg = getattr(cfg, "info", None)
    if info_cfg is not None:
        path = getattr(info_cfg, "output_path", None)
        if path:
            return str(path)

    exp_name = getattr(cfg.run, "experiment_name", None)
    paths_cfg = getattr(cfg, "paths", None)
    log_base = getattr(paths_cfg, "logs", None) if paths_cfg is not None else None
    if log_base and exp_name:
        return os.path.join(log_base, exp_name, "model_info.txt")

    out_dir = getattr(cfg.run, "out_dir", None)
    if out_dir:
        return os.path.join(str(out_dir), "model_info.txt")

    return None


def _gather_info(cfg) -> Dict[str, str]:
    device = build_device(cfg)
    forward = build_forward(cfg)
    conditioning = build_conditioning(cfg)
    parametrization = build_parametrization(cfg, forward=forward)
    model = build_model(cfg, device=device, parametrization=parametrization, conditioning=conditioning)

    with torch.no_grad():
        batch_size = int(getattr(getattr(cfg, "info", None), "init_batch_size", 1))
        x0 = {
            mod: torch.zeros(batch_size, 1, dim, device=device)
            for mod, dim in model.modality_dims.items()
        }
        sigma = {mod: torch.ones(batch_size, device=device) for mod in model.modality_dims.keys()}
        model.forward(x0, sigma)

    total_params, trainable_params, skipped_params = _count_params(model)
    total_bytes = _bytes_of_params(model)

    info: Dict[str, str] = {
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "num_params": str(total_params),
        "num_trainable_params": str(trainable_params),
        "num_uninitialized_params": str(skipped_params),
        "size_bytes": str(total_bytes),
        "size_readable": _format_bytes(total_bytes),
    }

    info_cfg = getattr(cfg, "info", None)
    profile = bool(getattr(info_cfg, "profile_gpu_memory", False)) if info_cfg is not None else False
    if profile and device.type == "cuda":
        batch_size = int(getattr(info_cfg, "profile_batch_size", getattr(getattr(cfg, "train", None), "batch_size", 1)))
        info.update(_profile_gpu_memory(cfg, batch_size=batch_size))
    elif profile:
        info["gpu_profile_status"] = "skipped (device is not cuda)"

    return info


def _profile_gpu_memory(cfg, *, batch_size: int) -> Dict[str, str]:
    art = build_training(cfg)
    device = art.device

    x0 = {
        mod: torch.zeros(batch_size, 1, dim, device=device)
        for mod, dim in art.model.modality_dims.items()
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        train_step(
            model=art.model,
            forward=art.forward,
            parametrization=art.parametrization,
            schedule=art.schedule,
            loss_fn=art.loss,
            optimizer=art.optimizer,
            x0=x0,
        )
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        return {
            "gpu_profile_status": "ok",
            "gpu_profile_batch_size": str(batch_size),
            "gpu_max_allocated_bytes": str(allocated),
            "gpu_max_allocated_readable": _format_bytes(allocated),
            "gpu_max_reserved_bytes": str(reserved),
            "gpu_max_reserved_readable": _format_bytes(reserved),
        }
    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower():
            return {
                "gpu_profile_status": "oom",
                "gpu_profile_batch_size": str(batch_size),
            }
        raise


def run(cfg) -> None:
    info = _gather_info(cfg)

    lines = [
        "MODEL INFO",
        "==========",
    ]
    for key, value in info.items():
        lines.append(f"{key}: {value}")

    text = "\n".join(lines)
    print(text)

    if int(info.get("num_uninitialized_params", "0")) > 0:
        print("WARNING: Some parameters are still uninitialized; counts may be incomplete.")

    info_cfg = getattr(cfg, "info", None)
    save = bool(getattr(info_cfg, "save", False)) if info_cfg is not None else False
    if not save:
        return

    out_path = _resolve_output_path(cfg)
    if not out_path:
        raise ValueError("info.save is true but no output path could be resolved")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
