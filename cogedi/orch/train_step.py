from __future__ import annotations

import inspect
import time
from typing import Dict, Optional, Tuple

import torch

from cogedi.dtypes import State, Sigma, ObservedMask, DenoiseInput, Descriptor

def train_step(
    *,
    model,
    forward,
    parametrization,
    schedule,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    x0: State,
    sigma_override: Optional[Sigma] = None,
    descriptor: Optional[Descriptor] = None,
    chosen_modality_idx: Optional[torch.Tensor] = None,
    chosen_point_idx: Optional[torch.Tensor] = None,
    modality_order: Optional[Tuple[str, ...]] = None,
    observed_mask: Optional[ObservedMask] = None,
    observed: Optional[State] = None,
    clamp_prob: float = 0.0,
    apply_prob: float = 1.0,
    sigma_mode: str = "default",
    generator: Optional[torch.Generator] = None,
    enable_timings: bool = False,
    return_sigma: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], Sigma]:
    """
    One generic optimization step.

    Works for:
      - any parametrization target via parametrization.target(...)
      - UniDiff conditioning via observed_mask/observed
      - EDM-style denoiser mode if model.precond.supports_denoised() is True

    Assumptions:
      - x0[m] has shape [B, N, D] 
      - sigma[m] has shape [B]
    """
    model.train()
    model_module = model.module if hasattr(model, "module") else model

    def _sync_cuda() -> None:
        if not enable_timings:
            return
        try:
            p = next(model.parameters())
            if p.is_cuda:
                torch.cuda.synchronize(p.device)
        except Exception:
            pass

    timings: Dict[str, float] = {}
    step_t0 = time.perf_counter()

    # Infer batch size
    first = next(iter(x0.values()))
    B = first.shape[0]

    # 1) sample sigma per modality: dict of [B]
    _sync_cuda()
    t0 = time.perf_counter()
    sigma = sigma_override
    if sigma is None:
        sigma = schedule.sample_training_sigma(
            batch_size=B,
            sigma_mode=sigma_mode,
            clamp_prob=clamp_prob,
            apply_prob=apply_prob,
        )
    _sync_cuda()
    if enable_timings:
        timings["sigma_sample"] = time.perf_counter() - t0
    
    # 2) sample eps: same shape as x0
    _sync_cuda()
    t0 = time.perf_counter()
    eps: State = {m: torch.randn_like(x0[m], generator=generator) for m in x0.keys()}
    _sync_cuda()
    if enable_timings:
        timings["eps_sample"] = time.perf_counter() - t0

    # 3) noisy input x
    _sync_cuda()
    t0 = time.perf_counter()
    x: State = forward.q_sample(x0=x0, sigma=sigma, eps=eps)
    _sync_cuda()
    if enable_timings:
        timings["q_sample"] = time.perf_counter() - t0

    optimizer.zero_grad(set_to_none=True)

    # 4) Branch: EDM-precond denoiser mode vs generic parametrization mode
    if getattr(model_module, "precond", None) is not None and model_module.precond.supports_denoised():
        # EDM mode: run through model(...) so DDP hooks are active, then map F -> x0.
        _sync_cuda()
        t0 = time.perf_counter()
        F_state = model(
            x,
            sigma,
            descriptor,
            observed_mask=observed_mask,
            observed=observed,
        )
        pred = model_module.precond.denoised(x=x, sigma=sigma, F=F_state)
        _sync_cuda()
        if enable_timings:
            timings["model_forward"] = time.perf_counter() - t0
        target = x0

        # Here, loss_fn should be compatible with pred/target both being State.
        loss_kwargs = {
            "pred": pred,
            "target": target,
            "sigma": sigma,
            "observed_mask": observed_mask,
        }
        loss_sig = inspect.signature(loss_fn.__call__).parameters
        if "descriptor" in loss_sig:
            loss_kwargs["descriptor"] = descriptor
        if "chosen_modality_idx" in loss_sig:
            loss_kwargs["chosen_modality_idx"] = chosen_modality_idx
        if "chosen_point_idx" in loss_sig:
            loss_kwargs["chosen_point_idx"] = chosen_point_idx
        if "modality_order" in loss_sig:
            loss_kwargs["modality_order"] = modality_order
        if "timing_enabled" in loss_sig:
            loss_kwargs["timing_enabled"] = enable_timings
        if "timing_buffer" in loss_sig:
            loss_kwargs["timing_buffer"] = timings
        _sync_cuda()
        t0 = time.perf_counter()
        loss_out = loss_fn(**loss_kwargs)
        _sync_cuda()
        if enable_timings:
            timings["loss_total"] = time.perf_counter() - t0
        loss = loss_out.loss

    else:
        # Generic mode: model.forward predicts in parametrization space
        _sync_cuda()
        t0 = time.perf_counter()
        pred = model.forward(x, sigma, descriptor, observed_mask=observed_mask, observed=observed)
        _sync_cuda()
        if enable_timings:
            timings["model_forward"] = time.perf_counter() - t0
        target = parametrization.target(x0=x0, eps=eps, sigma=sigma)

        loss_kwargs = {
            "pred": pred,
            "target": target,
            "sigma": sigma,
            "observed_mask": observed_mask,
        }
        loss_sig = inspect.signature(loss_fn.__call__).parameters
        if "descriptor" in loss_sig:
            loss_kwargs["descriptor"] = descriptor
        if "chosen_modality_idx" in loss_sig:
            loss_kwargs["chosen_modality_idx"] = chosen_modality_idx
        if "chosen_point_idx" in loss_sig:
            loss_kwargs["chosen_point_idx"] = chosen_point_idx
        if "modality_order" in loss_sig:
            loss_kwargs["modality_order"] = modality_order
        if "timing_enabled" in loss_sig:
            loss_kwargs["timing_enabled"] = enable_timings
        if "timing_buffer" in loss_sig:
            loss_kwargs["timing_buffer"] = timings
        _sync_cuda()
        t0 = time.perf_counter()
        loss_out = loss_fn(**loss_kwargs)
        _sync_cuda()
        if enable_timings:
            timings["loss_total"] = time.perf_counter() - t0
        loss = loss_out.loss

    _sync_cuda()
    t0 = time.perf_counter()
    loss.backward()
    _sync_cuda()
    if enable_timings:
        timings["backward"] = time.perf_counter() - t0

    _sync_cuda()
    t0 = time.perf_counter()
    optimizer.step()
    _sync_cuda()
    if enable_timings:
        timings["optimizer_step"] = time.perf_counter() - t0

    # logs
    _sync_cuda()
    if enable_timings:
        timings["step_total"] = time.perf_counter() - step_t0

    logs: Dict[str, float] = {"loss": float(loss.detach().cpu())}
    for k, v in loss_out.terms.items():
        logs[k] = float(v.detach().cpu())
    if enable_timings:
        for k, v in timings.items():
            logs[f"time/{k}"] = float(v)
    if return_sigma:
        return logs, sigma
    return logs
