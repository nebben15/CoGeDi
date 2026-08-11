from __future__ import annotations

"""Denoising utilities for sampling, including classifier-free guidance."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from cogedi.conditioning.base import ConditioningContext
from cogedi.dtypes import DenoiseInput, ObservedMask, State, Descriptor


@dataclass(frozen=True)
class GuidanceConfig:
    """Config for classifier-free guidance during sampling."""

    scale: float
    mode: str
    sigma_max: float
    enabled: bool


def build_denoise_fn(
    *,
    model,
    conditioning,
    observed_mask: Optional[ObservedMask],
    observed: Optional[State],
    guidance: GuidanceConfig,
    descriptor: Optional[Descriptor] = None,
) -> tuple[ConditioningContext, callable]:
    """Build a denoise function with optional classifier-free guidance.

    Returns the conditioning context and a callable denoise_fn(x, sigma, obs_mask).
    """
    ctx = ConditioningContext(observed=observed, observed_mask=observed_mask)

    def _cond_denoise(x_in: State, sigma_in: Dict[str, torch.Tensor], obs_mask: Optional[ObservedMask]) -> State:
        # Apply conditioning and run the model in denoise (x0) mode.
        x_c, s_c = conditioning.apply(x_in, sigma_in, ctx)
        denoise_in = DenoiseInput(
            state=x_c,
            sigma=s_c,
            descriptor=descriptor,
            observed_mask=obs_mask,
            observed=observed,
        )
        out = model.denoise(denoise_in)
        return out.x0_hat

    def _uncond_denoise(x_in: State, sigma_in: Dict[str, torch.Tensor]) -> State:
        # Unconditional branch: no observed conditioning.
        denoise_in = DenoiseInput(
            state=x_in,
            sigma=sigma_in,
            descriptor=descriptor,
            observed_mask=None,
            observed=None,
        )
        out = model.denoise(denoise_in)
        return out.x0_hat

    def _replace_with_noise(
        x_in: State,
        sigma_in: Dict[str, torch.Tensor],
        *,
        keep_mods: Optional[set[str]] = None,
    ) -> tuple[State, Dict[str, torch.Tensor]]:
        # Replace selected modalities with Gaussian noise at sigma_max.
        x_out: State = {}
        s_out: Dict[str, torch.Tensor] = {}
        for mod, x_m in x_in.items():
            if keep_mods is not None and mod in keep_mods:
                x_out[mod] = x_m
                s_out[mod] = sigma_in[mod]
                continue
            noise = torch.randn_like(x_m) * guidance.sigma_max
            x_out[mod] = noise
            s_out[mod] = torch.full_like(sigma_in[mod], guidance.sigma_max)
        return x_out, s_out

    def denoise_fn(x: State, sigma: Dict[str, torch.Tensor], obs_mask: Optional[ObservedMask]) -> State:
        # Optional classifier-free guidance for conditional or joint sampling.
        if not guidance.enabled:
            return _cond_denoise(x, sigma, obs_mask)

        cond_out = _cond_denoise(x, sigma, obs_mask)
        scale = guidance.scale

        if guidance.mode == "conditional":
            if not obs_mask:
                raise ValueError("Guided conditional sampling requires observed_mask")
            keep_mods = {m for m in x.keys() if not obs_mask.get(m, False)}
            x_uncond, s_uncond = _replace_with_noise(x, sigma, keep_mods=keep_mods)
            uncond_out = _uncond_denoise(x_uncond, s_uncond)

            out: State = {}
            for mod, pred in cond_out.items():
                if obs_mask.get(mod, False):
                    out[mod] = pred
                else:
                    out[mod] = (1.0 + scale) * pred - scale * uncond_out[mod]
            return out

        if guidance.mode == "joint":
            out: State = {}
            for mod in cond_out.keys():
                x_uncond, s_uncond = _replace_with_noise(x, sigma, keep_mods={mod})
                uncond_out = _uncond_denoise(x_uncond, s_uncond)
                out[mod] = (1.0 + scale) * cond_out[mod] - scale * uncond_out[mod]
            return out

        return _cond_denoise(x, sigma, obs_mask)

    return ctx, denoise_fn
