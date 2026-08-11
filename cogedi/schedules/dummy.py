from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from cogedi.schedules.base import BaseSigmaSchedule, ScheduleConfig
from cogedi.dtypes import Sigma, ObservedMask


class ComposedDummySigmaSchedule(BaseSigmaSchedule):
    """
    Reads:
      cfg.params.train_sigma (constant sigma)
      cfg.params.sample_sigma (linear grid)
    """
    name = "composed_sigma_schedule"

    def __init__(self, cfg, modalities, device):
        # cfg is the YAML namespace under schedule (type/params)
        # Use sample_sigma sigma_min/max for base config.
        s_min = float(cfg.params.sample_sigma.params.sigma_min)
        s_max = float(cfg.params.sample_sigma.params.sigma_max)
        super().__init__(ScheduleConfig(sigma_min=s_min, sigma_max=s_max), modalities=modalities, device=device)
        self._cfg = cfg
        self.bias_strength = float(getattr(cfg.params, "bias_strength", 1.0))
        if not 0.0 <= self.bias_strength <= 1.0:
            raise ValueError("bias_strength must be in [0, 1] for composed_sigma_schedule")

    def sample_training_sigma(
        self,
        batch_size: int,
        *,
        observed_mask: Optional[ObservedMask] = None,
        bias: Optional[object] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Sigma:
        sigma_val = float(self._cfg.params.train_sigma.params.sigma)
        s_min = float(self._cfg.params.sample_sigma.params.sigma_min)
        s_max = float(self._cfg.params.sample_sigma.params.sigma_max)
        sigma: Sigma = {
            m: torch.full((batch_size,), sigma_val, device=self.device, dtype=torch.float32)
            for m in self.modalities
        }
        if bias is not None:
            for m in self.modalities:
                bias_m = self._resolve_bias(bias, m)
                # per-modality tensor of per-sample codes
                if isinstance(bias_m, torch.Tensor):
                    # expect codes: 1 = high, -1 = low, 0 = none
                    b = bias_m.to(device=self.device)
                    sigma_m = torch.full((batch_size,), sigma_val, device=self.device, dtype=torch.float32)
                    high_mask = b == 1
                    low_mask = b == -1
                    if high_mask.any():
                        target_high = sigma_val + self.bias_strength * (s_max - sigma_val)
                        sigma_m = torch.where(high_mask, float(target_high), sigma_m)
                    if low_mask.any():
                        target_low = sigma_val - self.bias_strength * (sigma_val - s_min)
                        sigma_m = torch.where(low_mask, float(target_low), sigma_m)
                    sigma[m] = sigma_m
                else:
                    if bias_m == "high":
                        target = sigma_val + self.bias_strength * (s_max - sigma_val)
                    elif bias_m == "low":
                        target = sigma_val - self.bias_strength * (sigma_val - s_min)
                    else:
                        continue
                    sigma[m] = torch.full((batch_size,), float(target), device=self.device, dtype=torch.float32)
        return self.apply_observed_mask(sigma, observed_mask)

    def sampling_sigmas(
        self,
        steps: int,
        batch_size: int,
        *,
        observed_mask: Optional[ObservedMask] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Sequence[Sigma]:
        # ignore `steps` argument, read from config (keeps single source of truth)
        steps_cfg = int(self._cfg.params.sample_sigma.params.steps)
        if steps != steps_cfg:
            # guard while wiring
            steps = steps_cfg

        s_min = float(self._cfg.params.sample_sigma.params.sigma_min)
        s_max = float(self._cfg.params.sample_sigma.params.sigma_max)

        grid = torch.linspace(s_max, s_min, steps + 1, device=self.device, dtype=torch.float32)
        sigmas: list[Sigma] = []
        for i in range(steps + 1):
            sigma_i: Sigma = {m: grid[i].expand(batch_size) for m in self.modalities}  # [B]
            sigmas.append(self.apply_observed_mask(sigma_i, observed_mask))
        return sigmas
