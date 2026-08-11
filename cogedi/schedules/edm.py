from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from cogedi.dtypes import Sigma, ObservedMask
from cogedi.schedules.base import BaseSigmaSchedule, ScheduleConfig


class EDMSigmaSchedule(BaseSigmaSchedule):
    """
    EDM-style sigma schedule.

    - Training sigmas are sampled from log-normal: exp(N(P_mean, P_std)).
    - Sampling sigmas follow the EDM power-law grid with exponent rho.
    """

    name = "edm_sigma_schedule"

    def __init__(self, cfg, modalities, device):
        s_min = float(cfg.params.sigma_min)
        s_max = float(cfg.params.sigma_max)
        super().__init__(ScheduleConfig(sigma_min=s_min, sigma_max=s_max), modalities=modalities, device=device)
        self._cfg = cfg

        self.rho = float(getattr(cfg.params, "rho", 7.0))
        self.P_mean = float(getattr(cfg.params, "P_mean", -1.2))
        self.P_std = float(getattr(cfg.params, "P_std", 1.2))

    def sample_training_sigma(
            self,
            batch_size,
            sigma_mode,
            clamp_prob,
            apply_prob=1.0,
            generator: Optional[torch.Generator] = None,
    ) -> Sigma:
        # catch bad config
        if sigma_mode not in ["default", "joint", "clamp", "clamp_joint"]:
            raise ValueError('sigma_mode not in "default", "joint", "clamp", "clamp_joint"')
        if clamp_prob < 0 or clamp_prob > 1:
            raise ValueError('clamp_prob must be in [0,1]')
        if apply_prob < 0 or apply_prob > 1:
            raise ValueError('apply_prob must be in [0,1]')
        
        M = len(self.modalities)

        def _sample_for_mode(mode: str) -> torch.Tensor:
            if mode in {"default", "clamp"}:
                sampled = (torch.randn(batch_size, M, generator=generator) * self.P_std + self.P_mean).exp().to(dtype=torch.float32)
            elif mode in {"joint", "clamp_joint"}:
                sig = (torch.randn(batch_size, 1, generator=generator) * self.P_std + self.P_mean).exp().to(dtype=torch.float32)
                sampled = sig.expand(batch_size, M)
            else:
                raise ValueError('sigma_mode not in "default", "joint", "clamp", "clamp_joint"')

            if mode in {"clamp", "clamp_joint"}:
                sample_flag = torch.rand(batch_size, generator=generator) < clamp_prob
                modality_flag = torch.randint(0, M, (batch_size,), generator=generator)
                clamp_mask = torch.zeros(batch_size, M, dtype=torch.bool)
                clamp_mask[sample_flag, modality_flag[sample_flag]] = True
                sampled = sampled.masked_fill(clamp_mask, 0.0)
            return sampled

        if sigma_mode == "default" or apply_prob >= 1.0:
            sigmas = _sample_for_mode(sigma_mode)
        elif apply_prob <= 0.0:
            sigmas = _sample_for_mode("default")
        else:
            default_sigmas = _sample_for_mode("default")
            mode_sigmas = _sample_for_mode(sigma_mode)
            apply_mask = (torch.rand(batch_size, generator=generator) < apply_prob).unsqueeze(1)
            sigmas = torch.where(apply_mask, mode_sigmas, default_sigmas)

        # to dict
        sigma_dict: Sigma = {}
        for i, m in enumerate(self.modalities):
            sigma_dict[m] = sigmas[:, i].to(self.device)

        return sigma_dict

    def sampling_sigmas(
        self,
        steps: int,
        batch_size: int,
        *,
        observed_mask: Optional[ObservedMask] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Sequence[Sigma]:
        if steps is None:
            raise ValueError("sampling steps must be provided explicitly")

        step_indices = torch.arange(steps, dtype=torch.float64, device=self.device)
        s_min = self.cfg.sigma_min
        s_max = self.cfg.sigma_max
        rho = self.rho
        
        # Karras et al.
        t_steps = (s_max ** (1 / rho) + step_indices / (steps - 1) * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        t_steps = t_steps.to(dtype=torch.float32)

        sigmas: list[Sigma] = []
        for i in range(steps + 1):
            sigma_i: Sigma = {m: t_steps[i].expand(batch_size) for m in self.modalities}
            sigmas.append(self.apply_observed_mask(sigma_i, observed_mask))
        return sigmas
