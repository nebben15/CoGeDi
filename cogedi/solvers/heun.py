from __future__ import annotations

from typing import Optional

import torch

from cogedi.dtypes import ObservedMask, Sigma, State
from cogedi.solvers.base import BaseSolver, DenoiseFn, SolverConfig


class HeunSolver(BaseSolver):
    """
    Generic Heun (predictor-corrector) solver in sigma-space.
    """

    name = "heun_solver"

    def __init__(self, cfg):
        params = getattr(cfg, "params", cfg)
        steps = int(getattr(params, "steps", 18))
        s_churn = float(getattr(params, "s_churn", 0.0))
        s_tmin = float(getattr(params, "s_tmin", 0.0))
        s_tmax = float(getattr(params, "s_tmax", float("inf")))
        s_noise = float(getattr(params, "s_noise", 1.0))
        super().__init__(SolverConfig(steps=steps, s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise))

    @staticmethod
    def _broadcast(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        return s

    def _randn_like(self, x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

    def step(
        self,
        *,
        x: State,
        sigma: Sigma,
        sigma_next: Sigma,
        denoise_fn: DenoiseFn,
        observed_mask: Optional[ObservedMask] = None,
        generator: Optional[torch.Generator] = None,
    ) -> State:
        if not x:
            raise ValueError("Empty state passed to HeunSolver")

        ref_mod = next(iter(x.keys()))
        s_ref = sigma[ref_mod]

        gamma = min(self.cfg.s_churn / max(self.cfg.steps, 1), (2 ** 0.5) - 1.0)
        if gamma > 0:
            mask = (s_ref >= self.cfg.s_tmin) & (s_ref <= self.cfg.s_tmax)
            gamma_t = torch.where(mask, torch.full_like(s_ref, gamma), torch.zeros_like(s_ref))
        else:
            gamma_t = torch.zeros_like(s_ref)

        sigma_hat: Sigma = {}
        for m in x.keys():
            sigma_hat[m] = sigma[m] * (1.0 + gamma_t)

        x_hat: State = {}
        for m, x_m in x.items():
            s = sigma[m]
            s_hat = sigma_hat[m]
            s_b = self._broadcast(s, x_m)
            s_hat_b = self._broadcast(s_hat, x_m)
            noise = self._randn_like(x_m, generator) * self.cfg.s_noise
            delta = torch.sqrt(torch.clamp(s_hat_b * s_hat_b - s_b * s_b, min=0.0))
            x_hat[m] = x_m + delta * noise

        x0_hat = denoise_fn(x_hat, sigma_hat, observed_mask)
        d_cur: State = {}
        for m, x_m in x_hat.items():
            s_hat = sigma_hat[m]
            s_hat_b = self._broadcast(s_hat, x_m)
            d_cur[m] = (x_m - x0_hat[m]) / s_hat_b

        x_euler: State = {}
        for m, x_m in x_hat.items():
            s_hat = sigma_hat[m]
            s_next = sigma_next[m]
            step = self._broadcast(s_next - s_hat, x_m)
            x_euler[m] = x_m + step * d_cur[m]

        if torch.all(sigma_next[ref_mod] == 0):
            return x_euler

        x0_hat_next = denoise_fn(x_euler, sigma_next, observed_mask)
        d_prime: State = {}
        for m, x_m in x_euler.items():
            s_next = sigma_next[m]
            s_next_b = self._broadcast(s_next, x_m)
            d_prime[m] = (x_m - x0_hat_next[m]) / s_next_b

        x_next: State = {}
        for m, x_m in x_hat.items():
            s_hat = sigma_hat[m]
            s_next = sigma_next[m]
            step = self._broadcast(s_next - s_hat, x_m)
            x_next[m] = x_m + step * (0.5 * d_cur[m] + 0.5 * d_prime[m])

        return x_next
