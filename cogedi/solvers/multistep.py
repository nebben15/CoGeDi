from __future__ import annotations

from typing import Optional

import torch

from cogedi.dtypes import ObservedMask, Sigma, State
from cogedi.solvers.base import BaseSolver, DenoiseFn, SolverConfig


class DPM2StyleSolver(BaseSolver):
    """
    Multistep Heun (DPM2-style) solver in sigma-space.

    - Uses previous derivative instead of second denoise call
    - 2nd order method with 1 model evaluation per step
    """

    name = "dpm2_solver"

    def __init__(self, cfg):
        params = getattr(cfg, "params", cfg)

        steps = int(getattr(params, "steps", 18))
        s_churn = float(getattr(params, "s_churn", 0.0))
        s_tmin = float(getattr(params, "s_tmin", 0.0))
        s_tmax = float(getattr(params, "s_tmax", float("inf")))
        s_noise = float(getattr(params, "s_noise", 1.0))

        super().__init__(
            SolverConfig(
                steps=steps,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
            )
        )

        # NEW: store previous derivative
        self._prev_d: Optional[State] = None

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
            raise ValueError("Empty state passed to DPM2Solver")

        ref_mod = next(iter(x.keys()))
        s_ref = sigma[ref_mod]

        # --- EDM churn (same as Heun) ---
        gamma = min(self.cfg.s_churn / max(self.cfg.steps, 1), (2 ** 0.5) - 1.0)
        if gamma > 0:
            mask = (s_ref >= self.cfg.s_tmin) & (s_ref <= self.cfg.s_tmax)
            gamma_t = torch.where(mask, torch.full_like(s_ref, gamma), torch.zeros_like(s_ref))
        else:
            gamma_t = torch.zeros_like(s_ref)

        sigma_hat: Sigma = {m: sigma[m] * (1.0 + gamma_t) for m in x.keys()}

        # --- noise injection ---
        x_hat: State = {}
        for m, x_m in x.items():
            s = sigma[m]
            s_hat = sigma_hat[m]

            s_b = self._broadcast(s, x_m)
            s_hat_b = self._broadcast(s_hat, x_m)

            noise = self._randn_like(x_m, generator) * self.cfg.s_noise
            delta = torch.sqrt(torch.clamp(s_hat_b * s_hat_b - s_b * s_b, min=0.0))

            x_hat[m] = x_m + delta * noise

        # --- single denoise ---
        x0_hat = denoise_fn(x_hat, sigma_hat, observed_mask)

        # --- current derivative ---
        d_cur: State = {}
        for m, x_m in x_hat.items():
            s_hat = sigma_hat[m]
            s_hat_b = self._broadcast(s_hat, x_m)
            d_cur[m] = (x_m - x0_hat[m]) / s_hat_b

        # --- multistep combination ---
        if self._prev_d is None:
            # first step → fallback to Euler
            d_combined = d_cur
        else:
            d_combined: State = {}
            for m in d_cur.keys():
                d_combined[m] = 1.5 * d_cur[m] - 0.5 * self._prev_d[m]

        # --- update ---
        x_next: State = {}
        for m, x_m in x_hat.items():
            step = self._broadcast(sigma_next[m] - sigma_hat[m], x_m)
            x_next[m] = x_m + step * d_combined[m]

        # --- store derivative ---
        self._prev_d = d_cur

        return x_next

    def sample(
        self,
        request,
        denoise_fn,
        *,
        return_trajectory: bool = False,
    ):
        # IMPORTANT: reset state between runs
        self._prev_d = None
        return super().sample(request, denoise_fn, return_trajectory=return_trajectory)