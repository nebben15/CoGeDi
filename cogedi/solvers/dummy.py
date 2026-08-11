from __future__ import annotations

from typing import Optional

import torch

from cogedi.solvers.base import BaseSolver, SolverConfig, DenoiseFn
from cogedi.dtypes import State, Sigma, ObservedMask


class DummySolver(BaseSolver):
    """
    Smoke-test solver step: jump directly to x0_hat (ignores sigma_next).
    """
    name = "dummy_solver"

    def __init__(self, cfg):
        super().__init__(SolverConfig(steps=int(getattr(cfg.params, "steps", 0)) if hasattr(cfg, "params") else 0))

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
        return denoise_fn(x, sigma, observed_mask)

