from __future__ import annotations

import torch

from cogedi.models.precond.base import BasePreconditioning
from cogedi.dtypes import State, Sigma


class EDMPreconditioning(BasePreconditioning):
    """
    EDM preconditioning (Karras et al.).

    Coefficients (per batch element):
      c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
      c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
      c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
      c_noise = 0.25 * log(sigma)

    Denoiser form:
      x0_hat = c_skip * x + c_out * F
    where F is the raw network output (same shape as x).
    """
    name = "edm_preconditioning"

    def __init__(self, sigma_data: float = 1.0, eps: float = 1e-12, **kwargs):
        self.sigma_data = float(sigma_data)
        self.eps = float(eps)

    def supports_denoised(self) -> bool:
        return True

    # ---- helpers ----
    def _broadcast(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # s: [B] -> [B,1,1,...] to match x
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        return s

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        sd = self.sigma_data
        return 1.0 / torch.sqrt(sigma * sigma + sd * sd + self.eps)

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        sd = self.sigma_data
        return (sd * sd) / (sigma * sigma + sd * sd + self.eps)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        sd = self.sigma_data
        return (sigma * sd) / torch.sqrt(sigma * sigma + sd * sd + self.eps)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma must be > 0 for log; clamp for safety
        sigma_safe = torch.clamp(sigma, min=self.eps)
        return 0.25 * torch.log(sigma_safe)

    # ---- interface ----
    def scale_input(self, x: State, sigma: Sigma) -> State:
        out: State = {}
        for m, xm in x.items():
            cin = self._broadcast(self.c_in(sigma[m]), xm)
            out[m] = cin * xm
        return out

    def denoised(self, *, x: State, sigma: Sigma, F: State) -> State:
        out: State = {}
        for m, xm in x.items():
            cskip = self._broadcast(self.c_skip(sigma[m]), xm)
            cout = self._broadcast(self.c_out(sigma[m]), xm)
            out[m] = cskip * xm + cout * F[m]
        return out
