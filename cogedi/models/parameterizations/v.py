from __future__ import annotations

import torch

from cogedi.models.parameterizations.base import BaseParametrization
from cogedi.dtypes import State, Sigma


class VParametrization(BaseParametrization):
    """
    v-prediction (sigma-space, VE-friendly default).

    We use the common definition (with sigma as noise level):
      alpha(sigma) = 1 / sqrt(1 + sigma^2)
      v = alpha * eps - sigma * x0

    Then:
      x = x0 + sigma * eps

    Solve for x0 given (x, v, sigma):
      eps = (v + sigma * x0) / alpha
      x = x0 + sigma*(v + sigma*x0)/alpha
      => x0 * (1 + sigma^2/alpha) = x - (sigma/alpha) v
      => x0 = (x - (sigma/alpha) v) / (1 + sigma^2/alpha)

    This is a convenient sigma-based v.
    """
    name = "v_parametrization"

    def _broadcast(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        return s

    def _alpha(self, s: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(1.0 + s * s)

    def target(self, *, x0: State, eps: State, sigma: Sigma) -> State:
        out: State = {}
        for m, x0_m in x0.items():
            s = sigma[m]            # [B]
            a = self._alpha(s)      # [B]
            a_b = self._broadcast(a, x0_m)
            s_b = self._broadcast(s, x0_m)
            out[m] = a_b * eps[m] - s_b * x0_m
        return out

    def pred_to_x0(self, *, x: State, pred: State, sigma: Sigma) -> State:
        x0_hat: State = {}
        for m, x_m in x.items():
            s = sigma[m]            # [B]
            a = self._alpha(s)      # [B]

            s_b = self._broadcast(s, x_m)
            a_b = self._broadcast(a, x_m)

            denom = 1.0 + (s * s) / a   # [B]
            denom_b = self._broadcast(denom, x_m)

            x0_hat[m] = (x_m - (s_b / a_b) * pred[m]) / denom_b
        return x0_hat
