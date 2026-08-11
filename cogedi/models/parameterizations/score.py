from __future__ import annotations

import torch

from cogedi.models.parameterizations.base import BaseParametrization
from cogedi.dtypes import State, Sigma


class ScoreParametrization(BaseParametrization):
    """
    score-prediction (VE/sigma-space).

    target:
      For VE, an "oracle" score target for training is:
        score = -(x - x0) / sigma^2 = -eps / sigma
      (when x = x0 + sigma*eps)

    pred_to_x0:
      x0_hat = x + sigma^2 * score_hat
    """
    name = "score_parametrization"

    def _broadcast(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        return s

    def target(self, *, x0: State, eps: State, sigma: Sigma) -> State:
        out: State = {}
        for m, x0_m in x0.items():
            s = sigma[m]  # [B]
            s_b = self._broadcast(s, x0_m)
            # score = -eps / sigma   (VE)
            out[m] = -eps[m] / s_b
        return out

    def pred_to_x0(self, *, x: State, pred: State, sigma: Sigma) -> State:
        x0_hat: State = {}
        for m, x_m in x.items():
            s = sigma[m]  # [B]
            s2_b = self._broadcast(s * s, x_m)
            x0_hat[m] = x_m + s2_b * pred[m]
        return x0_hat
