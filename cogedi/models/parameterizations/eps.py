from __future__ import annotations

from cogedi.models.parameterizations.base import BaseParametrization
from cogedi.dtypes import State, Sigma


class EpsParametrization(BaseParametrization):
    """
    ε-prediction:
      target = eps
      x0_hat computed via forward process inversion (x0_from_eps)
    """
    name = "eps_parametrization"

    def target(self, *, x0: State, eps: State, sigma: Sigma) -> State:
        return eps

    def pred_to_x0(self, *, x: State, pred: State, sigma: Sigma) -> State:
        # pred is interpreted as eps_hat
        return self.forward.x0_from_eps(x=x, sigma=sigma, eps=pred)
