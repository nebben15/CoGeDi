from __future__ import annotations

from cogedi.models.parameterizations.base import BaseParametrization
from cogedi.dtypes import State, Sigma


class X0Parametrization(BaseParametrization):
    """
    x0-prediction:
      target = x0
      pred_to_x0 is identity
    """
    name = "x0_parametrization"

    def target(self, *, x0: State, eps: State, sigma: Sigma) -> State:
        return x0

    def pred_to_x0(self, *, x: State, pred: State, sigma: Sigma) -> State:
        return pred
