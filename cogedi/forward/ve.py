from __future__ import annotations

from cogedi.forward.base import BaseForwardProcess
from cogedi.dtypes import State, Sigma


class VEForwardProcess(BaseForwardProcess):
    """ Variance-Exploding Noising Process """
    name = "ve_forward_process"

    def q_sample(self, *, x0: State, sigma: Sigma, eps: State) -> State:
        x: State = {}
        for m, x0_m in x0.items():
            s = sigma[m]
            while s.ndim < x0_m.ndim:
                s = s.unsqueeze(-1)
            x[m] = x0_m + s * eps[m]
        return x

    def x0_from_eps(self, *, x: State, sigma: Sigma, eps: State) -> State:
        x0: State = {}
        for m, x_m in x.items():
            s = sigma[m]
            while s.ndim < x_m.ndim:
                s = s.unsqueeze(-1)
            x0[m] = x_m - s * eps[m]
        return x0
