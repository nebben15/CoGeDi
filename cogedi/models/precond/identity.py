from __future__ import annotations

from cogedi.models.precond.base import BasePreconditioning
from cogedi.dtypes import State, Sigma


class IdentityPreconditioning(BasePreconditioning):
    name = "identity_preconditioning"

    def __init__(self, **kwargs):
        pass

    def scale_input(self, x: State, sigma: Sigma) -> State:
        return x
