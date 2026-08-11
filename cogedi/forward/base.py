from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch

from cogedi.dtypes import State, Sigma


@dataclass(frozen=True)
class ForwardConfig:
    """Base config for forward processes."""
    name: str = "forward_process"


class BaseForwardProcess(abc.ABC):
    """
    Forward corruption process q(x | x0, sigma).

    Invariants:
      - sigma is the public noise coordinate.
      - State/Sigma are dict-based.
    """
    name: str

    @abc.abstractmethod
    def q_sample(
        self,
        *,
        x0: State,
        sigma: Sigma,
        eps: State,
    ) -> State:
        """Return noisy x given clean x0, noise eps, and sigma."""

    @abc.abstractmethod
    def x0_from_eps(
        self,
        *,
        x: State,
        sigma: Sigma,
        eps: State,
    ) -> State:
        """Invert the corruption (given eps) to recover x0."""
