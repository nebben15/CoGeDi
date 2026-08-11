from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import torch

from cogedi.dtypes import State, Sigma


class BasePreconditioning(abc.ABC):
    """
    Preconditioning interface.

    - scale_input: applied BEFORE embedding/backbone to normalize magnitudes across sigma.
    - apply_output: optional post-processing of backbone outputs (default no-op).
    - denoised: optional EDM-style denoiser that returns x0_hat directly using skip/out.

    Design:
      * If supports_denoised() is True, DiffusionModel may bypass parametrization
        and use denoised(x, sigma, F) to produce x0_hat.
      * Otherwise, DiffusionModel uses parametrization.pred_to_x0(...).
    """

    name: str

    def supports_denoised(self) -> bool:
        return False

    @abc.abstractmethod
    def scale_input(self, x: State, sigma: Sigma) -> State:
        """Return preconditioned input state to feed into embedders/backbone."""

    def apply_output(self, pred: State, x: State, sigma: Sigma) -> State:
        """Optional post-processing of backbone output (default: identity)."""
        return pred

    def denoised(self, *, x: State, sigma: Sigma, F: State) -> State:
        """
        Optional EDM-style denoiser: return x0_hat from raw network output F.
        Only valid if supports_denoised() is True.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement denoised()")
