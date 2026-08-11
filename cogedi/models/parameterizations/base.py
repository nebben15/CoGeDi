from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict

import torch

from cogedi.dtypes import State, Sigma
from cogedi.forward.base import BaseForwardProcess


class BaseParametrization(abc.ABC):
    """
    Abstract interface for diffusion prediction parameterizations.

    Conventions:
      - Operates in normalized space.
      - Uses per-modality State and Sigma dicts.
      - sigma[mod] has shape [B].
      - All state tensors have leading batch dim [B, ...].

    Responsibilities:
      (1) Define the training target.
      (2) Convert model predictions into x0_hat for solvers.
    """

    name: str  # e.g. "eps", "x0", "v", "score"

    def __init__(self, *, forward: BaseForwardProcess, **kwargs):
        self.forward = forward

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def target(self, *, x0: State, eps: State, sigma: Sigma) -> State:
        """
        Compute the training target for each modality.
        """

    # ------------------------------------------------------------------
    # Sampling / solver interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def pred_to_x0(self, *, x: State, pred: State, sigma: Sigma) -> State:
        """
        Convert model prediction to x0_hat (normalized).
        """

    # ------------------------------------------------------------------
    # Convenience conversions
    # ------------------------------------------------------------------
    # def pred_to_eps(self, *, x: State, pred: State, sigma: Sigma) -> State:
    #     """
    #     Convert prediction to eps_hat via x0_hat.
    #     """
    #     x0_hat = self.pred_to_x0(x=x, pred=pred, sigma=sigma)
    #     eps_hat: State = {}
    #     for m, x_m in x.items():
    #         s = sigma[m]
    #         while s.ndim < x_m.ndim:
    #             s = s.unsqueeze(-1)
    #         eps_hat[m] = (x_m - x0_hat[m]) / s
    #     return eps_hat

    # def pred_to_score(self, *, x: State, pred: State, sigma: Sigma) -> State:
    #     """
    #     Convert prediction to score estimate via x0_hat.
    #     """
    #     x0_hat = self.pred_to_x0(x=x, pred=pred, sigma=sigma)
    #     score_hat: State = {}
    #     for m, x_m in x.items():
    #         s = sigma[m]
    #         while s.ndim < x_m.ndim:
    #             s = s.unsqueeze(-1)
    #         score_hat[m] = (x0_hat[m] - x_m) / (s * s)
    #     return score_hat
