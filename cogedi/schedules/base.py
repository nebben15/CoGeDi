from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from cogedi.dtypes import Sigma, ObservedMask


@dataclass(frozen=True)
class ScheduleConfig:
    """
    Base configuration for sigma schedules.
    """
    sigma_min: float
    sigma_max: float


class BaseSigmaSchedule(abc.ABC):
    """
    Abstract base class for sigma schedules.

    Responsibilities:
      (1) Sample training sigmas per modality.
      (2) Construct sigma grids for sampling (EDM-style).
      (3) Apply conditioning/clamping policies.

    Invariants:
      - Uses sigma-space.
      - Returns Sigma dicts (modality -> tensor[B]).
      - Sampling grids have length steps + 1.
      - sampling_sigmas(...)[-1] == sigma_min (unless overridden).
    """

    name: str

    def __init__(
        self,
        cfg: ScheduleConfig,
        modalities: Sequence[str],
        device: torch.device | str,
    ):
        self.cfg = cfg
        self.modalities = tuple(modalities)
        self.device = torch.device(device)

        if self.cfg.sigma_min < 0:
            raise ValueError("sigma_min must be >= 0")
        if self.cfg.sigma_max <= 0:
            raise ValueError("sigma_max must be > 0")
        if self.cfg.sigma_min >= self.cfg.sigma_max:
            raise ValueError("sigma_min must be < sigma_max")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def sample_training_sigma(
        self,
        batch_size: int,
        *,
        observed_mask: Optional[ObservedMask] = None,
        bias: Optional[object] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Sigma:
        """
        Sample per-modality sigmas for training.

        Returns:
          Sigma dict where each value has shape [B].
        """

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def sampling_sigmas(
        self,
        steps: int,
        batch_size: int,
        *,
        observed_mask: Optional[ObservedMask] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Sequence[Sigma]:
        """
        Construct sigma grid for sampling.

        Must return a sequence of length steps + 1:
          sigmas[0]   = sigma_max
          sigmas[-1]  = sigma_min
        """

    # ------------------------------------------------------------------
    # Conditioning hook
    # ------------------------------------------------------------------
    def apply_observed_mask(
        self,
        sigma: Sigma,
        observed_mask: Optional[ObservedMask],
    ) -> Sigma:
        """
        Default hard-conditioning behavior:
          if observed_mask[mod] is True -> sigma[mod] = 0

        Override for soft conditioning or alternative policies.
        """
        if not observed_mask:
            return sigma

        out: Sigma = {}
        for m, s in sigma.items():
            if observed_mask.get(m, False):
                out[m] = torch.zeros_like(s)
            else:
                out[m] = s
        return out

    # ------------------------------------------------------------------
    # Bias helpers
    # ------------------------------------------------------------------
    def _resolve_bias(self, bias: Optional[object], modality: str) -> Optional[str]:
        if bias is None:
            return None
        if isinstance(bias, str):
            return bias
        if isinstance(bias, dict):
            return bias.get(modality)
        raise TypeError("bias must be None, a string, or a dict of modality -> bias")
