from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from cogedi.dtypes import State, Sigma, ObservedMask


@dataclass(frozen=True)
class LossOutput:
    """
    Output of a loss computation.

    loss: scalar tensor to backpropagate
    terms: optional dictionary of named loss components (for logging)
    """
    loss: torch.Tensor
    terms: Optional[Dict[str, torch.Tensor]] = None


class BaseLoss(abc.ABC):
    """
    Abstract base class for diffusion training losses.

    Invariants:
      - pred and target are State dicts with identical structure.
      - sigma is per-modality dict with tensors of shape [B].
      - observed modalities (if any) should typically be ignored.
    """

    name: str

    @abc.abstractmethod
    def __call__(
        self,
        *,
        pred: State,
        target: State,
        sigma: Sigma,
        observed_mask: Optional[ObservedMask] = None,
    ) -> LossOutput:
        """
        Compute training loss.

        Args:
          pred: model prediction (in parameterization space)
          target: training target (same space as pred)
          sigma: per-modality noise levels
          observed_mask: modality -> True if observed (should be excluded)

        Returns:
          LossOutput with scalar loss and optional diagnostics.
        """
