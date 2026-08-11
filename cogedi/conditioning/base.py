from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from cogedi.dtypes import State, Sigma, ObservedMask


@dataclass(frozen=True)
class ConditioningContext:
    """
    Conditioning information for UniDiff-style hard conditioning.

    observed:
      Clean (normalized) data for observed modalities.
      Only modalities with observed_mask=True must be present.

    observed_mask:
      modality -> True if observed (fixed), False if generated.
    """
    observed: Optional[State] = None
    observed_mask: Optional[ObservedMask] = None


class BaseConditioningPolicy(abc.ABC):
    """
    Conditioning policy for joint diffusion models (UniDiff-style).

    Conditioning is implemented by:
      - passing clean data for observed modalities
      - setting their sigma to 0
      - ignoring solver updates for them
    """

    name: str

    @abc.abstractmethod
    def apply(
        self,
        x: State,
        sigma: Sigma,
        ctx: ConditioningContext,
    ) -> tuple[State, Sigma]:
        """
        Apply conditioning before a model/solver step.

        Returns:
          x_out: State to pass to the model
          sigma_out: Sigma to pass to the model
        """

    @abc.abstractmethod
    def clamp(
        self,
        x: State,
        ctx: ConditioningContext,
    ) -> State:
        """
        Enforce hard constraints after a solver step.

        For UniDiff:
          overwrite observed modalities with clean data.
        """
        return x
