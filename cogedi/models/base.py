from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from cogedi.dtypes import State, Sigma, DenoiseInput, DenoiseOutput, Descriptor


@dataclass(frozen=True)
class NormalizationStats:
    mean: torch.Tensor  # [D]
    std: torch.Tensor   # [D]


class BaseGeometryModel(nn.Module, metaclass=abc.ABCMeta):
    """
    Base for multi-modality geometry diffusion models.

    Convention:
      - The model operates in *normalized* coordinates.
      - Per-modality normalization stats are stored as buffers and saved in checkpoints.
    """

    def __init__(self, modality_dims: Mapping[str, int]):
        super().__init__()
        if not modality_dims:
            raise ValueError("modality_dims must be non-empty")

        self.modality_dims: Dict[str, int] = dict(modality_dims)

        for name, dim in self.modality_dims.items():
            if dim <= 0:
                raise ValueError(f"Dimension for modality '{name}' must be > 0, got {dim}")
            self.register_buffer(f"{name}_mean", torch.full((dim,), float("nan")))
            self.register_buffer(f"{name}_std", torch.full((dim,), float("nan")))

    @property
    def modalities(self) -> Tuple[str, ...]:
        return tuple(self.modality_dims.keys())

    def modality_dim(self, modality: str) -> int:
        if modality not in self.modality_dims:
            raise KeyError(f"Unknown modality '{modality}'. Known: {list(self.modality_dims)}")
        return self.modality_dims[modality]

    # --- normalization stats (checkpointed) ---
    def set_normalization_stats(self, modality: str, mean, std) -> None:
        dim = self.modality_dim(modality)
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)

        if mean_t.shape != (dim,) or std_t.shape != (dim,):
            raise ValueError(f"Stats for '{modality}' must have shape ({dim},)")
        if torch.any(std_t <= 0):
            raise ValueError(f"Std must be positive for modality '{modality}'")

        buf_mean = getattr(self, f"{modality}_mean")
        buf_std = getattr(self, f"{modality}_std")
        buf_mean.data.copy_(mean_t.to(buf_mean.device))
        buf_std.data.copy_(std_t.to(buf_std.device))

    def get_normalization_stats(self, modality: str) -> NormalizationStats:
        self.modality_dim(modality)
        mean = getattr(self, f"{modality}_mean")
        std = getattr(self, f"{modality}_std")
        if torch.isnan(mean).any() or torch.isnan(std).any():
            raise RuntimeError(f"Normalization stats for '{modality}' are not set (still NaN).")
        return NormalizationStats(mean=mean, std=std)

    def normalize(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        stats = self.get_normalization_stats(modality)
        return (x - stats.mean) / stats.std

    def denormalize(self, x: torch.Tensor, modality: str, strength: float = 1.0) -> torch.Tensor:
        stats = self.get_normalization_stats(modality)
        s = float(strength)
        if s < 0.0 or s > 1.0:
            raise ValueError(f"denormalize strength must be in [0, 1], got {strength}")

        full = x * stats.std + stats.mean
        if s == 1.0:
            return full
        if s == 0.0:
            return x
        # Interpolate between normalized (s=0) and fully denormalized (s=1).
        return x + (full - x) * s


class BaseDiffusionModel(BaseGeometryModel, metaclass=abc.ABCMeta):
    """
    Diffusion model interface.

    Solvers call `denoise(...)` (canonical).
    `forward(...)` can return any raw prediction; wrappers/adapters interpret it.
    """

    @abc.abstractmethod
    def forward(
        self,
        x: State,
        sigma: Sigma,
        descriptor: Optional[Descriptor] = None,
        *,
        observed_mask: Optional[Dict[str, bool]] = None,
    ) -> object:
        """Return raw model prediction (eps/v/x0/score depending on parameterization)."""

    @abc.abstractmethod
    def denoise(self, inp: DenoiseInput) -> DenoiseOutput:
        """Return x0_hat (normalized space) for each modality."""
