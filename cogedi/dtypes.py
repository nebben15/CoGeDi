# cogedi/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, TypeAlias, NamedTuple

import torch


# -----------------------------------------------------------------------------
# Core type aliases
# -----------------------------------------------------------------------------

Tensor: TypeAlias = torch.Tensor

# Multi-modality "state": modality name -> tensor with leading batch dim [B, ...]
State: TypeAlias = Dict[str, Tensor]

# Per-modality noise levels: modality name -> sigma tensor of shape [B]
Sigma: TypeAlias = Dict[str, Tensor]

# Descriptor vector: descriptor type, descriptor tensor [B, ...]
class Descriptor(NamedTuple):
    type: str
    data: Tensor

# Boolean mask for modalities: modality name -> is_observed (True means clamped/observed)
ObservedMask: TypeAlias = Dict[str, bool]


# -----------------------------------------------------------------------------
# Data containers passed between components
# -----------------------------------------------------------------------------

@dataclass(frozen=True) # tensors are not frozen -> can still be mutated
class Batch:
    """
    A training/eval batch.

    x0: clean (unnormalized or normalized—your choice, but be consistent) data per modality.
        Each tensor must have shape [B, ...].

    observed_mask: indicates which modalities are treated as observed/conditioned-on for
        conditional training or sampling. If you do unconditional training, this can be empty
        or all False.

    meta: optional additional info (ids, normalization keys, etc.)
    """
    x0: State
    observed_mask: Optional[ObservedMask] = None
    meta: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class NoisyBatch:
    """
    Result of applying the forward noising process to a clean Batch.

    x: noisy state (same shapes as x0), per modality.
    sigma: per-modality noise levels, each tensor shape [B].
    eps: the sampled Gaussian noise used to corrupt x0 (same shapes as x0), per modality.
    """
    x0: State
    x: State
    sigma: Sigma
    eps: State
    observed_mask: Optional[ObservedMask] = None
    meta: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class DenoiseInput:
    """
    Standard input to a denoiser/model call during sampling.

    state: current noisy state.
    sigma: current per-modality sigmas (shape [B]).
    observed_mask: optional mask for conditional sampling (hard/soft handled elsewhere).
    """
    state: State
    sigma: Sigma
    descriptor: Optional[Descriptor] = None
    observed_mask: Optional[ObservedMask] = None
    observed: Optional[State] = None


@dataclass(frozen=True)
class DenoiseOutput:
    """
    Standard output from a denoiser call.

    x0_hat: estimated clean state for each modality (shape [B, ...]).
    aux: optional dictionary for diagnostics (e.g., predicted eps, v, score, etc.).
    """
    x0_hat: State
    aux: Optional[Mapping[str, Tensor]] = None


# -----------------------------------------------------------------------------
# Small validation helpers
# -----------------------------------------------------------------------------

def assert_state_has_batch_dim(state: State) -> None:
    """Raise ValueError if any tensor in state is missing a batch dimension."""
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"state['{k}'] must be a torch.Tensor, got {type(v)}")
        if v.ndim < 1:
            raise ValueError(f"state['{k}'] must have a batch dim, got shape {tuple(v.shape)}")


def assert_sigma_shapes(sigma: Sigma, batch_size: int) -> None:
    """Raise ValueError if any sigma tensor isn't shape [B]."""
    for k, s in sigma.items():
        if not isinstance(s, torch.Tensor):
            raise TypeError(f"sigma['{k}'] must be a torch.Tensor, got {type(s)}")
        if s.shape != (batch_size,):
            raise ValueError(f"sigma['{k}'] must have shape ({batch_size},), got {tuple(s.shape)}")
