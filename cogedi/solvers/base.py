from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch

from cogedi.dtypes import State, Sigma, ObservedMask


# A denoiser function: input noisy state + per-modality sigma -> x0 estimate
DenoiseFn = Callable[[State, Sigma, Optional[ObservedMask]], State]


@dataclass(frozen=True)
class SolverConfig:
    """
    Generic solver configuration.

    Specific solvers can subclass this dataclass or ignore fields they don't use.
    """
    steps: int
    # Optional stochasticity knobs (EDM-style)
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0


@dataclass(frozen=True)
class SampleRequest:
    """
    What the solver needs to generate a sample.
    """
    # If provided, solver starts from this state. Otherwise it should sample noise internally.
    x_init: Optional[State]

    # Per-modality sigma trajectory. Length = steps +1 (EDM style).
    # Each element is a Sigma dict with tensors of shape [B].
    sigmas: Sequence[Sigma]

    # Which modalities are observed (hard/soft conditioning handled externally).
    observed_mask: Optional[ObservedMask] = None

    # Random generator for reproducibility (optional)
    generator: Optional[torch.Generator] = None


@dataclass(frozen=True)
class SampleResult:
    """
    Output of the solver.
    """
    x_final: State                  # final state (typically near sigma=0)
    x0_hat_final: Optional[State] = None  # optional final denoised estimate
    traj: Optional[Sequence[State]] = None  # optional trajectory for debugging / transformation from Gaussian to geometry


class BaseSolver(abc.ABC):
    """
    Abstract base class for ODE-based samplers (Euler/Heun/...).

    Contract:
      - Operates in sigma-space.
      - Uses dict-based State and Sigma.
      - Calls a denoise function that returns x0_hat (normalized space).
    """

    name: str

    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def step(
        self,
        *,
        x: State,
        sigma: Sigma,
        sigma_next: Sigma,
        denoise_fn: DenoiseFn,
        observed_mask: Optional[ObservedMask] = None,
        generator: Optional[torch.Generator] = None,
    ) -> State:
        """Advance one step from sigma -> sigma_next."""
        raise NotImplementedError

    def sample(
        self,
        request: SampleRequest,
        denoise_fn: DenoiseFn,
        *,
        return_trajectory: bool = False,
    ) -> SampleResult:
        """
        Convenience default sampler implemented via step().
        Orchestration will typically own the loop for better control (tqdm, clamps).
        """
        if request.x_init is None:
            raise ValueError("SampleRequest.x_init is None; provide x_init.")

        x = request.x_init
        traj = [x] if return_trajectory else None

        for i in range(len(request.sigmas) - 1):
            sigma = request.sigmas[i]
            sigma_next = request.sigmas[i + 1]
            x = self.step(
                x=x,
                sigma=sigma,
                sigma_next=sigma_next,
                denoise_fn=denoise_fn,
                observed_mask=request.observed_mask,
                generator=request.generator,
            )
            if return_trajectory:
                traj.append(x)

        x0_hat_final = denoise_fn(x, request.sigmas[-1], request.observed_mask)
        return SampleResult(x_final=x, x0_hat_final=x0_hat_final, traj=traj)

