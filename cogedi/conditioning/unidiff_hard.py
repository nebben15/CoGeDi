from __future__ import annotations

from cogedi.conditioning.base import BaseConditioningPolicy, ConditioningContext
from cogedi.dtypes import State, Sigma


class UniDiffHardConditioning(BaseConditioningPolicy):
    """
    UniDiff-style hard conditioning:
      - If observed, pass clean value and set sigma=0.
      - Clamp observed modalities after each step.
    """
    name = "unidiff_hard_conditioning"

    def apply(self, x: State, sigma: Sigma, ctx: ConditioningContext) -> tuple[State, Sigma]:
        if not ctx.observed_mask:
            return x, sigma
        if ctx.observed is None:
            raise ValueError("ctx.observed must be provided when observed_mask is set")

        x_out: State = dict(x)
        sigma_out: Sigma = dict(sigma)
        for m, is_obs in ctx.observed_mask.items():
            if is_obs:
                x_out[m] = ctx.observed[m]
                sigma_out[m] = sigma_out[m].new_zeros(sigma_out[m].shape)  # [B]
        return x_out, sigma_out

    def clamp(self, x: State, ctx: ConditioningContext) -> State:
        if not ctx.observed_mask:
            return x
        if ctx.observed is None:
            raise ValueError("ctx.observed must be provided when observed_mask is set")

        x_out: State = dict(x)
        for m, is_obs in ctx.observed_mask.items():
            if is_obs:
                x_out[m] = ctx.observed[m]
        return x_out
