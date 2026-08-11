from __future__ import annotations

from typing import Optional, Dict

import torch

from cogedi.losses.base import BaseLoss, LossOutput
from cogedi.dtypes import State, Sigma, ObservedMask


class MSELoss(BaseLoss):
    name = "mse_loss"

    def __init__(self, cfg=None):
        pass

    def __call__(
        self,
        *,
        pred: State,
        target: State,
        sigma: Sigma,
        observed_mask: Optional[ObservedMask] = None,
    ) -> LossOutput:
        # Total is connected to the computation graph even if we skip everything.
        # This prevents "does not require grad" when all modalities are observed.
        pred_any = next(iter(pred.values()))
        device = pred_any.device

        pred_sum = torch.zeros((), device=device)
        for v in pred.values():
            pred_sum = pred_sum + v.sum()

        total = 0.0 * pred_sum  # scalar with grad_fn
        terms: Dict[str, torch.Tensor] = {}

        any_term = False
        for m in pred.keys():
            if observed_mask and observed_mask.get(m, False):
                continue
            loss_m = (pred[m] - target[m]).pow(2).mean()
            terms[f"mse/{m}"] = loss_m
            total = total + loss_m
            any_term = True

        terms["mse/total"] = total

        if not any_term and observed_mask:
            terms["mse/all_observed"] = torch.tensor(1.0, device=device)

        return LossOutput(loss=total, terms=terms)
