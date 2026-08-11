from __future__ import annotations

from typing import Dict, Optional

import torch

from cogedi.dtypes import ObservedMask, Sigma, State
from cogedi.losses.base import BaseLoss, LossOutput


class GeomDistEDMLoss(BaseLoss):
    """
    EDM-style weighted MSE loss used by the original GeomDist model.

    This adapts the weight term to the framework's explicit sigma input.
    Noise injection is handled by the forward process, not inside the loss.
    """

    name = "geomdist_edm_loss"

    def __init__(self, cfg=None, **kwargs):
        params = getattr(cfg, "params", cfg)
        sigma_data = getattr(params, "sigma_data", 1.0)
        eps = getattr(params, "eps", 1e-12)
        self.sigma_data = float(sigma_data)
        self.eps = float(eps)

    def __call__(
        self,
        *,
        pred: State,
        target: State,
        sigma: Sigma,
        observed_mask: Optional[ObservedMask] = None,
    ) -> LossOutput:
        pred_any = next(iter(pred.values()))
        device = pred_any.device

        total = torch.zeros((), device=device)
        terms: Dict[str, torch.Tensor] = {}

        any_term = False
        for m in pred.keys():
            if observed_mask and observed_mask.get(m, False):
                continue

            s = sigma[m]  # [B]
            s_safe = torch.clamp(s, min=self.eps)
            w = (s_safe * s_safe + self.sigma_data * self.sigma_data) / (s_safe * self.sigma_data) ** 2

            while w.ndim < pred[m].ndim:
                w = w.unsqueeze(-1)

            loss_m = (w * (pred[m] - target[m]).pow(2)).mean()
            terms[f"geomdist_edm/{m}"] = loss_m
            total = total + loss_m
            any_term = True

        terms["geomdist_edm/total"] = total
        if not any_term and observed_mask:
            terms["geomdist_edm/all_observed"] = torch.tensor(1.0, device=device)

        return LossOutput(loss=total, terms=terms)