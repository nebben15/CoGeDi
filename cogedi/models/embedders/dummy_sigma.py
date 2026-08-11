from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from cogedi.dtypes import Sigma
from cogedi.models.embedders.base import BaseSigmaEmbedder


class DummySigmaEmbedder(BaseSigmaEmbedder):
    """
    Packs per-modality sigma dict into a tensor.

    Input:  sigma[m] shape [B]
        Output:
            - sigma_emb: [B, M, 1]   (aligned with `modalities` order)
            - meta: {"modalities": [...]} 
    """
    name = "dummy_sigma_embedder"

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        sigma: Sigma,
        *,
        modalities: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not modalities:
            raise ValueError("modalities must be non-empty")

        B = None
        cols = []
        for m in modalities:
            if m not in sigma:
                raise KeyError(f"Missing sigma for modality '{m}'")
            s = sigma[m]
            if s.ndim != 1:
                raise ValueError(f"Expected sigma['{m}'] shape [B], got {tuple(s.shape)}")

            if B is None:
                B = s.shape[0]
            elif s.shape[0] != B:
                raise ValueError("All sigma tensors must share the same batch size")

            cols.append(s.unsqueeze(1))  # [B,1]

        sigma_packed = torch.cat(cols, dim=1)  # [B,M]
        sigma_emb = sigma_packed.unsqueeze(-1)  # [B,M,1]

        meta: Dict[str, Any] = {"modalities": list(modalities)}
        return sigma_emb, meta
