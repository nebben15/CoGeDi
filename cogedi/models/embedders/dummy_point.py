from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from cogedi.dtypes import State
from cogedi.models.embedders.base import BasePointEmbedder, PackedTokens


class DummyPointEmbedder(BasePointEmbedder):
    """
    Packs per-modality point tensors into a single token tensor.

    Expected input per modality: [B, 1, D]
    Output tokens: [B, M, D]

    Dummy constraint:
      - All modalities must share the same feature dim D so we can concatenate.
      - No learned projection; this is pure packing/unpacking.

    Meta contains:
      - modalities: list[str] in packing order
      - lengths: dict[modality -> N_m]
      - feature_dim: int
    """
    name = "dummy_point_embedder"

    def __init__(self, modality_dims: Dict[str, int], **kwargs):
        super().__init__()
        self.modality_dims = dict(modality_dims)

    def forward(self, x: State) -> PackedTokens:
        if not x:
            raise ValueError("Empty State passed to DummyPointEmbedder")

        modalities = list(x.keys())
        B = None
        D = None

        lengths: Dict[str, int] = {}
        tokens_list = []

        for m in modalities:
            xm = x[m]
            if xm.ndim != 3:
                raise ValueError(f"Expected x['{m}'] shape [B,N,D], got {tuple(xm.shape)}")

            if B is None:
                B = xm.shape[0]
            elif xm.shape[0] != B:
                raise ValueError("All modalities must share the same batch size")

            if D is None:
                D = xm.shape[2]
            elif xm.shape[2] != D:
                raise ValueError(
                    "DummyPointEmbedder requires equal feature dim across modalities. "
                    f"Got D={xm.shape[2]} for '{m}', expected D={D}."
                )

            if xm.shape[1] != 1:
                raise ValueError(f"DummyPointEmbedder expects N=1, got N={xm.shape[1]} for '{m}'")
            lengths[m] = xm.shape[1]
            tokens_list.append(xm)

        tokens = torch.cat(tokens_list, dim=1)  # [B, M, D]

        meta: Dict[str, Any] = {
            "modalities": modalities,
            "lengths": lengths,
            "feature_dim": int(D),
        }
        return PackedTokens(tokens=tokens, meta=meta)

    def unembed(self, tokens: torch.Tensor, meta: Dict[str, Any]) -> State:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B,T,D], got {tuple(tokens.shape)}")

        modalities = meta["modalities"]
        lengths = meta["lengths"]

        out: State = {}
        start = 0
        for m in modalities:
            n = int(lengths[m])
            out[m] = tokens[:, start:start + n, :]
            start += n

        if start != tokens.shape[1]:
            raise ValueError(
                f"Unembed consumed {start} tokens but tokens has T={tokens.shape[1]}."
            )
        return out
