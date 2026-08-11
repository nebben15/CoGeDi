from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from cogedi.dtypes import Descriptor
from cogedi.models.embedders.base import BaseDescriptorEmbedder

class DescriptorMLPEmbedder(BaseDescriptorEmbedder):
    """
    Embed descriptor vectors with an MLP, with explicit null-row handling.

    Null descriptors are represented by non-finite values (NaN or +/-Inf) in the
    input descriptor row. During forward:
    1) A per-row null mask is computed from finiteness.
    2) Input is sanitized (non-finite -> 0) and normalized before MLP projection.
    3) Output rows flagged as null are replaced by a learned `null_embedding`.

    This keeps output shape stable while preserving null-vs-regular semantics via
    metadata (`meta["is_null"]`).
    """

    name = "geomdist_descriptor_embedder"

    def __init__(self, hidden_size=256):
        super().__init__()

        self.eps = 1e-6

        self.net = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.null_embedding = nn.Parameter(torch.randn(hidden_size))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)

    def forward(self, desc: Descriptor, *, modalities: Sequence[str]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            desc: Descriptor named tuple where `desc.data` has shape [B, K].
                  Rows containing any non-finite value are treated as null.
            modalities: Ordered modality names associated with this descriptor.

        Returns:
            emb: Tensor [B, hidden_size]. For null rows, this is the learned
                 `null_embedding`; otherwise it is the MLP projection.
            meta: Dict with:
                - "modalities": list of modality names
                - "is_null": bool tensor [B] marking rows that used null embedding
                - "descriptor_dim": input descriptor width K
        """
        d = desc.data
        if d.ndim != 2:
            raise ValueError(f"Descriptor tensor must have shape [B, K], got {tuple(d.shape)}")

        B, K = d.shape

        row_is_null = ~torch.isfinite(d).all(dim=-1)
        d_clean = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        x = self._normalize(d_clean)
        emb = self.net(x)

        if row_is_null.any():
            null_vec = self.null_embedding.to(device=emb.device, dtype=emb.dtype).view(1, -1)
            emb = torch.where(row_is_null[:, None], null_vec, emb)

        meta: Dict[str, Any] = {
            "modalities": list(modalities),
            "is_null": row_is_null,
            "descriptor_dim": K,
        }
        return emb, meta