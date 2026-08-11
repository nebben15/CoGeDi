from typing import Any, Dict
import torch
from torch import nn

from cogedi.models.backbones.base import BaseBackbone

class DummyBackbone(BaseBackbone):
    """
    Minimal backbone: returns zeros with same shape as tokens.
    Ignores sigma inputs (but accepts them to match real signature).
    """
    name = "dummy_packed_backbone"

    def __init__(self, modality_dims: Dict[str, int], token_dim: int | None = None, **kwargs):
        super().__init__()
        # assume all modalities share same D in dummy runs; use max dim as token dim
        d = int(token_dim) if token_dim is not None else max(modality_dims.values())
        self.proj = nn.LazyLinear(d) if token_dim is None else nn.Linear(d, d)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        sigma_emb: torch.Tensor,
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        return self.proj(tokens)
