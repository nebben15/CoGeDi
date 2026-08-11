from __future__ import annotations

import abc
from typing import Any, Dict

import torch
from torch import nn


class BaseBackbone(nn.Module, metaclass=abc.ABCMeta):
    """
    Backbone interface for packed-token diffusion models.

    Expected inputs (N=1 per modality):
      - tokens:   [B, M, D]  (B=batch, M=#modalities, D=token_dim)
      - sigma_emb:[B, M, E]  (E=embedded sigma dim)
      - meta:     dict with modality order and lengths

    Expected output:
      - tokens_out: [B, M, D]
    """

    name: str

    @abc.abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        *,
        sigma_emb: torch.Tensor,
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        """Return transformed tokens with the same shape as `tokens`."""