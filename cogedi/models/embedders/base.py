from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from cogedi.dtypes import State, Sigma, Descriptor


@dataclass(frozen=True)
class PackedTokens:
    """Packed token representation plus metadata required to unpack."""
    tokens: torch.Tensor            # [B, T, D]
    meta: Dict[str, Any]            # modality order, lengths/slices, etc.

class BaseDescriptorEmbedder(nn.Module, metaclass=abc.ABCMeta):
    name: str
    
    @abc.abstractmethod
    def forward(
        self,
        descriptor: Descriptor,
        *,
        modalities: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            desc_emb: [B, E]  (E = embedded_dim)
            meta: optional info (e.g., modality order)
        """

class BasePointEmbedder(nn.Module, metaclass=abc.ABCMeta):
    name: str

    @abc.abstractmethod
    def forward(self, x: State) -> PackedTokens:
        """Pack State -> tokens."""

    @abc.abstractmethod
    def unembed(self, tokens: torch.Tensor, meta: Dict[str, Any]) -> State:
        """Unpack tokens -> State."""


class BaseSigmaEmbedder(nn.Module, metaclass=abc.ABCMeta):
        name: str

        @abc.abstractmethod
        def forward(
                self,
                sigma: Sigma,
                *,
                modalities: Sequence[str],
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
                """
                Returns:
                    sigma_emb: [B, M, E]  (E = embedded_dim)
                    meta: optional info (e.g., modality order)
                """
