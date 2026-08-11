from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from cogedi.dtypes import State
from cogedi.models.embedders.base import BasePointEmbedder, PackedTokens
from cogedi.models.utils.mp import MPConv


class GeomDistPointEmbedder(BasePointEmbedder):
    """
    GeomDist-style point embedder adapted to the packed-token interface.

    - Per-modality embed: (xyz + optional extra channels) -> token_dim
    - Unembed: token_dim -> original modality dim

        Contract:
            - Input per modality: [B, 1, D]
            - Output tokens: [B, M, token_dim]
    """

    name = "geomdist_point_embedder"

    def __init__(
        self,
        modality_dims: Dict[str, int],
        *,
        token_dim: int = 128,
        hidden_dim: int = 48,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_dim % 6 != 0:
            raise ValueError("hidden_dim must be divisible by 6")

        self.modality_dims = dict(modality_dims)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)

        e = torch.pow(2, torch.arange(self.hidden_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat([e, torch.zeros(self.hidden_dim // 6), torch.zeros(self.hidden_dim // 6)]),
                torch.cat([torch.zeros(self.hidden_dim // 6), e, torch.zeros(self.hidden_dim // 6)]),
                torch.cat([torch.zeros(self.hidden_dim // 6), torch.zeros(self.hidden_dim // 6), e]),
            ]
        )
        self.register_buffer("basis", e)  # 3 x (hidden_dim/2)

        self.embed_mlps = nn.ModuleDict()
        self.unembed_mlps = nn.ModuleDict()
        for mod, dim in self.modality_dims.items():
            if dim < 3:
                raise ValueError(f"Modality '{mod}' must have dim >= 3, got {dim}")
            other_dim = dim - 3
            self.embed_mlps[mod] = MPConv(self.hidden_dim + 3 + other_dim, self.token_dim, kernel=[])
            self.unembed_mlps[mod] = MPConv(self.token_dim, dim, kernel=[])

    @staticmethod
    def _embed_coords(coords: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        projections = torch.einsum("nd,de->ne", coords, basis)
        return torch.cat([projections.sin(), projections.cos()], dim=1)

    def forward(self, x: State) -> PackedTokens:
        if not x:
            raise ValueError("Empty State passed to GeomDistPointEmbedder")

        modalities = list(x.keys())
        B = None
        tokens_list = []
        lengths: Dict[str, int] = {}

        for mod in modalities:
            xm = x[mod]
            if xm.ndim != 3:
                raise ValueError(f"Expected x['{mod}'] shape [B,N,D], got {tuple(xm.shape)}")

            if B is None:
                B = xm.shape[0]
            elif xm.shape[0] != B:
                raise ValueError("All modalities must share the same batch size")

            N, D = xm.shape[1], xm.shape[2]
            if N != 1:
                raise ValueError(f"GeomDistPointEmbedder expects N=1, got N={N} for '{mod}'")
            if D != self.modality_dims[mod]:
                raise ValueError(
                    f"Modality '{mod}' dim mismatch: got {D}, expected {self.modality_dims[mod]}"
                )

            coords = xm[..., :3].reshape(B * N, 3)
            others = xm[..., 3:].reshape(B * N, -1) if D > 3 else None

            embed = self._embed_coords(coords, self.basis)
            if others is None or others.numel() == 0:
                inp = torch.cat([embed, coords], dim=1)
            else:
                inp = torch.cat([embed, coords, others], dim=1)

            tokens = self.embed_mlps[mod](inp).reshape(B, N, self.token_dim)
            tokens_list.append(tokens)
            lengths[mod] = N

        packed = torch.cat(tokens_list, dim=1)  # [B, M, token_dim]
        meta: Dict[str, Any] = {
            "modalities": modalities,
            "lengths": lengths,
            "token_dim": self.token_dim,
        }
        return PackedTokens(tokens=packed, meta=meta)

    def unembed(self, tokens: torch.Tensor, meta: Dict[str, Any]) -> State:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B,T,D], got {tuple(tokens.shape)}")

        modalities = meta["modalities"]
        lengths = meta["lengths"]

        out: State = {}
        start = 0
        for mod in modalities:
            n = int(lengths[mod])
            tok = tokens[:, start:start + n, :]
            start += n

            B, N, D = tok.shape
            tok_flat = tok.reshape(B * N, D)
            out_flat = self.unembed_mlps[mod](tok_flat)
            out[mod] = out_flat.reshape(B, N, self.modality_dims[mod])

        if start != tokens.shape[1]:
            raise ValueError(
                f"Unembed consumed {start} tokens but tokens has T={tokens.shape[1]}."
            )
        return out