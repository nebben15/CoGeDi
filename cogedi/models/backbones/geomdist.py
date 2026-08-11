from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from cogedi.models.utils.mp import MPConv, mp_silu, normalize, mp_sum
from cogedi.models.backbones.base import BaseBackbone


class GeomDistBackbone(BaseBackbone):
    """
    GeomDist-style MLP backbone adapted to the packed-token interface.

        Expects (N=1 per modality):
            - tokens: [B, M, D] where M = #modalities
            - sigma_emb: [B, M, E] per-modality sigma embeddings

        Behavior:
            - Concatenates all modality tokens into a single vector [B, M*D]
            - Projects to [B, D] (token_dim) to match the original GeomDist design
            - Concatenates per-modality sigma embeddings into [B, M*E] and projects to [B, D]
            - Runs the standard GeomDist MLP blocks (modulation via projected sigma)
    """

    name = "geomdist_backbone"

    def __init__(
        self,
        modality_dims: Dict[str, int],
        *,
        token_dim: int = 128,
        depth: int = 6,
        res_balance: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.res_balance = float(res_balance)

        self.gains = nn.ParameterList([
            torch.nn.Parameter(torch.zeros([])) for _ in range(depth)
        ])

        self.layers = nn.ModuleList([
            nn.ModuleList([
                MPConv(self.token_dim, self.token_dim, []),
                MPConv(self.token_dim, self.token_dim, []),
                MPConv(self.token_dim, self.token_dim, []),
            ]) for _ in range(depth)
        ])

        self.final_emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.final_out_gain = torch.nn.Parameter(torch.zeros([]))
        self.final_layer = nn.ModuleList([
            MPConv(self.token_dim, self.token_dim, []),
            MPConv(self.token_dim, self.token_dim, []),
            MPConv(self.token_dim, self.token_dim, []),
        ])

        # Project concatenated modality tokens and sigmas back to token_dim.
        self.point_proj = nn.LazyLinear(self.token_dim, bias=True)
        self.sigma_proj = nn.LazyLinear(self.token_dim, bias=True)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        sigma_emb: torch.Tensor,
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B,T,D], got {tuple(tokens.shape)}")

        B, M, D = tokens.shape
        if D != self.token_dim:
            raise ValueError(f"Token dim mismatch: got {D}, expected {self.token_dim}")

        if sigma_emb.ndim != 3:
            raise ValueError(f"Expected sigma_emb shape [B,M,E], got {tuple(sigma_emb.shape)}")
        if sigma_emb.shape[0] != B:
            raise ValueError("sigma_emb batch size must match tokens")

        # Concatenate all modality tokens and project to token_dim.
        x = tokens.reshape(B, M * D)
        x = self.point_proj(x)

        # Project per-modality sigmas to token_dim for modulation.
        t = self.sigma_proj(sigma_emb.reshape(B, -1))

        for (x_proj_pre, x_proj_post, emb_linear), emb_gain in zip(self.layers, self.gains):
            c = emb_linear(t, gain=emb_gain) + 1

            x = normalize(x)
            y = x_proj_pre(mp_silu(x))
            y = mp_silu(y * c.to(y.dtype))
            y = x_proj_post(y)
            x = mp_sum(x, y, t=self.res_balance)

        x_proj_pre, x_proj_post, emb_linear = self.final_layer
        c = emb_linear(t, gain=self.final_emb_gain) + 1
        y = x_proj_pre(mp_silu(normalize(x)))
        y = mp_silu(y * c.to(y.dtype))
        out = x_proj_post(y, gain=self.final_out_gain)

        # Broadcast the single mixed token back across T modalities.
        return out[:, None, :].expand(B, M, self.token_dim)
