from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from cogedi.models.utils.mp import MPConv, mp_silu, normalize, mp_sum
from cogedi.models.backbones.base import BaseBackbone


class GeomDistDescriptorBackbone(BaseBackbone):
    """
    GeomDist-style MLP backbone adapted to the packed-token interface.

    Expects:
      - tokens: [B, M, D] where M = number of modalities (one token per modality)
      - sigma_emb: [B, M, E] per-modality sigma embeddings
      - descriptor_emb: [B, E] global descriptor embedding (E must match sigma_emb last dim)

    Operation:
      - Concatenates all modality tokens -> [B, M*D], projects to [B, D] (token_dim)
      - Concatenates descriptor_emb and flattened sigma_emb -> [B, E + M*E], projects to [B, D]
      - Runs depth MLP blocks; each block modulates activations with the projected conditional vector
      - Produces a single mixed token per batch and broadcasts it to [B, M, D] on output

    Notes:
      - Designed for N=1 per modality. Input shape checks in forward will raise on mismatch.
    """

    name = "geomdist_descriptor_backbone"

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
        self.cond_proj = nn.LazyLinear(self.token_dim, bias=True)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        sigma_emb: torch.Tensor,
        descriptor_emb: torch.Tensor,
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
        if descriptor_emb.ndim != 2:
            raise ValueError(f"Expected descriptor_emb shape [B,E], got {tuple(descriptor_emb.shape)}")
        if descriptor_emb.shape[0] != B:
            raise ValueError("descriptor_emb batch size must match tokens")
        if descriptor_emb.shape[1] != sigma_emb.shape[2]:
            raise ValueError("descriptor_emb dim must match sigma_emb dim for modulation")

        # Concatenate all modality tokens and project to token_dim.
        x = tokens.reshape(B, M * D)
        x = self.point_proj(x)

        # concat descriptor embedding and sigma embedding for conditional modulation
        cond = torch.cat([descriptor_emb, sigma_emb.reshape(B, -1)], dim=-1)
        # Project concatenated modality sigmas and descriptor embedding to token_dim for modulation.
        t = self.cond_proj(cond)

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
