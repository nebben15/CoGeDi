from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from cogedi.models.utils.mp import MPConv, mp_silu, normalize, mp_sum
from cogedi.models.backbones.base import BaseBackbone

class GeoFusionAttentionBackbone(BaseBackbone):

    name = "geofusion_backbone_cross_attn"

    def __init__(
        self,
        modality_dims: Dict[str, int],
        *,
        token_dim: int = 128,
        sigma_dim: int = 128,
        stem_depth: int = 4,
        fusion_depth: int = 2,
        head_depth: int = 2,
        num_heads: int = 4,
        res_balance: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.sigma_dim = sigma_dim
        self.num_modalities = len(modality_dims)
        self.res_balance = res_balance

        # ---------- modality embeddings ----------
        self.modality_emb = nn.Parameter(
            torch.randn(self.num_modalities, token_dim)
        )

        # ---------- GeomDist blocks ----------
        def make_mlp_stack(depth):
            blocks = nn.ModuleList()
            for _ in range(depth):
                blocks.append(nn.ModuleList([
                    MPConv(token_dim, token_dim, []),
                    MPConv(token_dim, token_dim, []),
                    MPConv(sigma_dim, token_dim, []),
                ]))
            return blocks

        self.stems = nn.ModuleList([
            make_mlp_stack(stem_depth)
            for _ in range(self.num_modalities)
        ])

        self.heads = nn.ModuleList([
            make_mlp_stack(head_depth)
            for _ in range(self.num_modalities)
        ])

        self.final_layers = nn.ModuleList([
            nn.ModuleList([
                MPConv(token_dim, token_dim, []),
                MPConv(token_dim, token_dim, []),
                MPConv(sigma_dim, token_dim, []),
            ])
            for _ in range(self.num_modalities)
        ])

        # ---------- cross-attention fusion ----------
        self.fusion_attn = nn.ModuleList([
            nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
            for _ in range(fusion_depth)
        ])

        self.fusion_sigma = nn.ModuleList([
            MPConv(sigma_dim, token_dim, [])
            for _ in range(fusion_depth)
        ])

        # ---------- sigma fusion projection ----------
        self.proj_t_fusion = nn.Linear(
            self.num_modalities * sigma_dim,
            sigma_dim,
            bias=True
        )

        # ---------- per-modality fusion projection ----------
        self.proj_x_head = nn.ModuleList([
            nn.Linear(token_dim, token_dim)
            for _ in range(self.num_modalities)
        ])

        # ---------- gains ----------
        self.gains_stems = nn.ParameterList([
            nn.Parameter(torch.zeros(())) for _ in range(stem_depth)
        ])
        self.gains_heads = nn.ParameterList([
            nn.Parameter(torch.zeros(())) for _ in range(head_depth)
        ])
        self.gains_fusion = nn.ParameterList([
            nn.Parameter(torch.zeros(())) for _ in range(fusion_depth)
        ])

        self.final_emb_gain = nn.Parameter(torch.zeros(()))
        self.final_out_gain = nn.Parameter(torch.zeros(()))

        # ---------- gates ----------
        self.fusion_gates = nn.Parameter(torch.zeros(self.num_modalities))

    def forward(self, tokens, *, sigma_emb, meta):

        B, M, D = tokens.shape
        assert M == self.num_modalities

        x = tokens
        t = sigma_emb

        # =====================
        # add modality embeddings
        # =====================
        x = x + self.modality_emb.unsqueeze(0)

        # =====================
        # STEMS
        # =====================
        x_stem = torch.empty_like(x)

        for i, stem in enumerate(self.stems):
            x_i = x[:, i, :]
            t_i = t[:, i, :]

            for (x_pre, x_post, emb), gain in zip(stem, self.gains_stems):
                c = emb(t_i, gain=gain) + 1
                x_i = normalize(x_i)
                y = x_pre(mp_silu(x_i))
                y = mp_silu(y * c.to(y.dtype))
                y = x_post(y)
                x_i = mp_sum(x_i, y, t=self.res_balance)

            x_stem[:, i, :] = x_i

        # =====================
        # FUSION (cross-attention)
        # =====================

        # concat + linear sigma fusion
        t_fusion = self.proj_t_fusion(t.flatten(1))  # [B, sigma_dim]

        x_fused = x_stem

        for attn, emb, gain in zip(
            self.fusion_attn,
            self.fusion_sigma,
            self.gains_fusion
        ):
            x_norm = normalize(x_fused)

            # self-attention over modalities
            y, _ = attn(x_norm, x_norm, x_norm)

            # sigma conditioning
            c = emb(t_fusion, gain=gain) + 1
            y = mp_silu(y * c.unsqueeze(1).to(y.dtype))

            x_fused = mp_sum(x_fused, y, t=self.res_balance)

        # =====================
        # HEADS
        # =====================
        out = torch.empty_like(x)

        for i, head in enumerate(self.heads):

            fusion_i = x_fused[:, i, :]

            g = torch.sigmoid(self.fusion_gates[i])

            x_head = x_stem[:, i, :] + g * self.proj_x_head[i](fusion_i)

            t_i = t[:, i, :]

            for (x_pre, x_post, emb), gain in zip(head, self.gains_heads):
                c = emb(t_i, gain=gain) + 1
                x_head = normalize(x_head)
                y = x_pre(mp_silu(x_head))
                y = mp_silu(y * c.to(y.dtype))
                y = x_post(y)
                x_head = mp_sum(x_head, y, t=self.res_balance)

            # final
            x_pre, x_post, emb = self.final_layers[i]
            c = emb(t_i, gain=self.final_emb_gain) + 1
            y = x_pre(mp_silu(normalize(x_head)))
            y = mp_silu(y * c.to(y.dtype))

            out[:, i, :] = x_post(y, gain=self.final_out_gain)

        return out