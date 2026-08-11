# This code is adapted from https://github.com/facebookresearch/DiT/tree/main

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
from timm.models.vision_transformer import Attention, Mlp

from cogedi.models.backbones.base import BaseBackbone


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
    if scale.ndim == 2:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning per token."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final adaLN layer with linear projection per token."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTBackbone(BaseBackbone):
    """DiT-style backbone adapted to packed tokens.

    Expects:
        - tokens: [B, M, D]
        - sigma_emb: [B, M, E]

    Behavior:
        - Project tokens and sigma embeddings to width D and sum.
        - Add a learned modality embedding per token.
        - Use per-token conditioning for adaLN.
    """

    name = "dit_backbone"

    def __init__(
        self,
        modality_dims: Dict[str, int],
        *,
        token_dim: int = 256,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_tokens: int = 16,
        use_pos_emb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.use_pos_emb = bool(use_pos_emb)
        self.max_tokens = int(max_tokens)

        if self.use_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, self.token_dim), requires_grad=False)
        else:
            self.register_parameter("pos_embed", None)

        self.mod_embed = nn.Parameter(torch.zeros(1, self.max_tokens, self.token_dim))
        self.token_proj = nn.Linear(self.token_dim, self.token_dim, bias=True)
        self.sigma_proj = nn.LazyLinear(self.token_dim, bias=True)
        self.cond_proj = nn.LazyLinear(self.token_dim, bias=True)

        self.blocks = nn.ModuleList([
            DiTBlock(self.token_dim, self.num_heads, mlp_ratio=self.mlp_ratio) for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(self.token_dim, self.token_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                if isinstance(module.weight, UninitializedParameter):
                    return
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        sigma_emb: torch.Tensor,
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B,M,D], got {tuple(tokens.shape)}")
        if sigma_emb.ndim != 3:
            raise ValueError(f"Expected sigma_emb shape [B,M,E], got {tuple(sigma_emb.shape)}")

        B, M, D = tokens.shape
        if D != self.token_dim:
            raise ValueError(f"Token dim mismatch: got {D}, expected {self.token_dim}")
        if sigma_emb.shape[0] != B or sigma_emb.shape[1] != M:
            raise ValueError("sigma_emb must match tokens batch and modality dims")

        if M > self.max_tokens:
            raise ValueError(f"M={M} exceeds max_tokens={self.max_tokens}")

        hx = self.token_proj(tokens)
        ht = self.sigma_proj(sigma_emb)

        x = hx + ht + self.mod_embed[:, :M, :]
        if self.use_pos_emb:
            x = x + self.pos_embed[:, :M, :]

        cond = self.cond_proj(ht)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_layer(x, cond)
        return x
