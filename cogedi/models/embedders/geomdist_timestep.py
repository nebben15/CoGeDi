from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import nn

from cogedi.dtypes import Sigma
from cogedi.models.embedders.base import BaseSigmaEmbedder
from cogedi.models.utils.mp import MPConv, MPFourier, mp_silu


class GeomDistTimestepEmbedder(BaseSigmaEmbedder):
    """
    Sigma embedder modeled after the original GeomDist timestep embedding.

    Returns:
            - sigma_emb: [B, M, E] (per-modality embedded sigmas)
    """
    name = "geom_dist_timestep_embedder"

    def __init__(
        self,
        hidden_size: int = 256,
        frequency_embedding_size: int = 256,
        use_log_sigma: bool = True,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_log_sigma = bool(use_log_sigma)
        self.eps = float(eps)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.proj = nn.Identity()

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(
        self,
        sigma: Sigma,
        *,
        modalities: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not modalities:
            raise ValueError("modalities must be non-empty")

        B = None
        cols = []
        for m in modalities:
            if m not in sigma:
                raise KeyError(f"Missing sigma for modality '{m}'")
            s = sigma[m]
            if s.ndim != 1:
                raise ValueError(f"Expected sigma['{m}'] shape [B], got {tuple(s.shape)}")

            if B is None:
                B = s.shape[0]
            elif s.shape[0] != B:
                raise ValueError("All sigma tensors must share the same batch size")

            cols.append(s.unsqueeze(1))

        sigma_packed = torch.cat(cols, dim=1)  # [B, M]

        embeds = []
        for j in range(sigma_packed.shape[1]):
            s = sigma_packed[:, j]
            if self.use_log_sigma:
                s = torch.log(torch.clamp(s, min=self.eps)) / 4.0
            t_freq = self.timestep_embedding(s, self.frequency_embedding_size)
            embeds.append(self.mlp(t_freq))  # [B, E]

        sigma_emb = torch.stack(embeds, dim=1)  # [B, M, E]

        meta: Dict[str, Any] = {"modalities": list(modalities)}
        return sigma_emb, meta


class GeomDistFourierEmbedder(BaseSigmaEmbedder):
    """
    GeomDist-style Fourier sigma embedder using MPFourier + MPConv + mp_silu.

    Returns:
            - sigma_emb: [B, M, E] (per-modality embedded sigmas)
    """

    name = "geom_dist_fourier_embedder"

    def __init__(
        self,
        hidden_size: int = 256,
        use_log_sigma: bool = True,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_log_sigma = bool(use_log_sigma)
        self.eps = float(eps)
        self.emb_fourier = MPFourier(hidden_size)
        self.emb_noise = MPConv(hidden_size, hidden_size, kernel=[])
        self.proj = nn.Identity()

    def forward(
        self,
        sigma: Sigma,
        *,
        modalities: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not modalities:
            raise ValueError("modalities must be non-empty")

        B = None
        cols = []
        for m in modalities:
            if m not in sigma:
                raise KeyError(f"Missing sigma for modality '{m}'")
            s = sigma[m]
            if s.ndim != 1:
                raise ValueError(f"Expected sigma['{m}'] shape [B], got {tuple(s.shape)}")

            if B is None:
                B = s.shape[0]
            elif s.shape[0] != B:
                raise ValueError("All sigma tensors must share the same batch size")

            cols.append(s.unsqueeze(1))

        sigma_packed = torch.cat(cols, dim=1)  # [B, M]

        embeds = []
        for j in range(sigma_packed.shape[1]):
            s = sigma_packed[:, j]
            if self.use_log_sigma:
                s = torch.log(torch.clamp(s, min=self.eps))
            embeds.append(mp_silu(self.emb_noise(self.emb_fourier(s))))  # [B, E]

        sigma_emb = torch.stack(embeds, dim=1)  # [B, M, E]

        meta: Dict[str, Any] = {"modalities": list(modalities)}
        return sigma_emb, meta
    
