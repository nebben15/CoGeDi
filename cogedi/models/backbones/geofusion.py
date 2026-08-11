from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from cogedi.models.utils.mp import MPConv, mp_silu, normalize, mp_sum
from cogedi.models.backbones.base import BaseBackbone


class GeoFusionBackbone(BaseBackbone):
    """
    GeoFusion backbone inspired by GeomDist and adapted to the UniDiff joint/conditional modeling goal.

        Expects (N=1 per modality):
            - tokens: [B, M, D] where M = #modalities
            - sigma_emb: [B, M, E] per-modality sigma embeddings

        Behavior:
            - Each modality has its own stem of GeomDist-style MLP blocks, which modulate via the corresponding per-modality sigma embedding
            - Outputs of all stems are fused via a learned gated residual connection and another stack of GeomDist-style blocks, which modulate on a fusion of all sigma embeddings
            - The fused latent is then fed into a head per modality, which again consists of GeomDist-style blocks modulated on the corresponding per-modality sigma
            - Final output is produced by a last GeomDist-style block per modality, modulated on the corresponding per-modality sigma
    """

    name = "geofusion_backbone"

    def __init__(
        self,
        modality_dims: Dict[str, int],
        *,
        token_dim: int = 128,
        sigma_dim: int = 128,
        stem_depth: int = 4,
        fusion_depth: int = 4,
        head_depth: int = 2,    # does not include final block (always + 1)
        res_balance: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.sigma_dim = sigma_dim
        self.res_balance = float(res_balance)
        self.num_modalities = int(len(modality_dims))

        def make_mlp_stack(token_dim, sigma_dim, depth):
            blocks = nn.ModuleList()
            for i in range(depth):
                x_pre = MPConv(token_dim, token_dim, [])
                x_post = MPConv(token_dim, token_dim, [])
                emb = MPConv(sigma_dim, token_dim, [])
                blocks.append(nn.ModuleList([x_pre, x_post, emb]))
            return blocks
        
        # per modality stems 
        self.stems = nn.ModuleList([make_mlp_stack(token_dim=self.token_dim, sigma_dim=self.sigma_dim, depth=stem_depth) for _ in range(self.num_modalities)])
        # fusion 
        self.fusion = make_mlp_stack(token_dim=self.token_dim, sigma_dim=self.sigma_dim, depth=fusion_depth)
        # heads
        self.heads = nn.ModuleList([make_mlp_stack(token_dim=self.token_dim, sigma_dim=self.sigma_dim, depth=head_depth) for _ in range(self.num_modalities)])
        self.final_layers = nn.ModuleList([
            nn.ModuleList([MPConv(self.token_dim, self.token_dim, []), MPConv(self.token_dim, self.token_dim, []), MPConv(self.sigma_dim, self.token_dim, [])]) 
            for _ in range(self.num_modalities)
            ])
        
        # gains for sigma layer
        self.gains_stems = nn.ParameterList([
            torch.nn.Parameter(torch.zeros([])) for _ in range(stem_depth)
        ])

        self.gains_fusion = nn.ParameterList([
            torch.nn.Parameter(torch.zeros([])) for _ in range(fusion_depth)
        ])

        self.gains_heads = nn.ParameterList([
            torch.nn.Parameter(torch.zeros([])) for _ in range(head_depth)
        ])
        self.final_emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.final_out_gain = torch.nn.Parameter(torch.zeros([]))
        
        # projections
        self.proj_x_fusion = nn.Linear(self.token_dim*self.num_modalities, self.token_dim, bias=True)
        self.proj_t_fusion = nn.Linear(self.sigma_dim*self.num_modalities, self.sigma_dim, bias=True)
        # for extra expresivity, gives per modality fusion latent
        self.proj_x_head = nn.ModuleList([nn.Linear(self.token_dim, self.token_dim, bias=True) for _ in range(self.num_modalities)]) 

        # gate for heads
        self.fusion_gates = nn.Parameter(torch.zeros(self.num_modalities))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        sigma_emb: torch.Tensor,
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        # Dim checks
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens shape [B,T,D], got {tuple(tokens.shape)}")
        B, M, D = tokens.shape
        _, _, E = sigma_emb.shape
        if M != self.num_modalities:
            raise ValueError(f"Expected M={self.num_modalities} modalities, got M={M}")
        if sigma_emb.shape[1] != M:
            raise ValueError("sigma_emb modality dim must match tokens")
        if E != self.sigma_dim:
            raise ValueError(f"sigma_emb last dim E={E} must equal sigma_dim={self.sigma_dim}")
        if D != self.token_dim:
            raise ValueError(f"Token dim mismatch: got {D}, expected {self.token_dim}")
        if sigma_emb.ndim != 3:
            raise ValueError(f"Expected sigma_emb shape [B,M,E], got {tuple(sigma_emb.shape)}")
        if sigma_emb.shape[0] != B:
            raise ValueError("sigma_emb batch size must match tokens")
        # assign names
        x = tokens
        t = sigma_emb
        ### stems ###
        x_stem = torch.empty_like(x)
        for i, stem in enumerate(self.stems):
            # stem are per modality
            x_i = x[:,i,:]
            t_i = t[:,i,:]
            # apply stem_depth many GeomDist blocks
            for (x_proj_pre, x_proj_post, emb_linear), emb_gain in zip(stem, self.gains_stems):
                c = emb_linear(t_i, gain=emb_gain) + 1
                x_i = normalize(x_i)
                y = x_proj_pre(mp_silu(x_i))
                y = mp_silu(y * c.to(y.dtype))
                y = x_proj_post(y)
                x_i = mp_sum(x_i, y, t=self.res_balance)
            x_stem[:,i,:] = x_i
        
        ### fusion ###
        # x and sigmas have to be combined -> projection to fusion dim happens with the first x_proj_pre
        x_fusion = self.proj_x_fusion(x_stem.flatten(start_dim=1))
        t_fusion = self.proj_t_fusion(t.flatten(start_dim=1))
        for (x_proj_pre, x_proj_post, emb_linear), emb_gain in zip(self.fusion, self.gains_fusion):
            c = emb_linear(t_fusion, gain=emb_gain) + 1
            x_fusion = normalize(x_fusion)
            y = x_proj_pre(mp_silu(x_fusion))
            y = mp_silu(y * c.to(y.dtype))
            y = x_proj_post(y)
            x_fusion = mp_sum(x_fusion, y, t=self.res_balance)

        ### heads ###
        out = torch.empty_like(x)
        # one head per modality
        for i, head in enumerate(self.heads):
            # fusion with gated residual connection
            g = torch.sigmoid(self.fusion_gates[i])  # gate for modality i
            x_head = x_stem[:, i, :] + g * self.proj_x_head[i](x_fusion) # gated fusion
            t_head = t[:,i,:]
            # geomdist blocks
            for (x_proj_pre, x_proj_post, emb_linear), emb_gain in zip(head, self.gains_heads):
                c = emb_linear(t_head, gain=emb_gain) + 1
                x_head = normalize(x_head)
                y = x_proj_pre(mp_silu(x_head))
                y = mp_silu(y * c.to(y.dtype))
                y = x_proj_post(y)
                x_head = mp_sum(x_head, y, t=self.res_balance)
            # final block
            x_proj_pre, x_proj_post, emb_linear = self.final_layers[i]
            c = emb_linear(t_head, gain=self.final_emb_gain) + 1
            y = x_proj_pre(mp_silu(normalize(x_head)))
            y = mp_silu(y * c.to(y.dtype))
            out[:,i,:] = x_proj_post(y, gain=self.final_out_gain)
        
        return out