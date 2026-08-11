from __future__ import annotations

from typing import Dict, Optional

import torch

from cogedi.dtypes import ObservedMask, Sigma, State
from cogedi.losses.base import BaseLoss, LossOutput
from cogedi.utils.distance import MeshGeodesicDistance, build_distance_fn
from cogedi.utils.mesh import load_mesh_paths_from_cfg

class GeomDistEDMWithAuxContrastiveLoss(BaseLoss):
    """
    EDM loss + geodesic alignment + contrastive loss.
    """

    name = "geomdist_edm_with_aux_contrastive"

    def __init__(self, cfg=None, **kwargs):
        params = getattr(cfg, "params", cfg)
        full_cfg = kwargs.get("full_cfg", None)

        self.sigma_data = float(getattr(params, "sigma_data", 1.0))
        self.eps = float(getattr(params, "eps", 1e-12))

        self.lambda_align = float(getattr(params, "lambda_align", 0.05))
        self.lambda_contrast = float(getattr(params, "lambda_contrast", 0.05))
        self.margin = float(getattr(params, "margin", 0.05))
        self.sigma_max = float(getattr(params, "sigma_max", 0.5))

        self.distance_type = str(getattr(params, "distance_type", "geodesic")).lower()
        self.distance_fn = getattr(params, "distance_fn", None)
        if self.distance_fn is None:
            self.distance_fn = getattr(params, "geodesic_fn", None)
        if self.distance_fn is None:
            self.distance_fn = build_distance_fn(params)

        self.meshes = None
        if self.distance_type == "geodesic":
            if full_cfg is None:
                raise ValueError("full_cfg is required for geodesic distance")
            mesh_paths = load_mesh_paths_from_cfg(full_cfg)
            cache_all_pairs = bool(getattr(params, "cache_all_pairs", True))
            max_all_pairs = int(getattr(params, "max_all_pairs", 8000))
            chunk_size = int(getattr(params, "chunk_size", 4096))
            self.meshes = {
                mod: MeshGeodesicDistance(
                    mesh_path=path,
                    cache_all_pairs=cache_all_pairs,
                    max_all_pairs=max_all_pairs,
                    chunk_size=chunk_size,
                )
                for mod, path in mesh_paths.items()
            }

    def __call__(self, *, pred, target, sigma, observed_mask=None):

        device = next(iter(pred.values())).device
        total = torch.zeros((), device=device)
        terms = {}

        for m in pred.keys():
            if observed_mask and observed_mask.get(m, False):
                continue

            s = sigma[m]
            s_safe = torch.clamp(s, min=self.eps)

            # ===== EDM =====
            w = (s_safe**2 + self.sigma_data**2) / (
                (s_safe * self.sigma_data) ** 2
            )

            while w.ndim < pred[m].ndim:
                w = w.unsqueeze(-1)

            edm_loss = (w * (pred[m] - target[m]).pow(2)).mean()

            # ===== AUX =====
            mask = (s < self.sigma_max).float()
            if self.meshes is None:
                d_pos = self.distance_fn(pred[m], target[m])
            else:
                mesh = self.meshes.get(m)
                if mesh is None:
                    raise KeyError(f"Missing mesh for modality '{m}'")
                d_pos = self.distance_fn(pred[m], target[m], mesh)
            aux_loss = self.lambda_align * (mask * d_pos).mean()

            # ===== CONTRASTIVE =====
            target_neg = torch.roll(target[m], shifts=1, dims=0)
            if self.meshes is None:
                d_neg = self.distance_fn(pred[m], target_neg)
            else:
                mesh = self.meshes.get(m)
                if mesh is None:
                    raise KeyError(f"Missing mesh for modality '{m}'")
                d_neg = self.distance_fn(pred[m], target_neg, mesh)

            contrast = torch.clamp(
                self.margin + d_pos - d_neg,
                min=0.0
            )

            contrast_loss = self.lambda_contrast * (mask * contrast).mean()

            total_m = edm_loss + aux_loss + contrast_loss

            terms[f"edm/{m}"] = edm_loss
            terms[f"aux/{m}"] = aux_loss
            terms[f"contrast/{m}"] = contrast_loss
            total += total_m

        terms["total"] = total
        return LossOutput(loss=total, terms=terms)
