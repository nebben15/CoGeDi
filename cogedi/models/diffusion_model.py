from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from torch import nn

from cogedi.models.base import BaseDiffusionModel
from cogedi.dtypes import State, Sigma, DenoiseInput, DenoiseOutput, Descriptor
from cogedi.conditioning.base import BaseConditioningPolicy, ConditioningContext
from cogedi.models.parameterizations.base import BaseParametrization
from cogedi.models.embedders.base import BasePointEmbedder, BaseSigmaEmbedder, BaseDescriptorEmbedder
from cogedi.models.backbones.base import BaseBackbone
from cogedi.models.precond.base import BasePreconditioning


class DiffusionModel(BaseDiffusionModel):
    """
    Composition root for the model side, with packed internal representation.

    External:
      - forward(...) -> pred State (parametrization space) or raw F_state depending on precond usage
      - denoise(...) -> x0_hat (normalized) for solvers

    Internal packed flow:
      conditioning.apply -> precond.scale_input -> point_embed(pack) -> sigma_embed ->
      backbone(tokens) -> point_embed.unembed -> (F_state) -> either:
         EDM: x0_hat = precond.denoised(x, sigma, F_state)
         else: pred = precond.apply_output(F_state, x, sigma); x0_hat = parametrization.pred_to_x0(...)
    """

    name = "diffusion_model"

    def __init__(
        self,
        modality_dims: Dict[str, int],
        *,
        point_embedder: BasePointEmbedder,
        sigma_embedder: BaseSigmaEmbedder,
        descriptor_embedder: Optional[BaseDescriptorEmbedder],
        backbone: BaseBackbone,
        precond: BasePreconditioning,
        parametrization: BaseParametrization,
        conditioning: BaseConditioningPolicy,
    ):
        super().__init__(modality_dims=modality_dims)

        self.point_embedder = point_embedder
        self.sigma_embedder = sigma_embedder
        self.descriptor_embedder = descriptor_embedder
        self.backbone = backbone
        self.precond = precond
        self.parametrization = parametrization
        self.conditioning = conditioning
        self._backbone_accepts_descriptor = "descriptor_emb" in inspect.signature(self.backbone.forward).parameters
        if self.descriptor_embedder is not None and not self._backbone_accepts_descriptor:
            raise ValueError(
                "descriptor_embedder is enabled but backbone.forward does not accept `descriptor_emb`. "
                "Use a descriptor-aware backbone in landmarks supervision mode."
            )

    def forward(
        self,
        x: State,
        sigma: Sigma,
        descriptor: Optional[Descriptor] = None,
        *,
        observed_mask: Optional[Dict[str, bool]] = None,
        observed: Optional[State] = None,
    ) -> State:
        """
        Returns a per-modality State representing the model's raw output *before*
        parametrization conversion.

        For non-EDM preconditioning this corresponds to "pred" in parametrization space
        (often just the raw network output). For EDM it returns F_state (raw network output)
        and you typically should call denoise() to obtain x0_hat.
        """
        # 1) Apply UniDiff-style conditioning at inputs (clean observed + sigma=0)
        ctx = ConditioningContext(observed=observed, observed_mask=observed_mask)
        x_in, sigma_in = self.conditioning.apply(x, sigma, ctx)
        # 2) Precondition inputs (EDM c_in scaling; identity otherwise)
        x_scaled = self.precond.scale_input(x_in, sigma_in)

        # 3) Pack points into tokens
        packed = self.point_embedder(x_scaled)
        tokens, meta = packed.tokens, packed.meta  # [B,T,D], dict

        # 4) Embed sigma (packed + global)
        sigma_emb, _ = self.sigma_embedder(
            sigma_in,
            modalities=meta["modalities"],
        )

        # 5) Backbone
        if self.descriptor_embedder is None:
            tokens_out = self.backbone(
                tokens,
                sigma_emb=sigma_emb,
                meta=meta,
            )
        else:
            if descriptor is None:
                raise ValueError("descriptor is required when descriptor_embedder is enabled")
            if descriptor.data.ndim != 2:
                raise ValueError(f"descriptor.data must have shape [B,N], got {tuple(descriptor.data.shape)}")
            if descriptor.data.shape[0] != tokens.shape[0]:
                raise ValueError("descriptor batch size must match tokens batch size")

            descriptor_emb, _ = self.descriptor_embedder(
                descriptor,
                modalities=meta["modalities"],
            )

            tokens_out = self.backbone(
                tokens,
                sigma_emb=sigma_emb,
                descriptor_emb=descriptor_emb,
                meta=meta,
            )

        # 6) Unpack to state-shaped raw output F
        F_state = self.point_embedder.unembed(tokens_out, meta)

        # 7) Optional post-processing of outputs (usually identity)
        out = self.precond.apply_output(F_state, x_in, sigma_in)
        return out

    def denoise(self, inp: DenoiseInput) -> DenoiseOutput:
        """
        Canonical solver entrypoint: return x0_hat (normalized).
        """
        observed = getattr(inp, "observed", None)

        # Forward pass gives raw network output in state space (F_state or pred_state).
        out_state = self.forward(
            inp.state,
            inp.sigma,
            inp.descriptor,
            observed_mask=inp.observed_mask,
            observed=observed,
        )

        if self.precond.supports_denoised():
            # EDM-style: out_state is treated as F, and precond computes x0_hat directly.
            x0_hat = self.precond.denoised(x=inp.state, sigma=inp.sigma, F=out_state)
        else:
            # Non-EDM: out_state is interpreted as "pred" in parametrization space.
            x0_hat = self.parametrization.pred_to_x0(
                x=inp.state,
                pred=out_state,
                sigma=inp.sigma,
            )
            
        return DenoiseOutput(x0_hat=x0_hat, aux=None)
