from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type, Optional

import torch
from torch import nn
from tqdm import tqdm

from cogedi.models.base import BaseDiffusionModel
from cogedi.optim.optimizers import BaseOptimizerBuilder
from cogedi.optim.schedules import BaseLRSchedule, ComposedLRSchedule
from cogedi.models.parameterizations.base import BaseParametrization
from cogedi.models.embedders.base import BasePointEmbedder, BaseSigmaEmbedder, BaseDescriptorEmbedder
from cogedi.models.backbones.base import BaseBackbone
from cogedi.models.precond.base import BasePreconditioning
from cogedi.schedules.base import BaseSigmaSchedule
from cogedi.solvers.base import BaseSolver
from cogedi.losses.base import BaseLoss
from cogedi.conditioning.base import BaseConditioningPolicy
from cogedi.forward.base import BaseForwardProcess
from cogedi.orch.denoise import GuidanceConfig, build_denoise_fn



# -----------------------------------------------------------------------------
# Registries
# -----------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Type[BaseDiffusionModel]] = {}
PARAM_REGISTRY: Dict[str, Type[BaseParametrization]] = {}
SCHEDULE_REGISTRY: Dict[str, Type[BaseSigmaSchedule]] = {}
SOLVER_REGISTRY: Dict[str, Type[BaseSolver]] = {}
LOSS_REGISTRY: Dict[str, Type[BaseLoss]] = {}
COND_REGISTRY: Dict[str, Type[BaseConditioningPolicy]] = {}
FORWARD_REGISTRY: Dict[str, Type[BaseForwardProcess]] = {}
BACKBONE_REGISTRY: Dict[str, Type[BaseBackbone]] = {}
POINT_EMBED_REGISTRY: Dict[str, Type[BasePointEmbedder]] = {}
SIGMA_EMBED_REGISTRY: Dict[str, Type[BaseSigmaEmbedder]] = {}
DESCRIPTOR_EMBED_REGISTRY: Dict[str, Type[BaseDescriptorEmbedder]] = {}
PRECOND_REGISTRY: Dict[str, Type[BasePreconditioning]] = {}
LR_SCHEDULE_REGISTRY: Dict[str, Type[BaseLRSchedule]] = {}
OPTIM_REGISTRY: Dict[str, Type[BaseOptimizerBuilder]] = {}
DATA_REGISTRY: Dict[str, Type] = {}

def _get(reg: Dict[str, Type], key: str, kind: str):
    if key not in reg:
        known = ", ".join(sorted(reg.keys())) if reg else "(empty)"
        raise KeyError(f"Unknown {kind} '{key}'. Known: {known}")
    return reg[key]


# -----------------------------------------------------------------------------
# Mode-specific build artifacts
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingArtifacts:
    model: BaseDiffusionModel
    parametrization: BaseParametrization
    forward: BaseForwardProcess
    schedule: BaseSigmaSchedule
    loss: BaseLoss
    conditioning: BaseConditioningPolicy
    optimizer: torch.optim.Optimizer
    lr_schedule: Optional[BaseLRSchedule]
    data_source: Any
    device: torch.device

@dataclass(frozen=True)
class SamplingArtifacts:
    model: BaseDiffusionModel
    parametrization: BaseParametrization
    forward: BaseForwardProcess
    schedule: BaseSigmaSchedule
    solver: BaseSolver
    conditioning: BaseConditioningPolicy
    device: torch.device

@dataclass(frozen=True)
class EvalArtifacts:
    model: BaseDiffusionModel
    parametrization: BaseParametrization
    forward: BaseForwardProcess
    schedule: BaseSigmaSchedule
    solver: BaseSolver
    loss: BaseLoss
    conditioning: BaseConditioningPolicy
    device: torch.device

# -----------------------------------------------------------------------------
# Atomic builders 
# -----------------------------------------------------------------------------

def _kwargs(ns) -> dict:
    """Convert SimpleNamespace-like objects to a plain dict safely."""
    if ns is None:
        return {}
    d = getattr(ns, "__dict__", None)
    return {} if d is None else dict(d)


def _apply_lr_scaling(cfg) -> None:
    """Apply GeomDist-style LR scaling from optim.blr and train.batch_size.

    Rules:
      - If optim.lr is set, keep it.
      - Otherwise compute lr = blr * (batch_size * accum_iter * world_size) / 128.
      - Populate train.lr for LR schedules that read train.lr when params.lr is absent.
      - Ensure train.min_lr exists (default 5e-7 unless optim.min_lr provided).
    """
    opt_cfg = getattr(cfg, "optim", None)
    train_cfg = getattr(cfg, "train", None)
    if opt_cfg is None or train_cfg is None:
        return

    lr = getattr(opt_cfg, "lr", None)
    if lr is None:
        blr = float(getattr(opt_cfg, "blr", 5e-7))
        batch_size = int(getattr(train_cfg, "batch_size", 1))
        accum_iter = int(getattr(train_cfg, "accum_iter", 1))
        run_cfg = getattr(cfg, "run", None)
        world_size = int(getattr(run_cfg, "world_size", 1)) if run_cfg is not None else 1
        eff_batch = batch_size * accum_iter * world_size
        lr = blr * eff_batch / 128.0
        setattr(opt_cfg, "lr", float(lr))

    if getattr(train_cfg, "lr", None) is None:
        setattr(train_cfg, "lr", float(getattr(opt_cfg, "lr", 1e-4)))

    if getattr(train_cfg, "min_lr", None) is None:
        min_lr = getattr(opt_cfg, "min_lr", 5e-7)
        setattr(train_cfg, "min_lr", float(min_lr))


def build_sigma_embedder(cfg) -> nn.Module:
    se_cfg = cfg.model.params.sigma_embedder
    cls = _get(SIGMA_EMBED_REGISTRY, se_cfg.type, "sigma embedder")
    return cls(**_kwargs(getattr(se_cfg, "params", None)))


def build_descriptor_embedder(cfg) -> nn.Module:
    de_cfg = cfg.model.params.descriptor_embedder
    cls = _get(DESCRIPTOR_EMBED_REGISTRY, de_cfg.type, "descriptor embedder")
    return cls(**_kwargs(getattr(de_cfg, "params", None)))


def build_point_embedder(cfg, *, modality_dims: dict[str, int]) -> nn.Module:
    pe_cfg = cfg.model.params.point_embedder
    cls = _get(POINT_EMBED_REGISTRY, pe_cfg.type, "point embedder")
    return cls(modality_dims=modality_dims, **_kwargs(getattr(pe_cfg, "params", None)))


def build_backbone(cfg, *, modality_dims: dict[str, int]) -> nn.Module:
    bb_cfg = cfg.model.params.backbone
    cls = _get(BACKBONE_REGISTRY, bb_cfg.type, "backbone")
    return cls(modality_dims=modality_dims, **_kwargs(getattr(bb_cfg, "params", None)))


def build_precond(cfg):
    pc_cfg = cfg.model.params.precond
    cls = _get(PRECOND_REGISTRY, pc_cfg.type, "preconditioning")
    return cls(**_kwargs(getattr(pc_cfg, "params", None)))

def build_device(cfg) -> torch.device:
    return torch.device(getattr(cfg.run, "device", "cpu"))

def build_forward(cfg) -> BaseForwardProcess:
    f_cfg = cfg.forward
    cls = _get(FORWARD_REGISTRY, f_cfg.type, "forward process")
    return cls(**_kwargs(getattr(f_cfg, "params", None)))


def build_parametrization(cfg, *, forward: BaseForwardProcess) -> BaseParametrization:
    p_cfg = cfg.model.params.parametrization
    cls = _get(PARAM_REGISTRY, p_cfg.type, "parametrization")
    return cls(forward=forward, **_kwargs(getattr(p_cfg, "params", None)))


def build_conditioning(cfg) -> BaseConditioningPolicy:
    c_cfg = cfg.model.params.conditioning
    cls = _get(COND_REGISTRY, c_cfg.type, "conditioning policy")
    return cls(**_kwargs(getattr(c_cfg, "params", None)))


def build_model(
    cfg,
    *,
    device: torch.device,
    parametrization: BaseParametrization,
    conditioning: BaseConditioningPolicy,
) -> BaseDiffusionModel:
    m_cfg = cfg.model
    cls = _get(MODEL_REGISTRY, m_cfg.type, "model")

    modality_dims = dict(cfg.data.modalities.__dict__)
    supervision_mode = str(getattr(getattr(cfg, "run", None), "supervision", "full")).lower()

    sigma_embedder = build_sigma_embedder(cfg)
    descriptor_embedder = None
    if supervision_mode == "landmarks":
        descriptor_cfg = getattr(cfg.model.params, "descriptor_embedder", None)
        if descriptor_cfg is None:
            raise ValueError("run.supervision='landmarks' requires model.params.descriptor_embedder")
        descriptor_embedder = build_descriptor_embedder(cfg)

    point_embedder = build_point_embedder(cfg, modality_dims=modality_dims)
    backbone = build_backbone(cfg, modality_dims=modality_dims)
    precond = build_precond(cfg)

    model = cls(
        modality_dims=modality_dims,
        sigma_embedder=sigma_embedder,
        descriptor_embedder=descriptor_embedder,
        point_embedder=point_embedder,
        backbone=backbone,
        precond=precond,
        parametrization=parametrization,
        conditioning=conditioning,
    )
    model.to(device)
    return model


def build_schedule(cfg, *, modalities, device: torch.device) -> BaseSigmaSchedule:
    s_type = cfg.schedule.type
    cls = _get(SCHEDULE_REGISTRY, s_type, "schedule")
    return cls(cfg.schedule, modalities=modalities, device=device)


def build_solver(cfg) -> BaseSolver:
    sol_type = cfg.solver.type
    cls = _get(SOLVER_REGISTRY, sol_type, "solver")
    return cls(cfg.solver)


def build_loss(cfg) -> BaseLoss:
    l_type = cfg.loss.type
    cls = _get(LOSS_REGISTRY, l_type, "loss")
    return cls(cfg.loss, full_cfg=cfg)


def build_lr_schedule(cfg) -> Optional[BaseLRSchedule]:
    opt_cfg = getattr(cfg, "optim", None)
    if opt_cfg is None:
        return None

    lr_cfg = getattr(opt_cfg, "lr_schedule", None)
    if lr_cfg is None:
        return None

    lr_type = getattr(lr_cfg, "type", None)
    if lr_type is None:
        return None

    if lr_type == ComposedLRSchedule.name:
        params = getattr(lr_cfg, "params", None)
        scheds_cfg = getattr(params, "schedules", None) if params is not None else None
        if not scheds_cfg:
            raise ValueError("composed_lr_schedule requires params.schedules")

        schedules = []
        for s_cfg in list(scheds_cfg):
            s_type = getattr(s_cfg, "type", None)
            if s_type is None:
                raise ValueError("Each schedule needs a type")
            cls = _get(LR_SCHEDULE_REGISTRY, s_type, "lr schedule")
            schedules.append(cls(s_cfg))

        return ComposedLRSchedule(schedules=schedules)

    cls = _get(LR_SCHEDULE_REGISTRY, lr_type, "lr schedule")
    return cls(lr_cfg)


def build_optimizer(cfg, model: BaseDiffusionModel) -> torch.optim.Optimizer:
    opt_cfg = getattr(cfg, "optim", None)
    if opt_cfg is None:
        raise ValueError("Missing cfg.optim for optimizer build")

    opt_type = getattr(opt_cfg, "type", None)
    if opt_type is None:
        raise ValueError("cfg.optim.type is required")

    cls = _get(OPTIM_REGISTRY, str(opt_type).lower(), "optimizer")
    builder = cls()
    return builder.build(model, cfg)


def build_denoise(
    *,
    model: BaseDiffusionModel,
    conditioning: BaseConditioningPolicy,
    observed_mask,
    observed,
    guidance: GuidanceConfig,
    descriptor=None,
):
    return build_denoise_fn(
        model=model,
        conditioning=conditioning,
        observed_mask=observed_mask,
        observed=observed,
        guidance=guidance,
        descriptor=descriptor,
    )




# -----------------------------------------------------------------------------
# Mode-specific assembly
# -----------------------------------------------------------------------------

def build_training(cfg) -> TrainingArtifacts:
    device = build_device(cfg)
    _apply_lr_scaling(cfg)

    steps = [
        ("forward process", lambda: build_forward(cfg)),
        ("conditioning", lambda: build_conditioning(cfg)),
        ("parametrization", None),  # depends on forward
        ("model", None),            # depends on parametrization + conditioning
        ("schedule", None),
        ("loss", None),
        ("optimizer", None),
        ("data", None),
    ]

    with tqdm(total=len(steps), desc="Building", unit="component", ncols=100) as pbar:
        forward = build_forward(cfg)
        pbar.set_postfix_str("forward")
        pbar.update(1)

        conditioning = build_conditioning(cfg)
        pbar.set_postfix_str("conditioning")
        pbar.update(1)

        parametrization = build_parametrization(cfg, forward=forward)
        pbar.set_postfix_str("parametrization")
        pbar.update(1)

        model = build_model(cfg, device=device,
                            parametrization=parametrization,
                            conditioning=conditioning)
        pbar.set_postfix_str("model")
        pbar.update(1)

        schedule = build_schedule(cfg, modalities=model.modalities, device=device)
        pbar.set_postfix_str("schedule")
        pbar.update(1)

        loss = build_loss(cfg)
        pbar.set_postfix_str("loss")
        pbar.update(1)

        optimizer = build_optimizer(cfg, model)
        pbar.set_postfix_str("optimizer")
        pbar.update(1)

        from cogedi.data.base import build_data_source
        data_source = build_data_source(cfg, device=device)
        pbar.set_postfix_str("data")
        pbar.update(1)

        lr_schedule = build_lr_schedule(cfg)

    return TrainingArtifacts(
        model=model,
        parametrization=parametrization,
        forward=forward,
        schedule=schedule,
        loss=loss,
        conditioning=conditioning,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        data_source=data_source,
        device=device,
    )


def build_sampling(cfg) -> SamplingArtifacts:
    device = build_device(cfg)

    with tqdm(total=6, desc="Building (sample)", unit="component", ncols=100) as pbar:
        forward = build_forward(cfg)
        pbar.set_postfix_str("forward")
        pbar.update(1)

        conditioning = build_conditioning(cfg)
        pbar.set_postfix_str("conditioning")
        pbar.update(1)

        parametrization = build_parametrization(cfg, forward=forward)
        pbar.set_postfix_str("parametrization")
        pbar.update(1)

        model = build_model(cfg, device=device, parametrization=parametrization, conditioning=conditioning)
        pbar.set_postfix_str("model")
        pbar.update(1)

        schedule = build_schedule(cfg, modalities=model.modalities, device=device)
        pbar.set_postfix_str("schedule")
        pbar.update(1)

        solver = build_solver(cfg)
        pbar.set_postfix_str("solver")
        pbar.update(1)

    return SamplingArtifacts(
        model=model,
        parametrization=parametrization,
        forward=forward,
        schedule=schedule,
        solver=solver,
        conditioning=conditioning,
        device=device,
    )

def build_eval(cfg) -> EvalArtifacts:
    device = build_device(cfg)

    with tqdm(total=7, desc="Building (eval)", unit="component", ncols=100) as pbar:
        forward = build_forward(cfg)
        pbar.set_postfix_str("forward")
        pbar.update(1)

        conditioning = build_conditioning(cfg)
        pbar.set_postfix_str("conditioning")
        pbar.update(1)

        parametrization = build_parametrization(cfg, forward=forward)
        pbar.set_postfix_str("parametrization")
        pbar.update(1)

        model = build_model(cfg, device=device, parametrization=parametrization, conditioning=conditioning)
        pbar.set_postfix_str("model")
        pbar.update(1)

        schedule = build_schedule(cfg, modalities=model.modalities, device=device)
        pbar.set_postfix_str("schedule")
        pbar.update(1)

        loss = build_loss(cfg)
        pbar.set_postfix_str("loss")
        pbar.update(1)

        solver = build_solver(cfg)
        pbar.set_postfix_str("solver")
        pbar.update(1)

    return EvalArtifacts(
        model=model,
        parametrization=parametrization,
        forward=forward,
        schedule=schedule,
        solver=solver,
        loss=loss,
        conditioning=conditioning,
        device=device,
    )
