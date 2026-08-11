from __future__ import annotations

from tqdm import tqdm

from cogedi.build import (
    MODEL_REGISTRY,
    FORWARD_REGISTRY,
    PARAM_REGISTRY,
    COND_REGISTRY,
    SCHEDULE_REGISTRY,
    SOLVER_REGISTRY,
    LOSS_REGISTRY,
    BACKBONE_REGISTRY,
    POINT_EMBED_REGISTRY,
    SIGMA_EMBED_REGISTRY,
    DESCRIPTOR_EMBED_REGISTRY,
    PRECOND_REGISTRY,
    LR_SCHEDULE_REGISTRY,
    OPTIM_REGISTRY,
    DATA_REGISTRY,
)

# general
from cogedi.forward.ve import VEForwardProcess
from cogedi.models.diffusion_model import DiffusionModel
from cogedi.models.parameterizations.eps import EpsParametrization
from cogedi.models.parameterizations.x0 import X0Parametrization
from cogedi.models.parameterizations.v import VParametrization
from cogedi.models.parameterizations.score import ScoreParametrization
from cogedi.models.precond.edm import EDMPreconditioning
from cogedi.conditioning.unidiff_hard import UniDiffHardConditioning
from cogedi.losses.mse import MSELoss
from cogedi.schedules.edm import EDMSigmaSchedule
from cogedi.solvers.heun import HeunSolver
from cogedi.solvers.multistep import DPM2StyleSolver

# dummmy
from cogedi.schedules.dummy import ComposedDummySigmaSchedule
from cogedi.solvers.dummy import DummySolver
from cogedi.models.backbones.dummy import DummyBackbone
from cogedi.models.embedders.dummy_point import DummyPointEmbedder
from cogedi.models.embedders.dummy_sigma import DummySigmaEmbedder
from cogedi.models.precond.identity import IdentityPreconditioning

# geomdist
from cogedi.models.backbones.geomdist import GeomDistBackbone
from cogedi.models.embedders.geomdist_point import GeomDistPointEmbedder
from cogedi.models.embedders.geomdist_timestep import (
    GeomDistFourierEmbedder,
    GeomDistTimestepEmbedder,
)
from cogedi.losses.geomdist import GeomDistEDMLoss
from cogedi.losses.geomdist_unsupervised import GeomDistUnsupervisedLoss
from cogedi.losses.geomdist_aux import GeomDistEDMWithAuxLoss
from cogedi.losses.geomdist_contrastive import GeomDistEDMWithContrastiveLoss
from cogedi.losses.geomdist_aux_contrastive import GeomDistEDMWithAuxContrastiveLoss

# Backbones
from cogedi.models.backbones.DiT import DiTBackbone
from cogedi.models.backbones.DiT_geomdist import DiTGeomDistBackbone
from cogedi.models.backbones.geofusion import GeoFusionBackbone
from cogedi.models.backbones.geofusion_attention import GeoFusionAttentionBackbone

# data sources
from cogedi.data.synthetic import SyntheticDataSource
from cogedi.data.mesh import GenericMeshDataSource
from cogedi.data.faust import FAUSTDataSource
from cogedi.data.dfaust import DFAUSTDataSource
from cogedi.data.smal import SMALDataSource
from cogedi.data.smalr import SMALRDataSource

# lr schedules
from cogedi.optim.schedules import (
    ComposedLRSchedule,
    ConstantLRSchedule,
    CosineWarmupSchedule,
    MultiplierLRSchedule,
)
from cogedi.optim.optimizers import AdamOptimizer, AdamWOptimizer

# unsupervised
from cogedi.models.backbones.geomdist_descriptor import GeomDistDescriptorBackbone
from cogedi.models.embedders.descriptor_MLP import DescriptorMLPEmbedder
from cogedi.losses.geomdist_unsupervised import GeomDistUnsupervisedLoss



def register_all(show_progress: bool = True) -> None:
    """
    Populate all registries.

    This should be called exactly once at program startup.
    """
    registrations = [
        # --- model ---
        ("model", MODEL_REGISTRY, DiffusionModel),

        # --- forward processes ---
        ("forward", FORWARD_REGISTRY, VEForwardProcess),

        # --- parametrizations ---
        ("parametrization", PARAM_REGISTRY, EpsParametrization),
        ("parametrization", PARAM_REGISTRY, X0Parametrization),
        ("parametrization", PARAM_REGISTRY, VParametrization),
        ("parametrization", PARAM_REGISTRY, ScoreParametrization),

        # --- conditioning ---
        ("conditioning", COND_REGISTRY, UniDiffHardConditioning),

        # --- schedules ---
        ("schedule", SCHEDULE_REGISTRY, ComposedDummySigmaSchedule),
        ("schedule", SCHEDULE_REGISTRY, EDMSigmaSchedule),

        # --- solvers ---
        ("solver", SOLVER_REGISTRY, DummySolver),
        ("solver", SOLVER_REGISTRY, HeunSolver),
        ("solver", SOLVER_REGISTRY, DPM2StyleSolver),

        # --- losses ---
        ("loss", LOSS_REGISTRY, MSELoss),
        ("loss", LOSS_REGISTRY, GeomDistEDMLoss),
        ("loss", LOSS_REGISTRY, GeomDistUnsupervisedLoss),
        ("loss", LOSS_REGISTRY, GeomDistEDMWithAuxLoss),
        ("loss", LOSS_REGISTRY, GeomDistEDMWithContrastiveLoss),
        ("loss", LOSS_REGISTRY, GeomDistEDMWithAuxContrastiveLoss),

        # --- lr schedules ---
        ("lr_schedule", LR_SCHEDULE_REGISTRY, CosineWarmupSchedule),
        ("lr_schedule", LR_SCHEDULE_REGISTRY, ConstantLRSchedule),
        ("lr_schedule", LR_SCHEDULE_REGISTRY, MultiplierLRSchedule),
        ("lr_schedule", LR_SCHEDULE_REGISTRY, ComposedLRSchedule),

        # --- optimizers ---
        ("optimizer", OPTIM_REGISTRY, AdamWOptimizer),
        ("optimizer", OPTIM_REGISTRY, AdamOptimizer),

        # --- model internals ---
        ("backbone", BACKBONE_REGISTRY, DummyBackbone),
        ("backbone", BACKBONE_REGISTRY, GeomDistBackbone),
        ("backbone", BACKBONE_REGISTRY, DiTBackbone),
        ("backbone", BACKBONE_REGISTRY, DiTGeomDistBackbone),
        ("backbone", BACKBONE_REGISTRY, GeoFusionBackbone),
        ("backbone", BACKBONE_REGISTRY, GeoFusionAttentionBackbone),
        ("backbone", BACKBONE_REGISTRY, GeomDistDescriptorBackbone),
        ("point_embedder", POINT_EMBED_REGISTRY, DummyPointEmbedder),
        ("point_embedder", POINT_EMBED_REGISTRY, GeomDistPointEmbedder),
        ("sigma_embedder", SIGMA_EMBED_REGISTRY, DummySigmaEmbedder),
        ("sigma_embedder", SIGMA_EMBED_REGISTRY, GeomDistTimestepEmbedder),
        ("sigma_embedder", SIGMA_EMBED_REGISTRY, GeomDistFourierEmbedder),
        ("descriptor_embedder", DESCRIPTOR_EMBED_REGISTRY, DescriptorMLPEmbedder),
        ("preconditioning", PRECOND_REGISTRY, IdentityPreconditioning),
        ("preconditioning", PRECOND_REGISTRY, EDMPreconditioning),

        # --- data sources ---
        ("data", DATA_REGISTRY, SyntheticDataSource),
        ("data", DATA_REGISTRY, GenericMeshDataSource),
        ("data", DATA_REGISTRY, FAUSTDataSource),
        ("data", DATA_REGISTRY, DFAUSTDataSource),
        ("data", DATA_REGISTRY, SMALDataSource),
        ("data", DATA_REGISTRY, SMALRDataSource),
    ]

    iterator = registrations
    if show_progress:
        iterator = tqdm(registrations, desc="Registering components", unit="item", ncols=100)

    for kind, registry, cls in iterator:
        registry[cls.name] = cls