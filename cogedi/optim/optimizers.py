from __future__ import annotations

import abc
from typing import Callable, Dict, Iterable, List, Tuple

import torch


class BaseOptimizerBuilder(abc.ABC):
    """Abstract optimizer builder interface."""

    name: str

    @abc.abstractmethod
    def build(self, model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
        """Instantiate an optimizer for the given model and cfg."""


class AdamWOptimizer(BaseOptimizerBuilder):
    """AdamW optimizer builder."""

    name = "adamw"

    def build(self, model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
        opt_cfg = getattr(cfg, "optim", cfg)
        lr = float(getattr(opt_cfg, "lr", 1e-4))
        wd = float(getattr(opt_cfg, "weight_decay", 0.0))

        if bool(getattr(opt_cfg, "layerwise_decay", False)):
            params = _layerwise_param_groups(model, opt_cfg, base_lr=lr, weight_decay=wd)
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


class AdamOptimizer(BaseOptimizerBuilder):
    """Adam optimizer builder."""

    name = "adam"

    def build(self, model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
        opt_cfg = getattr(cfg, "optim", cfg)
        lr = float(getattr(opt_cfg, "lr", 1e-4))
        wd = float(getattr(opt_cfg, "weight_decay", 0.0))

        if bool(getattr(opt_cfg, "layerwise_decay", False)):
            params = _layerwise_param_groups(model, opt_cfg, base_lr=lr, weight_decay=wd)
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)

        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _layerwise_param_groups(
    model: torch.nn.Module,
    opt_cfg,
    *,
    base_lr: float,
    weight_decay: float,
) -> List[Dict]:
    """
    Build param groups with lr_scale for layer-wise LR decay (LLRD).

    Policy is controlled by optim.layer_id_policy:
      - "vit_like" (GeomDist): uses blocks/patch_embed naming
      - "name_depth": uses name segment depth
      - "flat": no layer decay (all layer_id = max)
      - callable: model.get_layer_id(param_name) if present
    """
    decay = float(getattr(opt_cfg, "layer_decay", 0.75))
    policy = str(getattr(opt_cfg, "layer_id_policy", "vit_like")).lower()
    no_wd = set(getattr(opt_cfg, "no_weight_decay", []) or [])

    named_params = list(model.named_parameters())
    if not named_params:
        return [{"params": [], "lr": base_lr, "weight_decay": weight_decay}]

    layer_id_fn = _make_layer_id_fn(model, opt_cfg, policy, named_params)

    max_layer = max(layer_id_fn(n) for n, _ in named_params)
    layer_scales = [decay ** (max_layer - i) for i in range(max_layer + 1)]

    groups: Dict[Tuple[int, float], Dict] = {}
    for name, param in named_params:
        if not param.requires_grad:
            continue

        layer_id = layer_id_fn(name)
        lr_scale = layer_scales[layer_id]

        if param.ndim == 1 or name in no_wd:
            this_wd = 0.0
        else:
            this_wd = weight_decay

        key = (layer_id, this_wd)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr_scale": lr_scale,
                "weight_decay": this_wd,
            }
        groups[key]["params"].append(param)

    return list(groups.values())


def _make_layer_id_fn(
    model: torch.nn.Module,
    opt_cfg,
    policy: str,
    named_params: List[Tuple[str, torch.nn.Parameter]],
) -> Callable[[str], int]:
    if hasattr(model, "get_layer_id") and callable(getattr(model, "get_layer_id")):
        return lambda n: int(model.get_layer_id(n))

    if policy == "flat":
        return lambda n: 0

    if policy == "name_depth":
        depths = {n: n.count(".") for n, _ in named_params}
        max_depth = max(depths.values()) if depths else 0
        return lambda n: int(depths.get(n, max_depth))

    # default: vit-like mapping (GeomDist behavior)
    num_layers = int(getattr(opt_cfg, "num_layers", 0))
    if num_layers <= 0 and hasattr(model, "blocks"):
        num_layers = len(getattr(model, "blocks")) + 1

    if num_layers <= 0:
        # fallback to name depth if model doesn't expose blocks
        depths = {n: n.count(".") for n, _ in named_params}
        max_depth = max(depths.values()) if depths else 0
        return lambda n: int(depths.get(n, max_depth))

    def vit_like(name: str) -> int:
        if name in {"cls_token", "pos_embed"}:
            return 0
        if name.startswith("patch_embed"):
            return 0
        if name.startswith("blocks"):
            parts = name.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1]) + 1
        return num_layers

    return vit_like
