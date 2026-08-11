from __future__ import annotations

import abc
import math
from typing import Any, Mapping, Sequence


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    d = getattr(obj, "__dict__", None)
    return {} if d is None else dict(d)


def _set_lr(optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * scale


def _cosine_warmup_lr(epoch: float, cfg, params: Mapping[str, Any]) -> float:
    train_cfg = getattr(cfg, "train", cfg)

    base_lr = float(params.get("lr", getattr(train_cfg, "lr", 1e-4)))
    min_lr = float(params.get("min_lr", getattr(train_cfg, "min_lr", 0.0)))
    warmup_epochs = float(params.get("warmup_epochs", getattr(train_cfg, "warmup_epochs", 0.0)))
    total_epochs = float(params.get("max_epochs", getattr(train_cfg, "max_epochs", getattr(train_cfg, "epochs", 1.0))))

    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * epoch / warmup_epochs

    denom = max(total_epochs - warmup_epochs, 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / denom))


class BaseLRSchedule(abc.ABC):
    """Abstract LR schedule interface."""

    name: str

    @abc.abstractmethod
    def step(self, optimizer, epoch: float, cfg) -> float:
        """Update optimizer LR and return the new LR."""


class CosineWarmupSchedule(BaseLRSchedule):
    """GeomDist-style cosine schedule with warmup."""

    name = "cosine_warmup"

    def __init__(self, cfg=None):
        self.params = _as_dict(getattr(cfg, "params", cfg))

    def step(self, optimizer, epoch: float, cfg) -> float:
        lr = _cosine_warmup_lr(epoch, cfg, self.params)
        _set_lr(optimizer, lr)
        return float(lr)


class ConstantLRSchedule(BaseLRSchedule):
    """Constant learning-rate schedule."""

    name = "constant"

    def __init__(self, cfg=None):
        self.params = _as_dict(getattr(cfg, "params", cfg))

    def step(self, optimizer, epoch: float, cfg) -> float:
        lr = float(self.params.get("lr", _cosine_warmup_lr(epoch, cfg, {})))
        _set_lr(optimizer, lr)
        return float(lr)


class MultiplierLRSchedule(BaseLRSchedule):
    """Multiply current LR by a constant factor."""

    name = "multiplier"

    def __init__(self, cfg=None):
        self.params = _as_dict(getattr(cfg, "params", cfg))

    def step(self, optimizer, epoch: float, cfg) -> float:
        factor = float(self.params.get("factor", 1.0))
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * factor
        return float(optimizer.param_groups[0]["lr"])


class ComposedLRSchedule(BaseLRSchedule):
    """Apply multiple LR schedules in sequence."""

    name = "composed_lr_schedule"

    def __init__(self, *, schedules: Sequence[BaseLRSchedule]):
        self.schedules = list(schedules)

    def step(self, optimizer, epoch: float, cfg) -> float:
        lr = 0.0
        for sched in self.schedules:
            lr = sched.step(optimizer, epoch, cfg)
        return float(lr)
