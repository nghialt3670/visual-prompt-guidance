"""Composable loss evaluation shared by guidance AND adaptive head scoring."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Mapping
import torch
from .contracts import Target

# A loss returns per-head, per-example scores [B,H], never reduces heads itself.
LossFunction = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Loss:
    name: str
    function: LossFunction
    weight: float = 1.0
    params: Mapping[str, Any] = field(default_factory=dict)


class Objective:
    def __init__(self, *terms: Loss):
        if not terms:
            raise ValueError("An objective must contain at least one loss")
        if len(set(term.name for term in terms)) != len(terms):
            raise ValueError("Loss labels must be unique")
        self.terms = terms

    def evaluate(self, maps: torch.Tensor, target: Target) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """`maps`: [B,heads,H,W]. Total and every component: [B,heads]."""
        if maps.ndim != 4 or maps.shape[0] != target.density.shape[0] or maps.shape[2:] != target.density.shape[2:]:
            raise ValueError("Spatial map and target dimensions differ")
        parts: dict[str, torch.Tensor] = {}
        total = torch.zeros(maps.shape[:2], device=maps.device, dtype=torch.float32)
        for term in self.terms:
            value = term.function(maps.float(), target, **term.params)
            if value.shape != total.shape:
                raise ValueError(f"Loss {term.name!r} returned {tuple(value.shape)}; expected {tuple(total.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Loss {term.name!r} produced non-finite values")
            parts[term.name] = value
            total = total + term.weight * value
        return total, parts

    def __call__(self, maps: torch.Tensor, target: Target) -> torch.Tensor:
        return self.evaluate(maps, target)[0]
