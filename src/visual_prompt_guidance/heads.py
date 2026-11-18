"""Pure head weighting. Does not know loss names or block naming conventions."""
from __future__ import annotations
import math
from typing import Mapping
import torch


def aggregate(maps: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted [B,H,h,w] -> [B,1,h,w], preserving gradients to maps."""
    if maps.ndim != 4 or weights.shape != (maps.shape[1],):
        raise ValueError("Mismatched head map and weight shapes")
    return (maps * weights.to(maps.device, maps.dtype)[None, :, None, None]).sum(1, keepdim=True)


class Uniform:
    needs_feedback = False

    def reset(self) -> None:
        pass

    def weights(self, labels: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        return torch.full((len(labels),), 1.0 / len(labels), device=device)

    def update(self, labels, before, after) -> None:
        return None


class Manual:
    needs_feedback = False

    def reset(self) -> None:
        pass

    def __init__(self, selected: Mapping[str, float]):
        if not selected or any((not math.isfinite(v) or v < 0) for v in selected.values()) or sum(selected.values()) <= 0:
            raise ValueError("Manual selection requires finite, nonnegative weights with nonzero sum")
        self.selected = dict(selected)

    def weights(self, labels: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        missing = self.selected.keys() - set(labels)
        if missing:
            raise ValueError(f"Unknown head IDs: {sorted(missing)}")
        w = torch.tensor([self.selected.get(k, 0.) for k in labels], device=device)
        return w / w.sum()

    def update(self, labels, before, after) -> None:
        return None


class Adaptive:
    """Uses decrease in any supplied scoring objective after a latent update."""
    needs_feedback = True

    def __init__(self, temperature: float = 0.01, ema: float = 0.5, mix: float = 0.1):
        if temperature <= 0 or not (0 < ema <= 1) or not (0 <= mix <= 1):
            raise ValueError("Invalid temperature, ema or mix")
        self.temperature, self.ema, self.mix = temperature, ema, mix
        self._labels: tuple[str, ...] | None = None
        self._weights: torch.Tensor | None = None

    def reset(self) -> None:
        self._labels = None
        self._weights = None

    def weights(self, labels: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        if self._weights is None:
            self._labels = labels
            self._weights = torch.full((len(labels),), 1 / len(labels), device=device)
        if self._labels != labels:
            raise ValueError("Head identities changed during adaptive weighting")
        return self._weights.to(device)

    @torch.no_grad()
    def update(self, labels: tuple[str, ...], before: torch.Tensor, after: torch.Tensor) -> None:
        self.weights(labels, device=before.device)
        if before.shape != after.shape or before.ndim != 2 or before.shape[1] != len(labels):
            raise ValueError("Expected matching per-head scores [B,heads]")
        improvements = (before.float() - after.float()).mean(dim=0)
        if not torch.isfinite(improvements).all():
            raise ValueError("Head scores contain non-finite values")
        soft = torch.softmax(improvements / self.temperature, dim=0)
        uniform = torch.full_like(soft, 1.0 / len(labels))
        candidate = (1 - self.mix) * soft + self.mix * uniform
        self._weights = ((1 - self.ema) * self._weights.to(soft.device) + self.ema * candidate).detach()
        self._weights = self._weights / self._weights.sum()
