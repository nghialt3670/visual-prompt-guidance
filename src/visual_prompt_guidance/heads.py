"""Pure head weighting. Does not know loss names or block naming conventions."""
from __future__ import annotations
import json
import math
from pathlib import Path
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

class Learned(torch.nn.Module):
    """Trainable, global per-head logits with a stable identity set.

    During training, initialize lazily on the first guided forward. The Trainer
    creates its optimizer *after* this initialization. At inference time load
    trained weights with Learned.load(path). Unselected heads receive exactly 0.
    """
    needs_feedback = False

    def __init__(self, *, selected: tuple[str, ...] | None = None, temperature: float = 1.0):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Learned temperature must be positive and finite")
        if selected is not None and (not selected or len(set(selected)) != len(selected)):
            raise ValueError("Selected heads must be nonempty and unique")
        self.selected = tuple(selected) if selected is not None else None
        self.temperature = float(temperature)
        self.labels: tuple[str, ...] | None = None
        self.logits: torch.nn.Parameter | None = None
        self._inference_checkpoint = False

    def reset(self) -> None:
        # Do not reset learned parameters between images; their meaning is global.
        pass

    def weights(self, labels: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Expected nonempty unique head labels")
        if self.labels is None:
            chosen = self.selected if self.selected is not None else labels
            absent = set(chosen) - set(labels)
            if absent:
                raise ValueError(f"Selected heads missing from backend: {sorted(absent)}")
            self.labels = labels
            self.logits = torch.nn.Parameter(torch.zeros(len(chosen), device=device, dtype=torch.float32))
        elif labels != self.labels:
            raise ValueError("Learned head identities/order changed; keep layers identical across the dataset")
        if self.logits.device != device:
            if self._inference_checkpoint:
                # Checkpoint inference can migrate to the backend's actual
                # device. Training must never replace a registered optimizer param.
                self.logits = torch.nn.Parameter(self.logits.detach().to(device))
            else:
                raise ValueError("Learned policy is on the wrong device; call policy.to(device) BEFORE optimizer creation")
        chosen = self.selected if self.selected is not None else labels
        indices = torch.tensor([labels.index(h) for h in chosen], device=device, dtype=torch.long)
        probs = torch.softmax(self.logits / self.temperature, dim=0)
        return torch.zeros(len(labels), dtype=probs.dtype, device=device).scatter(0, indices, probs)

    def update(self, labels, before, after) -> None:
        # Offline image-level training updates logits using torch.optim.
        pass

    def save(self, path: str | Path) -> None:
        if self.labels is None or self.logits is None:
            raise ValueError("Learned policy has not been initialized")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "visual-prompt-guidance/learned-heads-v1",
            "labels": list(self.labels),
            "selected": list(self.selected) if self.selected is not None else None,
            "temperature": self.temperature,
            "logits": [float(x) for x in self.logits.detach().cpu().tolist()],
        }
        dest.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> 'Learned':
        payload = json.loads(Path(path).read_text())
        if payload.get("format") != "visual-prompt-guidance/learned-heads-v1":
            raise ValueError("Unrecognized learned-head checkpoint format")
        labels = tuple(payload["labels"])
        selected = tuple(payload["selected"]) if payload["selected"] is not None else None
        obj = cls(selected=selected, temperature=float(payload["temperature"]))
        chosen = selected if selected is not None else labels
        if not labels or len(set(labels)) != len(labels) or len(payload["logits"]) != len(chosen):
            raise ValueError("Invalid learned-head checkpoint")
        if set(chosen) - set(labels):
            raise ValueError("Selected checkpoint heads absent from labels")
        obj.labels = labels
        obj._inference_checkpoint = True
        obj.logits = torch.nn.Parameter(torch.tensor(payload["logits"], device=device, dtype=torch.float32))
        if not torch.isfinite(obj.logits).all():
            raise ValueError("Checkpoint contains non-finite logits")
        return obj


class HeadSubset:
    """Optional, identity-based head filter for any existing policy.

    Unselected heads always have zero weight, including in Adaptive updates.
    Unlike filtering the attention tensor, this preserves native head labels
    and lets backends capture the same layer set.
    """
    def __init__(self, policy, selected: tuple[str, ...]):
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("HeadSubset requires unique, nonempty head IDs")
        self.policy = policy
        self.selected = tuple(selected)
        self.needs_feedback = policy.needs_feedback

    def reset(self):
        self.policy.reset()

    def _indices(self, labels):
        unavailable = set(self.selected) - set(labels)
        if unavailable:
            raise ValueError(f"Unknown head IDs: {sorted(unavailable)}")
        selected = tuple(h for h in labels if h in self.selected)
        indices = [labels.index(h) for h in selected]
        return selected, indices

    def weights(self, labels, *, device):
        selected, indices = self._indices(labels)
        active = self.policy.weights(selected, device=device)
        if active.shape != (len(indices),):
            raise ValueError("Child policy returned invalid subset weights")
        return torch.zeros(len(labels), device=device, dtype=active.dtype).scatter(
            0, torch.tensor(indices, device=device), active
        )

    def update(self, labels, before, after):
        selected, indices = self._indices(labels)
        self.policy.update(selected, before[:, indices], after[:, indices])
