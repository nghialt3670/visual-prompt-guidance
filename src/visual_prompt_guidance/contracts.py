"""Small, public contracts. The engine knows none of the implementations."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence
import torch


@dataclass(frozen=True)
class Request:
    """Opaque input owned by the backend; no model-specific keys in the engine."""
    data: Any
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Space:
    """Input coordinates (width,height) and attention grid (height,width)."""
    source_size: tuple[int, int]
    grid_size: tuple[int, int]


@dataclass(frozen=True)
class Target:
    """Spatial signal on the attention grid, plus optional task-specific data.

    density/support/edit_mask each have shape [B,1,H,W]. A prompt may choose
    different support and edit masks. Geometry remains opaque to the engine.
    """
    density: torch.Tensor
    support: torch.Tensor
    edit_mask: torch.Tensor
    features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.density.ndim != 4 or self.density.shape[1] != 1:
            raise ValueError("Target density must have shape [B,1,H,W]")
        if self.support.shape != self.density.shape or self.edit_mask.shape != self.density.shape:
            raise ValueError("Target support and edit_mask must match density shape")
        if not torch.isfinite(self.density).all() or (self.density < 0).any() or (self.density.sum((-1,-2)) <= 0).any():
            raise ValueError("Target must have a finite, nonempty, nonnegative density")

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "Target":
        return replace(self, density=self.density.to(device, dtype),
                       support=self.support.to(device, dtype), edit_mask=self.edit_mask.to(device, dtype))


@dataclass(frozen=True)
class AttentionRequest:
    """Opaque layer identifiers belong to the backend; None means backend default."""
    layers: tuple[str, ...] | None = None
    affinity_layers: tuple[str, ...] | None = None
    need_affinity: bool = False


@dataclass(frozen=True)
class AttentionBatch:
    maps: torch.Tensor                # [B,heads,H,W], differentiable when guiding
    labels: tuple[str, ...]           # stable opaque IDs, one per head
    affinity: torch.Tensor | None = None  # [B,H*W,H*W] optional

    def __post_init__(self):
        if self.maps.ndim != 4 or self.maps.shape[1] != len(self.labels) or not self.labels:
            raise ValueError("Head maps must be [B,num_heads,H,W] with matching labels")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Head labels must be unique")
        b, _, h, w = self.maps.shape
        if self.affinity is not None and self.affinity.shape != (b, h*w, h*w):
            raise ValueError("Affinity must be [B,H*W,H*W]")


@dataclass(frozen=True)
class Prediction:
    output: Any                        # backend-owned prediction for scheduler
    attention: AttentionBatch | None = None


@dataclass(frozen=True)
class StepEvent:
    index: int
    timestep: Any
    losses: Mapping[str, float]
    grad_rms: float | None
    labels: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    scores_before: tuple[float, ...] = ()
    scores_after: tuple[float, ...] = ()
    head_maps: torch.Tensor | None = None   # CPU copy, only when requested by observer
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    output: Any
    events: tuple[StepEvent, ...]
    target: Target


class VisualPrompt(Protocol):
    def project(self, space: Space) -> Target: ...


class ModelAdapter(Protocol):
    """Capability: differentiable latent + spatial attention + stepwise sampling.

    Other generator families may implement a DIFFERENT engine/protocol; this
    contract intentionally does not pretend all generators have latent gradients.
    """
    def space(self, request: Request) -> Space: ...
    def prepare(self, request: Request, target: Target) -> Any: ...
    def steps(self, state: Any) -> Sequence[Any]: ...
    def latent(self, state: Any) -> torch.Tensor: ...
    def set_latent(self, state: Any, latent: torch.Tensor) -> None: ...
    def forward(self, state: Any, step: Any, *, attention: AttentionRequest | None) -> Prediction: ...
    def advance(self, state: Any, step: Any, prediction: Prediction) -> None: ...
    def decode(self, state: Any) -> Any: ...


class DifferentiableModelAdapter(ModelAdapter, Protocol):
    """Additional capability required for image-level, higher-order training.

    Decode the current latent without torch.no_grad(), detaching, converting
    to NumPy/PIL, or silently changing to an intermediate low-resolution output.
    The returned tensor must be RGB [B,3,H,W], float in [0,1], at source size.
    """
    def decode_tensor(self, state: Any) -> torch.Tensor: ...


class TargetTransform(Protocol):
    def reset(self, target: Target) -> None: ...
    def requires_affinity(self, step_index: int) -> bool: ...
    def apply(self, target: Target, attention: AttentionBatch, step_index: int) -> Target: ...


class HeadPolicy(Protocol):
    needs_feedback: bool
    def reset(self) -> None: ...
    def weights(self, labels: tuple[str, ...], *, device: torch.device) -> torch.Tensor: ...
    def update(self, labels: tuple[str, ...], before: torch.Tensor, after: torch.Tensor) -> None: ...


class Observer(Protocol):
    wants_maps: bool
    wants_scores: bool
    def step(self, event: StepEvent) -> None: ...
    def end(self, result: Result) -> None: ...
