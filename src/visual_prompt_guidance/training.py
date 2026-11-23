"""Optional second-order image-supervised head-weight training.

The diffusion/flow backbone stays frozen. Only Learned.logits are optimized.
This module contains no backend- or Gradio-specific logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import shutil
import stat
import tempfile
from typing import Any, Mapping, Callable
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .contracts import Request, VisualPrompt
from .engine import Engine
from .heads import Learned
from .prompts import Point, Box, Scribble


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    optimizer: str = "adamw"  # adam, adamw, sgd
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    detect_anomaly: bool = False  # Diagnostic only: record the first NaN-producing autograd op.
    shuffle: bool = True
    shuffle_seed: int = 42
    checkpoint: str = "outputs/learned_heads.json"
    history: str = "outputs/head_learning_history.json"

    def __post_init__(self):
        if self.epochs < 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.optimizer.lower() not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer must be adam, adamw or sgd")
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError("weight_decay must be nonnegative and finite")
        if not (math.isfinite(self.grad_clip) and self.grad_clip >= 0):
            raise ValueError("grad_clip must be nonnegative and finite")


class RegionLoss:
    """Plugin interface for a region-aware training loss."""
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PixelRegionLoss(RegionLoss):
    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, predicted: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        pred = predicted.float()
        tgt = target.float()
        if self.mode == "l1":
            values = (pred - tgt).abs()
        elif self.mode == "mse":
            values = (pred - tgt).square()
        elif self.mode == "charbonnier":
            values = ((pred - tgt).square() + 1e-6).sqrt()
        elif self.mode == "smooth_l1":
            values = F.smooth_l1_loss(pred, tgt, reduction="none")
        else:
            raise ValueError(f"Unknown pixel region loss {self.mode!r}")
        values = values.mean(1, keepdim=True)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)


class MaskedLpipsRegionLoss(RegionLoss):
    """Spatial LPIPS, averaged only over the requested region.

    LPIPS sees the *original full images*, not blacked-out backgrounds. The
    spatial output is masked *after* feature extraction, which avoids diluting
    edit loss by the fraction of the canvas that was changed. Perceptual
    receptive fields can still sample context outside the region (intentional).
    """

    def __init__(self):
        self._lpips = None

    def __call__(self, predicted: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        if self._lpips is None:
            try:
                import lpips
            except ImportError as error:
                raise ImportError("Install lpips (uv sync --extra perceptual) for LPIPS image loss") from error
            self._lpips = lpips.LPIPS(net="alex", spatial=True).eval().requires_grad_(False)
        network = self._lpips.to(device=predicted.device)
        # LPIPS expects images in [-1,1]. Do NOT multiply by mask first:
        # doing so introduces a hard artificial edge and spatial dilution.
        difference = network(predicted.float() * 2 - 1, target.float() * 2 - 1)
        if difference.ndim != 4 or difference.shape[:2] != (predicted.shape[0], 1):
            raise ValueError("Spatial LPIPS must produce a [B,1,H,W] loss map")
        if difference.shape[-2:] != mask.shape[-2:]:
            difference = F.interpolate(difference, size=mask.shape[-2:],
                                       mode="bilinear", align_corners=False)
        weights = mask.to(device=difference.device, dtype=difference.dtype)
        weights = weights.expand_as(difference)
        # When the selected region is empty, contribute zero without NaNs.
        return (difference * weights).sum() / weights.sum().clamp_min(1.0)


IMAGE_LOSS_REGISTRY: dict[str, Callable[[], RegionLoss]] = {
    "l1": lambda: PixelRegionLoss("l1"),
    "mse": lambda: PixelRegionLoss("mse"),
    "charbonnier": lambda: PixelRegionLoss("charbonnier"),
    "smooth_l1": lambda: PixelRegionLoss("smooth_l1"),
    "lpips": MaskedLpipsRegionLoss,
}


def register_image_loss(name: str, factory: Callable[[], RegionLoss]) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Image loss name must be a non-empty string")
    IMAGE_LOSS_REGISTRY[name] = factory


@dataclass(frozen=True)
class ImageLossConfig:
    edit_loss: str = "l1"
    edit_weight: float = 1.0
    background_loss: str = "l1"
    background_weight: float = 1.0
    pixel: str | None = None  # Deprecated alias for both region losses.

    def __post_init__(self):
        if self.pixel is not None:
            object.__setattr__(self, 'edit_loss', self.pixel)
            object.__setattr__(self, 'background_loss', self.pixel)
        for name in (self.edit_loss, self.background_loss):
            if name not in IMAGE_LOSS_REGISTRY:
                raise ValueError(f"Unknown image loss {name!r}")
        if any(not math.isfinite(x) or x < 0 for x in (self.edit_weight, self.background_weight)):
            raise ValueError("Image loss weights must be finite and nonnegative")
        if self.edit_weight + self.background_weight <= 0:
            raise ValueError("Select at least one positive image loss weight")


@dataclass(frozen=True)
class TrainingExample:
    request: Request
    visual: VisualPrompt
    target_image: Image.Image
    edit_mask: Image.Image | None = None


def _load_visual(spec: Mapping[str, Any], root: Path) -> VisualPrompt:
    kind = str(spec.get("type", "")).lower()
    if kind == "point":
        return Point(float(spec["x"]), float(spec["y"]),
                     sigma=float(spec.get("sigma", 12)),
                     edit_radius=float(spec.get("edit_radius", 48)))
    if kind == "box":
        return Box(float(spec["x0"]), float(spec["y0"]),
                   float(spec["x1"]), float(spec["y1"]),
                   margin=float(spec.get("margin", 0)))
    if kind == "scribble":
        if "path" in spec:
            return Scribble(path=[(float(x), float(y)) for x, y in spec["path"]],
                           width=float(spec.get("width", 6)),
                           edit_margin=float(spec.get("edit_margin", 24)))
        if "mask" in spec:
            with Image.open(_safe_file(root, spec["mask"])) as mask:
                return Scribble(mask=mask.convert("L"),
                               edit_margin=float(spec.get("edit_margin", 24)))
        raise ValueError("Scribble must specify either path or mask")
    raise ValueError(f"Unsupported visual prompt type {kind!r}")


def _safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Dataset paths must be nonempty strings")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Missing or outside-dataset file: {relative}")
    return resolved


class EditingDataset:
    """Small-dataset loader for a folder/JSON file or a ZIP with manifest.json.

    The manifest is a JSON list, or {"samples": [...]} and paths are relative
    to the manifest folder. ZIPs are extracted safely into a temporary folder.
    """
    def __init__(self, path: str | Path, *, options: Mapping[str, Any] | None = None):
        self.path = Path(path)
        self.options = dict(options or {})
        self._temporary: Path | None = None
        self._samples: list[Mapping[str, Any]] | None = None
        self.root: Path | None = None

    def __enter__(self):
        if self.path.suffix.lower() == ".zip":
            self._temporary = Path(tempfile.mkdtemp(prefix="vpg-learning-"))
            try:
                with ZipFile(self.path) as archive:
                    members = archive.infolist()
                    if sum(z.file_size for z in members) > 5 * 1024**3:
                        raise ValueError("Dataset ZIP uncompressed size exceeds 5 GB")
                    for info in members:
                        if info.is_dir():
                            continue
                        if stat.S_ISLNK((info.external_attr >> 16) & 0xffff):
                            raise ValueError("Dataset ZIP cannot contain symbolic links")
                        dest = (self._temporary / info.filename).resolve()
                        if not dest.is_relative_to(self._temporary.resolve()):
                            raise ValueError("Unsafe path in dataset ZIP")
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as src, dest.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                manifest = self._temporary / "manifest.json"
                if not manifest.is_file():
                    raise ValueError("Dataset ZIP must contain manifest.json at its root")
            except Exception:
                self.__exit__(None, None, None)
                raise
        elif self.path.is_dir():
            manifest = self.path / "manifest.json"
        else:
            manifest = self.path
        try:
            self.root = manifest.resolve().parent
            items = json.loads(manifest.read_text())
            self._samples = items["samples"] if isinstance(items, dict) else items
            if not isinstance(self._samples, list) or not self._samples:
                raise ValueError("Dataset must contain a nonempty list of samples")
            for i, sample in enumerate(self._samples):
                if not isinstance(sample, dict):
                    raise ValueError(f"Sample {i} must be an object")
                for required in ("input_image", "target_image", "prompt", "object_phrase", "visual"):
                    if required not in sample:
                        raise ValueError(f"Sample {i} missing {required}")
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self._temporary is not None:
            shutil.rmtree(self._temporary, ignore_errors=True)
            self._temporary = None

    def __len__(self):
        if self._samples is None:
            raise RuntimeError("Use EditingDataset as a context manager")
        return len(self._samples)

    def __getitem__(self, index: int) -> TrainingExample:
        if self._samples is None or self.root is None:
            raise RuntimeError("Use EditingDataset as a context manager")
        sample = self._samples[index]
        with Image.open(_safe_file(self.root, sample["input_image"])) as f:
            image = ImageOps.exif_transpose(f).convert("RGB")
        with Image.open(_safe_file(self.root, sample["target_image"])) as f:
            target = ImageOps.exif_transpose(f).convert("RGB")
        if target.size != image.size:
            raise ValueError(f"Sample {index}: target and input must have equal dimensions")
        mask = None
        if sample.get("edit_mask"):
            with Image.open(_safe_file(self.root, sample["edit_mask"])) as f:
                mask = ImageOps.exif_transpose(f).convert("L")
            if mask.size != image.size:
                raise ValueError(f"Sample {index}: edit_mask must match input dimensions")
        visual = _load_visual(sample["visual"], self.root)
        opts = dict(self.options)
        opts.update(sample.get("options") or {})
        request = Request(
            data={"image": image, "prompt": str(sample["prompt"]),
                  "object_phrase": str(sample["object_phrase"])},
            options=opts,
        )
        return TrainingExample(request, visual, target, mask)


def _rgb_tensor(image: Image.Image, *, device: torch.device) -> torch.Tensor:
    values = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device=device)


class ImageObjective:
    """Differentiable image-level supervision, independent of the guidance loss."""
    def __init__(self, config: ImageLossConfig):
        self.config = config
        self.edit_loss = IMAGE_LOSS_REGISTRY[config.edit_loss]()
        self.background_loss = IMAGE_LOSS_REGISTRY[config.background_loss]()

    def __call__(self, predicted: torch.Tensor, example: TrainingExample,
                 projected_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        return self._compute(predicted, example, projected_mask, require_grad=True)

    @torch.no_grad()
    def evaluate(self, predicted: torch.Tensor, example: TrainingExample,
                 projected_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Same image objective for evaluating completed inference outputs."""
        return self._compute(predicted, example, projected_mask, require_grad=False)

    def _compute(self, predicted: torch.Tensor, example: TrainingExample,
                 projected_mask: torch.Tensor, *, require_grad: bool) -> tuple[torch.Tensor, dict[str, float]]:
        if predicted.ndim != 4 or predicted.shape[1] != 3 or predicted.shape[0] != 1:
            raise ValueError("Backend.decode_tensor must return [1,3,H,W] RGB image")
        if require_grad and not predicted.requires_grad:
            raise RuntimeError("Differentiable decode returned image without an autograd graph")
        target = _rgb_tensor(example.target_image, device=predicted.device)
        if target.shape != predicted.shape:
            raise ValueError("Decoded image and reference image dimensions differ")
        if example.edit_mask is not None:
            array = np.asarray(example.edit_mask, dtype=np.float32).copy() / 255
            mask = torch.from_numpy(array)[None, None].to(device=predicted.device)
        else:
            mask = F.interpolate(projected_mask.to(device=predicted.device, dtype=torch.float32),
                                 size=predicted.shape[-2:], mode="nearest")
        mask = mask.clamp(0, 1)
        outside_mask = (1 - mask).clamp(0, 1)
        inside = self.edit_loss(predicted, target, mask)
        outside = self.background_loss(predicted, target, outside_mask)
        total = self.config.edit_weight * inside + self.config.background_weight * outside
        parts = {
            f'edit/{self.config.edit_loss}': float(inside.detach()),
            f'background/{self.config.background_loss}': float(outside.detach()),
        }
        return total, parts


@dataclass(frozen=True)
class TrainResult:
    checkpoint: str
    history: tuple[dict[str, Any], ...]
    head_labels: tuple[str, ...]
    weights: tuple[float, ...]


class HeadTrainer:
    """Optimize only Learned.logits through differentiable guided generation.

    Each example is trained as a separate microbatch to limit peak graph memory.
    `epochs * len(dataset)` full sampling schedules are executed. Large diffusion
    models may still OOM; use a small resolution and few denoising steps first.
    """
    def __init__(self, engine: Engine, *, config: TrainConfig = TrainConfig(),
                 image_loss: ImageLossConfig = ImageLossConfig()):
        if not isinstance(engine.policy, Learned):
            raise TypeError("HeadTrainer requires Engine(policy=Learned(...))")
        self.engine, self.config = engine, config
        self.criterion = ImageObjective(image_loss)

    def fit(self, dataset_path: str | Path, *, options: Mapping[str, Any] | None = None,
            progress: Callable[[int, int, dict[str, Any]], None] | None = None,
            stage: Callable[[str], None] | None = None) -> TrainResult:
        policy: Learned = self.engine.policy
        optimizer = None
        history: list[dict[str, Any]] = []
        config = self.config
        if stage is not None:
            stage("Reading training dataset")
        with EditingDataset(dataset_path, options=options) as dataset:
            total_items = config.epochs * len(dataset)
            if stage is not None:
                stage(f"Dataset ready: {len(dataset)} samples, {config.epochs} epochs, {total_items} updates")
            for epoch in range(config.epochs):
                order = list(range(len(dataset)))
                if config.shuffle:
                    random.Random(config.shuffle_seed + epoch).shuffle(order)
                for sample_index in order:
                    if stage is not None:
                        stage(f"Sample {len(history) + 1}/{total_items} (epoch {epoch + 1}): loading images")
                    example = dataset[sample_index]
                    if stage is not None:
                        stage(f"Sample {len(history) + 1}/{total_items}: preparing model and denoising")
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    # Anomaly mode must cover forward *and* backward to provide
                    # the forward traceback of the first NaN-producing op.
                    # Off by default: autograd anomaly detection is expensive.
                    with torch.autograd.set_detect_anomaly(config.detect_anomaly, check_nan=True):
                        result = self.engine.run_differentiable(example.request, example.visual)
                        if stage is not None:
                            stage(f"Sample {len(history) + 1}/{total_items}: forward complete; computing image loss")
                        if policy.logits is None:
                            raise RuntimeError("No head weights initialized; require at least one guided step")
                        if optimizer is None:
                            name = config.optimizer.lower()
                            optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                                              "sgd": torch.optim.SGD}[name]
                            optimizer = optimizer_type([policy.logits], lr=config.learning_rate,
                                                        weight_decay=config.weight_decay)
                        optimizer.zero_grad(set_to_none=True)
                        image_loss, parts = self.criterion(result.output, example, result.target.edit_mask)
                        if not bool(torch.isfinite(image_loss).item()):
                            raise RuntimeError("Non-finite outer image loss")
                        if stage is not None:
                            stage(f"Sample {len(history) + 1}/{total_items}: backpropagating through denoising")
                        image_loss.backward()
                    if policy.logits.grad is None:
                        raise RuntimeError("Outer image loss did not reach learned head logits")
                    if not bool(torch.isfinite(policy.logits.grad).all().item()):
                        nan_count = int(torch.isnan(policy.logits.grad).sum().item())
                        inf_count = int(torch.isinf(policy.logits.grad).sum().item())
                        raise RuntimeError(
                            f"Non-finite learned-head gradient: {nan_count} NaN and "
                            f"{inf_count} Inf among {policy.logits.grad.numel()} heads; "
                            f"image loss={float(image_loss.detach()):.6g}. "
                            "The optimizer was not stepped. Try fewer denoising/guided "
                            "steps and float32; inspect the inner guidance updater."
                        )
                    gradient_norm = float(policy.logits.grad.norm().detach().item())
                    if config.grad_clip:
                        torch.nn.utils.clip_grad_norm_([policy.logits], config.grad_clip)
                    optimizer.step()
                    if not bool(torch.isfinite(policy.logits).all().item()):
                        raise RuntimeError("Head logits diverged to non-finite values")
                    record = {"epoch": epoch + 1, "sample": sample_index,
                              "loss": float(image_loss.detach()), "grad_norm": gradient_norm,
                              **parts}
                    if policy.logits.is_cuda:
                        gpu = policy.logits.device
                        record['cuda_peak_allocated_gib'] = round(
                            torch.cuda.max_memory_allocated(gpu) / (1024 ** 3), 3)
                        record['cuda_peak_reserved_gib'] = round(
                            torch.cuda.max_memory_reserved(gpu) / (1024 ** 3), 3)
                        if stage is not None:
                            stage(f"CUDA peak (sample {len(history) + 1}/{total_items}): "
                                  f"allocated {record['cuda_peak_allocated_gib']:.2f} GiB, "
                                  f"reserved {record['cuda_peak_reserved_gib']:.2f} GiB")
                    # Record completed-epoch means separately from individual
                    # samples. Comparing successive samples is misleading when
                    # their edit masks and intrinsic difficulty differ.
                    if len(history) % len(dataset) == len(dataset) - 1:
                        previous = history[-(len(dataset)-1):] if len(dataset) > 1 else []
                        current_epoch = [r["loss"] for r in previous]
                        current_epoch.append(record["loss"])
                        record["epoch_mean_loss"] = sum(current_epoch) / len(current_epoch)
                    history.append(record)
                    if progress is not None:
                        progress(len(history), total_items, record)
                    # Do not retain any output tensors/graphs beyond one example.
                    del image_loss, result
        if stage is not None:
            stage("Saving trained head weights and history")
        policy.save(config.checkpoint)
        hist = Path(config.history)
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text(json.dumps({"config": vars(config), "image_loss": vars(self.criterion.config),
                                    "samples": history,
                                    "epoch_means": [
                                        {"epoch": h["epoch"], "loss": h["epoch_mean_loss"]}
                                        for h in history if "epoch_mean_loss" in h
                                    ]}, indent=2))
        weights = policy.weights(policy.labels, device=policy.logits.device).detach().cpu().tolist()
        return TrainResult(config.checkpoint, tuple(history), policy.labels, tuple(weights))
