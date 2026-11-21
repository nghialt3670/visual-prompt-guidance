"""Optional second-order image-supervised head-weight training.

The diffusion/flow backbone stays frozen. Only Learned.logits are optimized.
This module contains no backend- or Gradio-specific logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ImageLossConfig:
    pixel: str = "l1"             # l1, mse, charbonnier, smooth_l1
    inside_weight: float = 1.0    # reconstruction inside the edited region
    outside_weight: float = 1.0   # preserve the unedited region
    lpips_weight: float = 0.0     # optional global learned perceptual loss

    def __post_init__(self):
        if self.pixel not in {"l1", "mse", "charbonnier", "smooth_l1"}:
            raise ValueError("Unknown image pixel loss")
        if any(not math.isfinite(x) or x < 0 for x in
               (self.inside_weight, self.outside_weight, self.lpips_weight)):
            raise ValueError("Image loss weights must be finite and nonnegative")
        if self.inside_weight + self.outside_weight + self.lpips_weight <= 0:
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
        self._lpips = None

    def __call__(self, predicted: torch.Tensor, example: TrainingExample,
                 projected_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if predicted.ndim != 4 or predicted.shape[1] != 3 or predicted.shape[0] != 1:
            raise ValueError("Backend.decode_tensor must return [1,3,H,W] RGB image")
        if not predicted.requires_grad:
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
        if self.config.pixel == "l1":
            pixel_loss = (predicted.float() - target).abs()
        elif self.config.pixel == "mse":
            pixel_loss = (predicted.float() - target).square()
        elif self.config.pixel == "charbonnier":
            pixel_loss = ((predicted.float() - target).square() + 1e-6).sqrt()
        else:
            pixel_loss = F.smooth_l1_loss(predicted.float(), target, reduction="none")
        pixel_loss = pixel_loss.mean(1, keepdim=True)
        inside = (pixel_loss * mask).sum() / mask.sum().clamp_min(1.0)
        outside_mask = 1 - mask
        outside = (pixel_loss * outside_mask).sum() / outside_mask.sum().clamp_min(1.0)
        total = self.config.inside_weight * inside + self.config.outside_weight * outside
        parts = {"inside": float(inside.detach()), "outside": float(outside.detach())}
        if self.config.lpips_weight > 0:
            if self._lpips is None:
                try:
                    import lpips
                except ImportError as error:
                    raise ImportError("pip install lpips to use lpips_weight > 0") from error
                self._lpips = lpips.LPIPS(net="alex").eval().requires_grad_(False)
            network = self._lpips.to(device=predicted.device)
            perceptual = network(predicted.float() * 2 - 1, target * 2 - 1).mean()
            total = total + self.config.lpips_weight * perceptual
            parts["lpips_global"] = float(perceptual.detach())
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
            progress: Callable[[int, int, dict[str, Any]], None] | None = None) -> TrainResult:
        policy: Learned = self.engine.policy
        optimizer = None
        history: list[dict[str, Any]] = []
        config = self.config
        with EditingDataset(dataset_path, options=options) as dataset:
            total_items = config.epochs * len(dataset)
            for epoch in range(config.epochs):
                order = list(range(len(dataset)))
                if config.shuffle:
                    random.Random(config.shuffle_seed + epoch).shuffle(order)
                for sample_index in order:
                    example = dataset[sample_index]
                    result = self.engine.run_differentiable(example.request, example.visual)
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
                    image_loss.backward()
                    if policy.logits.grad is None:
                        raise RuntimeError("Outer image loss did not reach learned head logits")
                    if not bool(torch.isfinite(policy.logits.grad).all().item()):
                        raise RuntimeError("Non-finite learned-head gradient")
                    gradient_norm = float(policy.logits.grad.norm().detach().item())
                    if config.grad_clip:
                        torch.nn.utils.clip_grad_norm_([policy.logits], config.grad_clip)
                    optimizer.step()
                    if not bool(torch.isfinite(policy.logits).all().item()):
                        raise RuntimeError("Head logits diverged to non-finite values")
                    record = {"epoch": epoch + 1, "sample": sample_index,
                              "loss": float(image_loss.detach()), "grad_norm": gradient_norm,
                              **parts}
                    history.append(record)
                    if progress is not None:
                        progress(len(history), total_items, record)
                    # Do not retain any output tensors/graphs beyond one example.
                    del image_loss, result
        policy.save(config.checkpoint)
        hist = Path(config.history)
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text(json.dumps({"config": vars(config), "image_loss": vars(self.criterion.config),
                                    "samples": history}, indent=2))
        weights = policy.weights(policy.labels, device=policy.logits.device).detach().cpu().tolist()
        return TrainResult(config.checkpoint, tuple(history), policy.labels, tuple(weights))
