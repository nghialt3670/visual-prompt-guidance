"""Compose public framework objects from shared application settings.

Used identically by the CLI and Gradio. No UI or CLI dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import losses
from .engine import Engine, Guidance
from .objectives import Objective, Loss
from .heads import Uniform, Adaptive, Manual, Learned, HeadSubset
from .transforms import AffinityGrow

MODEL_ALIASES = {"pix2pix": "InstructPix2Pix", "flux2": "FLUX.2 Klein"}


def canonical_model(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def build_backend(settings: Mapping[str, Any]):
    from .backends import Flux2Klein, InstructPix2Pix
    kwargs = {"device": settings.get("device", "auto"), "dtype": settings.get("dtype", "auto")}
    if settings.get("model_id"):
        kwargs["model_id"] = settings["model_id"]
    model = canonical_model(settings["model"])
    if model == "FLUX.2 Klein":
        return Flux2Klein(**kwargs)
    if model == "InstructPix2Pix":
        return InstructPix2Pix(**kwargs)
    raise ValueError(f"Unknown model {model!r}")


def build_objective(settings: Mapping[str, Any]) -> Objective:
    names = settings["selected_losses"]
    return Objective(*(Loss(name, losses.BUILTIN_LOSSES[name],
                            weight=float(settings[f"{name}_w"])) for name in names))


def parse_manual_weights(rows) -> dict[str, float]:
    if rows is None:
        return {}
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    elif hasattr(rows, "tolist") and not isinstance(rows, list):
        rows = rows.tolist()
    result = {}
    for row in rows:
        if row is not None and len(row) >= 2 and str(row[0]).strip():
            result[str(row[0]).strip()] = float(row[1])
    return result


def checkpoint_layers(checkpoint: Learned) -> tuple[str, ...]:
    if not checkpoint.labels:
        raise ValueError("Learned checkpoint contains no heads")
    return tuple(dict.fromkeys(label.rsplit(":H", 1)[0] for label in checkpoint.labels))


def build_policy(settings: Mapping[str, Any], *, training: bool = False):
    mode = "Learned" if training else settings["head_mode"]
    selected = tuple(settings.get("selected_heads") or ())
    if mode == "Learned":
        if training:
            return Learned(selected=selected or None, temperature=float(settings["learned_temperature"]))
        path = str(settings.get("learned_checkpoint") or "").strip()
        if not path:
            raise ValueError("Set learned_checkpoint to a trained learned_heads.json for inference")
        policy = Learned.load(path)
        expected = set(policy.selected or policy.labels)
        if selected and set(selected) != expected:
            raise ValueError("Selected heads do not match checkpoint; clear selected_heads or use matching IDs")
        return policy
    if mode == "Manual":
        return Manual(parse_manual_weights(settings.get("manual_rows")))
    if mode == "Adaptive":
        policy = Adaptive(temperature=float(settings["adaptive_temperature"]),
                          ema=float(settings["adaptive_ema"]), mix=float(settings["adaptive_mix"]))
    elif mode == "Uniform":
        policy = Uniform()
    else:
        raise ValueError(f"Unknown head policy {mode!r}")
    return HeadSubset(policy, selected) if selected else policy


def build_guidance(settings: Mapping[str, Any], policy=None) -> Guidance:
    layers = tuple(settings.get("selected_layers") or ())
    if isinstance(policy, Learned) and policy.labels is not None:
        trained_layers = checkpoint_layers(policy)
        if layers and layers != trained_layers:
            raise ValueError(f"Checkpoint requires attention layers {trained_layers}; configured {layers}")
        layers = trained_layers
    return Guidance(
        strength=float(settings["strength"]), max_steps=int(settings["guide_steps"]),
        layers=layers or None,
        affinity_layers=tuple(settings.get("propagation_layers") or ()) or None,
    )


def build_transform(settings: Mapping[str, Any]):
    if not settings.get("use_propagation"):
        return None
    return AffinityGrow(
        start=int(settings["propagation_start"]), stop=int(settings["propagation_stop"]),
        limit=int(settings["propagation_limit"]), threshold=float(settings["propagation_threshold"]),
        max_fraction=float(settings["propagation_max_fraction"]),
    )


def model_options(settings: Mapping[str, Any]) -> dict[str, Any]:
    options = {"steps": int(settings["steps"]), "seed": int(settings["seed"]),
               "max_side": int(settings["max_side"])}
    if canonical_model(settings["model"]) == "FLUX.2 Klein":
        options.update(mode=settings["flux_mode"],
                       inpaint_strength=float(settings["flux_inpaint_strength"]),
                       cfg=float(settings["flux_cfg"]))
    else:
        options.update(mode="base", cfg=float(settings["pix_text_cfg"]),
                       image_cfg=float(settings["pix_image_cfg"]))
    return options


def build_engine(settings: Mapping[str, Any], *, training: bool = False,
                 backend=None, observers=()) -> Engine:
    policy = build_policy(settings, training=training)
    return Engine(backend if backend is not None else build_backend(settings),
                  build_objective(settings), guidance=build_guidance(settings, policy),
                  policy=policy, transform=build_transform(settings), observers=observers)


def train_config(settings: Mapping[str, Any], checkpoint: Path, history: Path):
    from .training import TrainConfig, ImageLossConfig
    return (
        TrainConfig(epochs=int(settings["training_epochs"]),
                    learning_rate=float(settings["learning_rate"]),
                    optimizer=settings["optimizer_name"], weight_decay=float(settings["weight_decay"]),
                    grad_clip=float(settings["grad_clip"]),
                    checkpoint=str(checkpoint), history=str(history)),
        ImageLossConfig(pixel=settings["image_loss_name"],
                        inside_weight=float(settings["inside_weight"]),
                        outside_weight=float(settings["outside_weight"]),
                        lpips_weight=float(settings["lpips_weight"])),
    )
