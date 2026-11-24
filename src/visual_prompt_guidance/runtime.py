"""Compose public framework objects from shared application settings.

Used identically by the CLI and Gradio. No UI or CLI dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import losses
from .engine import Engine, Guidance, rms_update
from .objectives import Objective, Loss
from .heads import Uniform, Adaptive, Manual, Learned, HeadSubset
from .transforms import AffinityGrow
from .observers import FileObserver

MODEL_ALIASES = {"pix2pix": "InstructPix2Pix", "flux2": "FLUX.2 Klein"}


def canonical_model(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def resolve_training_precision(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve 'auto' to FP32 on Apple MPS for higher-order derivatives.

    This is an *effective runtime* configuration, not an implicit update to the
    user's persisted preferences. Explicit dtype choices are respected.
    Inference keeps its existing autocasting behavior.
    """
    effective = dict(settings)
    if effective.get('dtype', 'auto') != 'auto':
        return effective
    from .devices import detect_devices
    device = effective.get('device', 'auto')
    if device == 'auto':
        device = detect_devices()[0]
    if device == 'mps':
        effective['dtype'] = 'fp32'
    return effective


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


def build_loss_objective(names, settings: Mapping[str, Any]) -> Objective:
    unknown = set(names) - set(losses.BUILTIN_LOSSES)
    if unknown:
        raise ValueError(f"Unknown guidance/scoring losses: {sorted(unknown)}")
    return Objective(*(Loss(name, losses.BUILTIN_LOSSES[name],
                            weight=float(settings[f"{name}_w"])) for name in names))


def build_objective(settings: Mapping[str, Any]) -> Objective:
    return build_loss_objective(settings["selected_losses"], settings)


def build_scoring(settings: Mapping[str, Any]) -> Objective | None:
    """No separate scoring losses = the engine uses the guidance objective."""
    names = settings.get("scoring_losses") or []
    return build_loss_objective(names, settings) if names else None


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
        affinity_layers=(tuple(settings.get("propagation_layers") or ()) or None)
                        if settings.get("use_propagation") else None,
        normalize_gradient=bool(settings.get("normalize_gradient", True)),
    )


def build_transform(settings: Mapping[str, Any]):
    if not settings.get("use_propagation"):
        return None
    algorithm = settings.get("propagation_algorithm", "affinity_grow")
    if algorithm != "affinity_grow":
        raise ValueError(f"Unknown propagation algorithm {algorithm!r}")
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


UPDATERS = {"rms": rms_update}


def build_observers(settings: Mapping[str, Any], folder) -> list:
    """Shared export configuration for CLI and Inference UI."""
    interval = int(settings.get("observer_interval", 3))
    if interval < 1:
        raise ValueError("observer_interval must be positive")
    return [FileObserver(folder, maps=bool(settings["save_maps"]),
                         images=bool(settings["save_maps"]), interval=interval)]


def build_engine(settings: Mapping[str, Any], *, training: bool = False,
                 backend=None, observers=()) -> Engine:
    policy = build_policy(settings, training=training)
    update_name = settings.get("updater", "rms")
    if update_name not in UPDATERS:
        raise ValueError(f"Unknown latent updater {update_name!r}")
    instance = backend if backend is not None else build_backend(settings)
    if training and canonical_model(settings['model']) == 'FLUX.2 Klein':
        # Exact gradient recomputation. No truncation or gradient detachment.
        instance.train_checkpoint_transformer = bool(settings.get('train_checkpoint_transformer', True))
        instance.train_checkpoint_attention = bool(settings.get('train_checkpoint_attention', True))
        instance.train_checkpoint_vae = bool(settings.get('train_checkpoint_vae', True))
        instance.train_query_chunk = int(settings.get('train_query_chunk', 16))
        if instance.train_query_chunk < 1:
            raise ValueError('train_query_chunk must be positive')
    return Engine(instance,
                  build_objective(settings), guidance=build_guidance(settings, policy),
                  policy=policy, scoring=build_scoring(settings),
                  transform=build_transform(settings), observers=observers,
                  updater=UPDATERS[update_name])


def build_standard_engine(settings: Mapping[str, Any], *, backend=None, observers=()) -> Engine:
    """Run the backend's ordinary denoising/CFG path with no attention guidance.

    No attention policy/checkpoint, guidance objective or target propagation is
    executed. Attention is collected only if a map-saving observer requests it.
    The backend still owns the sampling schedule, CFG and decoder.
    """
    return Engine(backend if backend is not None else build_backend(settings), None,
                  guidance=Guidance(strength=0.0, max_steps=0),
                  policy=Uniform(), transform=None, observers=observers)


def train_config(settings: Mapping[str, Any], checkpoint: Path, history: Path):
    from .training import TrainConfig, ImageLossConfig
    return (
        TrainConfig(epochs=int(settings["training_epochs"]),
                    learning_rate=float(settings["learning_rate"]),
                    optimizer=settings["optimizer_name"], weight_decay=float(settings["weight_decay"]),
                    grad_clip=float(settings["grad_clip"]),
                    detect_anomaly=bool(settings.get("train_debug_autograd", False)),
                    checkpoint=str(checkpoint), history=str(history)),
        ImageLossConfig(edit_loss=settings["edit_loss_name"],
                        edit_weight=float(settings["edit_loss_weight"]),
                        background_loss=settings["background_loss_name"],
                        background_weight=float(settings["background_loss_weight"])),
    )
