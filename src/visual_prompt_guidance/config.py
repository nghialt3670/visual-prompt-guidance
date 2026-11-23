"""Shared, versioned experiment settings for CLI, Gradio, and other hosts.

The library API remains injection-based; configuration is an optional adapter.
All defaults live here, with two independently persisted workflow profiles.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

SCHEMA_VERSION = 1
PROFILE_NAMES = ("generate", "train")
DEFAULT_CONFIG_PATH = Path("configs/project.json")
LEGACY_CONFIG_PATH = Path("configs/ui_settings.json")

DEFAULT_SETTINGS: dict[str, Any] = {
    "tool": "Scribble", "prompt": "Insert a dog", "object_name": "dog",
    "inference_mode": "guided", "model": "InstructPix2Pix",
    "device": "auto", "dtype": "auto", "model_id": None,
    "selected_losses": ["coverage"], "scoring_losses": [],
    "coverage_w": 1.0, "focal_w": 1.0, "moment_w": 1.0,
    "centerline_w": 1.0, "centroid_w": 1.0, "outside_w": 1.0,
    "steps": 30, "guide_steps": 10, "strength": 0.01, "seed": 42, "max_side": 256,
    "normalize_gradient": True, "updater": "rms",
    "flux_mode": "base", "flux_inpaint_strength": 0.8, "flux_cfg": 4.0,
    "pix_text_cfg": 7.5, "pix_image_cfg": 1.5,
    "head_mode": "Uniform", "selected_layers": [], "selected_heads": [], "manual_rows": [],
    "adaptive_temperature": 0.01, "adaptive_ema": 0.5, "adaptive_mix": 0.1,
    "learned_checkpoint": "", "dataset_path": "", "learned_temperature": 1.0,
    "training_epochs": 1, "learning_rate": 0.001, "optimizer_name": "adamw", "weight_decay": 0.0,
    "train_debug_autograd": False,
    "train_checkpoint_transformer": True, "train_checkpoint_attention": True,
    "train_checkpoint_vae": True, "train_query_chunk": 16,
    "edit_loss_name": "l1", "edit_loss_weight": 1.0,
    "background_loss_name": "l1", "background_loss_weight": 1.0,
    "image_loss_name": "l1", "inside_weight": 1.0, "outside_weight": 1.0,
    "lpips_weight": 0.0, "grad_clip": 1.0,
    "use_propagation": False, "propagation_algorithm": "affinity_grow",
    "propagation_layers": [], "propagation_start": 3,
    "propagation_stop": 10, "propagation_limit": 12, "propagation_threshold": 0.2,
    "propagation_max_fraction": 0.3, "save_maps": False, "observer_interval": 3,
}


def config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("VPG_CONFIG") or os.environ.get("VPG_UI_CONFIG") or DEFAULT_CONFIG_PATH)


def _empty_store() -> dict:
    return {"schema_version": SCHEMA_VERSION, "active": "generate", "generate": {}, "train": {}}


def read_store(path: str | Path | None = None) -> dict:
    location = config_path(path)
    if not location.exists():
        # Read old UI settings on upgrade; first save migrates to project.json.
        if path is None and "VPG_CONFIG" not in os.environ and "VPG_UI_CONFIG" not in os.environ and LEGACY_CONFIG_PATH.is_file():
            location = LEGACY_CONFIG_PATH
        else:
            return _empty_store()
    try:
        data = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load project settings from {location}: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported project settings schema: {location}")
    if data.get("active") not in PROFILE_NAMES:
        raise ValueError("Project settings: invalid active profile")
    for name in PROFILE_NAMES:
        if not isinstance(data.get(name, {}), dict):
            raise ValueError(f"Project settings: {name} must be a JSON object")
        unknown = set(data.get(name, {})) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(f"Project settings: unknown {name} fields {sorted(unknown)}")
    return {"schema_version": SCHEMA_VERSION, "active": data["active"],
            **{name: dict(data.get(name, {})) for name in PROFILE_NAMES}}


def profile_settings(profile: str, *, path: str | Path | None = None) -> dict:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {profile!r}")
    settings = deepcopy(DEFAULT_SETTINGS)
    store_values = deepcopy(read_store(path)[profile])
    settings.update(store_values)
    # Backward compatibility: map the old monolithic image-loss fields into the
    # new edit/background loss configuration when the new keys were not saved yet.
    if "edit_loss_name" not in store_values:
        settings["edit_loss_name"] = settings.get("image_loss_name", "l1")
    if "edit_loss_weight" not in store_values:
        settings["edit_loss_weight"] = settings.get("inside_weight", 1.0)
    if "background_loss_name" not in store_values:
        settings["background_loss_name"] = settings.get("image_loss_name", "l1")
    if "background_loss_weight" not in store_values:
        settings["background_loss_weight"] = settings.get("outside_weight", 1.0)
    return settings


def resolve_settings(profile: str, overrides: Mapping[str, Any] | None = None, *,
                     path: str | Path | None = None) -> dict:
    settings = profile_settings(profile, path=path)
    for key, value in (overrides or {}).items():
        if key not in DEFAULT_SETTINGS:
            raise ValueError(f"Unknown setting {key!r}")
        if value is not None:
            settings[key] = deepcopy(value)
    return settings


def normalize_value(value: Any) -> Any:
    if hasattr(value, "values") and hasattr(value.values, "tolist"):
        value = value.values.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, dict, list)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [normalize_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    raise TypeError(f"Setting contains non-JSON value {type(value).__name__}")


def save_profile(profile: str, settings: Mapping[str, Any], *, path: str | Path | None = None,
                 replace: bool = False) -> Path:
    """Atomic update. By default merge with existing settings (including CLI-only fields)."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {profile!r}")
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    location = config_path(path)
    store = read_store(path)
    merged = {} if replace else dict(store[profile])
    merged.update({k: normalize_value(v) for k, v in settings.items()})
    store[profile] = merged
    store["active"] = profile
    location.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(store, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=location.parent,
                                         prefix=f".{location.name}.", suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            handle.write(payload)
        os.replace(temp, location)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return location
