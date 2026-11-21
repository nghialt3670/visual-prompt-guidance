"""Persistent user presets for the optional Gradio application.

No model weights or canvas/image data are stored here. The file is a JSON
preference store with independent generate/train snapshots; it is NOT a
framework-level dependency or a cross-user account/preferences service.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = Path("configs/ui_settings.json")
PROFILE_NAMES = ("generate", "train")


def config_path() -> Path:
    return Path(os.environ.get("VPG_UI_CONFIG", str(DEFAULT_CONFIG_PATH)))


def read_store(path: str | Path | None = None) -> dict:
    location = Path(path) if path is not None else config_path()
    if not location.exists():
        return {"schema_version": SCHEMA_VERSION, "active": "generate", "generate": {}, "train": {}}
    try:
        content = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load saved UI settings from {location}: {error}") from error
    if not isinstance(content, dict) or content.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported UI settings format: {location}")
    if content.get("active") not in PROFILE_NAMES:
        raise ValueError(f"Unknown active UI preset in {location}")
    if any(not isinstance(content.get(name, {}), dict) for name in PROFILE_NAMES):
        raise ValueError("Saved UI presets must be JSON objects")
    return {"schema_version": SCHEMA_VERSION,
            "active": content["active"],
            **{name: content.get(name, {}) for name in PROFILE_NAMES}}


def profile_settings(profile: str, *, path: str | Path | None = None) -> dict:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unsupported UI preset {profile!r}")
    return dict(read_store(path)[profile])


def save_profile(profile: str, settings: Mapping[str, Any], *,
                 path: str | Path | None = None) -> Path:
    """Atomically persist one profile while retaining the other one."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unsupported UI preset {profile!r}")
    location = Path(path) if path is not None else config_path()
    content = read_store(location)
    content[profile] = dict(settings)
    content["active"] = profile
    location.parent.mkdir(parents=True, exist_ok=True)
    # First serialize, so invalid component values never partially overwrite settings.
    payload = json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         prefix=f".{location.name}.", dir=location.parent,
                                         delete=False) as handle:
            temp = Path(handle.name)
            handle.write(payload)
        os.replace(temp, location)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return location


def normalize_component_value(value: Any) -> Any:
    """Convert UI Dataframe/array values to plain JSON; never persist file uploads."""
    if hasattr(value, "values") and hasattr(value.values, "tolist"):
        value = value.values.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, list, dict)):
        value = value.tolist()
    if isinstance(value, tuple):
        return [normalize_component_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_component_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): normalize_component_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Cannot save UI setting of type {type(value).__name__}")


def values_for_profile(profile: str, keys: list[str], defaults: Mapping[str, Any], *,
                       path: str | Path | None = None) -> dict[str, Any]:
    saved = profile_settings(profile, path=path)
    return {key: saved.get(key, defaults[key]) for key in keys}
