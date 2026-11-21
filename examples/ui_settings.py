"""Compatibility façade; CLI and Gradio now share visual_prompt_guidance.config."""
from visual_prompt_guidance.config import (
    SCHEMA_VERSION, PROFILE_NAMES, DEFAULT_CONFIG_PATH, DEFAULT_SETTINGS,
    config_path, read_store, save_profile, profile_settings,
    normalize_value as normalize_component_value,
)


def values_for_profile(profile, keys, defaults=None, *, path=None):
    resolved = profile_settings(profile, path=path)
    return {key: resolved[key] for key in keys}
