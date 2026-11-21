"""Optional, lightweight attention metadata for application head selectors.

Reading the backend's component config MUST NOT initialize model weights.
The resulting labels match the IDs used by the capture implementations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def component_config(model_id: str, component: str) -> dict[str, Any]:
    local = Path(model_id).expanduser()
    if local.is_dir():
        path = local / component / 'config.json'
        if not path.is_file():
            raise FileNotFoundError(f"Missing model config: {path}")
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=model_id, filename=f'{component}/config.json'))
    return json.loads(path.read_text(encoding='utf8'))


def flux_catalog(config: dict[str, Any]) -> dict[str, Any]:
    # Config fields come from Flux2Transformer2DModel; do not assume fixed block counts.
    double = int(config['num_layers'])
    single = int(config['num_single_layers'])
    heads = int(config['num_attention_heads'])
    if min(double, single, heads) <= 0:
        raise ValueError('Invalid FLUX.2 attention configuration')
    layers = [*(f'D{i}' for i in range(double)), *(f'S{i}' for i in range(single))]
    defaults = [f'D{i}' for i in range(double)]
    return {
        'layers': layers,
        'default_layers': defaults,
        'default_affinity_layers': defaults[:2],
        'head_counts': {layer: heads for layer in layers},
    }


def pix2pix_catalog(config: dict[str, Any]) -> dict[str, Any]:
    # Construct only the UNet *structure* on PyTorch's meta device. No parameters,
    # pipeline components or model weight files are materialized/downloaded.
    import torch
    from diffusers import UNet2DConditionModel
    from .instruct_pix2pix_attention import cross_attention_layers, default_layers

    with torch.device('meta'):
        unet = UNet2DConditionModel.from_config(config)
    modules = cross_attention_layers(unet)
    if not modules:
        raise ValueError('This U-Net configuration exposes no cross-attention heads')
    default = list(default_layers(unet))
    return {
        'layers': list(modules),
        'default_layers': default,
        'default_affinity_layers': default,
        'head_counts': {name: int(module.heads) for name, module in modules.items()},
    }


def labels_from_catalog(catalog: dict[str, Any], selected_layers: list[str] | None = None) -> list[str]:
    # None means unspecified; [] is intentionally empty (don't silently reselect defaults).
    layers = catalog['default_layers'] if selected_layers is None else selected_layers
    unknown = set(layers) - set(catalog['layers'])
    if unknown:
        raise ValueError(f'Unknown attention layers: {sorted(unknown)}')
    return [f'{layer}:H{idx:02d}' for layer in layers
            for idx in range(int(catalog['head_counts'][layer]))]
