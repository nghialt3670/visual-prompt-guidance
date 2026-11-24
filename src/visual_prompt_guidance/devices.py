"""Host device discovery shared by backends, the Gradio UI, and CLI entry points."""
from __future__ import annotations

import torch


def detect_devices() -> list[str]:
    """Concrete devices on this host, in the order `auto` prefers them."""
    devices = [f'cuda:{index}' for index in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        devices.append('mps')
    devices.append('cpu')
    return devices


def device_choices() -> list[str]:
    return ['auto', *detect_devices()]


def resolve_device(device: str = 'auto') -> str:
    """Map a configured device name to a concrete device on this host."""
    available = detect_devices()
    if device == 'auto':
        return available[0]
    requested = device
    if device == 'cuda':
        device = 'cuda:0'
    if device not in available:
        raise ValueError(f'Device {requested!r} is not available on this host; '
                         f'choose auto or one of {available}')
    return device
