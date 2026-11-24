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


def device_dropdown_choices() -> list[tuple[str, str]]:
    """(label, value) pairs for the UI; GPU labels show name and free memory."""
    options = [('auto', 'auto')]
    for device in detect_devices():
        options.append((_device_label(device), device))
    return options


def _device_label(device: str) -> str:
    if not device.startswith('cuda:'):
        return device
    index = int(device.split(':', 1)[1])
    name = torch.cuda.get_device_name(index)
    try:
        free_gib = torch.cuda.mem_get_info(index)[0] / 2**30
    except RuntimeError:
        return f'{device} ({name})'
    return f'{device} ({name}, {free_gib:.1f} GiB free)'


def _freest_cuda() -> str | None:
    """CUDA device with the most free memory; ties go to the lowest index."""
    count = torch.cuda.device_count()
    if count == 0:
        return None
    free = [torch.cuda.mem_get_info(index)[0] for index in range(count)]
    return f'cuda:{max(range(count), key=lambda index: free[index])}'


def resolve_device(device: str = 'auto') -> str:
    """Map a configured device name to a concrete device on this host."""
    available = detect_devices()
    if device == 'auto':
        return _freest_cuda() or available[0]
    requested = device
    if device == 'cuda':
        device = 'cuda:0'
    if device not in available:
        raise ValueError(f'Device {requested!r} is not available on this host; '
                         f'choose auto or one of {available}')
    return device
