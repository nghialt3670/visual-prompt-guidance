"""Device discovery follows the host's GPUs instead of hardcoded names."""
import sys
import types

import pytest
import torch

from visual_prompt_guidance.backends.flux2 import Flux2Klein
from visual_prompt_guidance.backends.instruct_pix2pix import InstructPix2Pix
from visual_prompt_guidance.devices import detect_devices, device_choices, device_dropdown_choices, resolve_device

GIB = 2**30


def _host(monkeypatch, gpus, mps, free=None):
    free = free or [16 * GIB] * gpus
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: gpus)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda index: (free[index], 16 * GIB))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: f'GPU{index}')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)


def test_every_cuda_gpu_is_listed(monkeypatch):
    _host(monkeypatch, gpus=2, mps=False)
    assert detect_devices() == ['cuda:0', 'cuda:1', 'cpu']
    assert device_choices() == ['auto', 'cuda:0', 'cuda:1', 'cpu']


def test_auto_and_legacy_cuda_map_to_first_gpu(monkeypatch):
    _host(monkeypatch, gpus=2, mps=False)
    assert resolve_device('auto') == 'cuda:0'
    assert resolve_device('cuda') == 'cuda:0'


def test_auto_picks_gpu_with_most_free_memory(monkeypatch):
    _host(monkeypatch, gpus=3, mps=False, free=[1 * GIB, 20 * GIB, 5 * GIB])
    assert resolve_device('auto') == 'cuda:1'


def test_auto_ties_go_to_lowest_gpu_index(monkeypatch):
    _host(monkeypatch, gpus=2, mps=False, free=[5 * GIB, 5 * GIB])
    assert resolve_device('auto') == 'cuda:0'


def test_dropdown_labels_show_gpu_name_and_free_memory(monkeypatch):
    _host(monkeypatch, gpus=2, mps=False, free=[2 * GIB, 23 * GIB])
    assert device_dropdown_choices() == [
        ('auto', 'auto'),
        ('cuda:0 (GPU0, 2.0 GiB free)', 'cuda:0'),
        ('cuda:1 (GPU1, 23.0 GiB free)', 'cuda:1'),
        ('cpu', 'cpu'),
    ]


def test_dropdown_label_survives_memory_query_failure(monkeypatch):
    _host(monkeypatch, gpus=1, mps=False)

    def broken(index):
        raise RuntimeError('driver too old')

    monkeypatch.setattr(torch.cuda, 'mem_get_info', broken)
    assert device_dropdown_choices()[1] == ('cuda:0 (GPU0)', 'cuda:0')


def test_explicit_gpu_index_is_kept(monkeypatch):
    _host(monkeypatch, gpus=2, mps=False)
    assert resolve_device('cuda:1') == 'cuda:1'


def test_out_of_range_gpu_index_is_rejected(monkeypatch):
    _host(monkeypatch, gpus=2, mps=False)
    with pytest.raises(ValueError, match="'cuda:2' is not available"):
        resolve_device('cuda:2')


def test_cuda_request_without_gpu_is_rejected(monkeypatch):
    _host(monkeypatch, gpus=0, mps=False)
    with pytest.raises(ValueError, match="'cuda' is not available"):
        resolve_device('cuda')


def test_auto_falls_back_to_mps_then_cpu(monkeypatch):
    _host(monkeypatch, gpus=0, mps=True)
    assert resolve_device('auto') == 'mps'
    _host(monkeypatch, gpus=0, mps=False)
    assert resolve_device('auto') == 'cpu'


@pytest.fixture
def fake_diffusers(monkeypatch):
    calls = []

    class FakeModule:
        def eval(self):
            return self

        def requires_grad_(self, flag=True):
            return self

    class FakePipe:
        def __init__(self):
            self.transformer = FakeModule()
            self.unet = FakeModule()
            self.vae = FakeModule()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(('load', kwargs))
            return cls()

        def to(self, device):
            calls.append(('to', device))
            return self

        def enable_model_cpu_offload(self, device=None):
            calls.append(('offload', device))

    module = types.ModuleType('diffusers')
    for name in ('Flux2KleinPipeline', 'Flux2KleinInpaintPipeline', 'StableDiffusionInstructPix2PixPipeline'):
        setattr(module, name, FakePipe)
    monkeypatch.setitem(sys.modules, 'diffusers', module)
    return calls


def test_flux2_loads_onto_selected_gpu(monkeypatch, fake_diffusers):
    _host(monkeypatch, gpus=2, mps=False)
    Flux2Klein(device='cuda:1')._load('base')
    assert ('to', 'cuda:1') in fake_diffusers


def test_pix2pix_loads_onto_selected_gpu(monkeypatch, fake_diffusers):
    _host(monkeypatch, gpus=2, mps=False)
    InstructPix2Pix(device='cuda:1')._load()
    assert ('to', 'cuda:1') in fake_diffusers


def test_cpu_offload_targets_selected_gpu(monkeypatch, fake_diffusers):
    _host(monkeypatch, gpus=2, mps=False)
    Flux2Klein(device='cuda:1', cpu_offload=True)._load('base')
    assert ('offload', 'cuda:1') in fake_diffusers


def test_cpu_offload_rejects_non_cuda(monkeypatch, fake_diffusers):
    _host(monkeypatch, gpus=0, mps=False)
    with pytest.raises(ValueError, match='CUDA'):
        Flux2Klein(device='cpu', cpu_offload=True)._load('base')
