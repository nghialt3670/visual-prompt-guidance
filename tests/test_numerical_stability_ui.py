"""Regression tests for the failed 2026-11-21 training run."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))

from visual_prompt_guidance.engine import Guidance, rms_update


def test_rms_updater_zero_gradient_has_finite_second_order_derivative():
    # Before v0.8.4: sqrt(mean(g**2)).clamp_min(eps) had a finite
    # forward pass, but produced a NaN when differentiating through g=0.
    h = torch.tensor(0.0, requires_grad=True)
    latent = torch.ones(2, requires_grad=True)
    gradient = h * torch.ones_like(latent)
    updated, rms = rms_update(latent, gradient, Guidance(strength=.01, normalize_gradient=True))
    assert rms == 0.0
    assert torch.equal(updated, latent)
    outer_grad = torch.autograd.grad(updated.sum(), h)[0]
    assert torch.isfinite(outer_grad)


def test_rms_updater_keeps_differentiable_normalization_at_nonzero_gradient():
    h = torch.tensor(0.75, requires_grad=True)
    latent = torch.tensor([0.5, 0.1])
    gradient = torch.stack([h, 2 * h])
    updated, rms = rms_update(latent, gradient, Guidance(strength=.01))
    grad = torch.autograd.grad(updated[0], h)[0]
    assert torch.isfinite(grad)
    assert rms > 0


def test_head_table_uninitialized_uses_all_architecture_heads(monkeypatch):
    import ui_common
    catalog = {
        'layers': ['D0', 'D1'], 'default_layers': ['D0'],
        'default_affinity_layers': ['D0'], 'head_counts': {'D0': 2, 'D1': 3},
    }
    monkeypatch.setattr(ui_common, 'get_attention_catalog', lambda model, mid: catalog)
    panel = ui_common.Panel.__new__(ui_common.Panel)
    panel.profile = 'train'
    panel.training = True
    panel.parts = {key: object() for key in ('model', 'model_id', 'selected_heads')}
    result = panel._updated_settings('FLUX.2 Klein', '', [])
    assert result['selected_heads'] == ['D0:H00', 'D0:H01', 'D1:H00', 'D1:H01', 'D1:H02']
    assert result['selected_layers'] == ['D0', 'D1']
    assert result['head_mode'] == 'Learned'


def test_head_table_explicit_uncheck_remains_invalid(monkeypatch):
    import ui_common
    monkeypatch.setattr(ui_common, 'get_attention_catalog', lambda *args: (_ for _ in ()).throw(
        AssertionError('Should not fetch metadata for an explicit user selection')))
    panel = ui_common.Panel.__new__(ui_common.Panel)
    panel.profile = 'train'
    panel.training = True
    panel.parts = {key: object() for key in ('model', 'model_id', 'selected_heads')}
    with pytest.raises(ValueError, match='Select at least one'):
        panel._updated_settings('FLUX.2 Klein', '', [[False, 'D0', 'H00']])


def test_switched_catalog_without_metadata_returns_five_outputs_and_clears_choices(monkeypatch):
    import ui_common
    monkeypatch.setattr(ui_common, 'get_attention_catalog', lambda *args: (_ for _ in ()).throw(
        RuntimeError('metadata offline')))
    result = ui_common.switched_catalog('FLUX.2 Klein', '')
    assert len(result) == 5
    catalog, table, affinity, layers, info = result
    assert catalog is None
    assert affinity['choices'] == [] and affinity['value'] == []
    assert layers['choices'] == [] and layers['value'] == []
    assert 'metadata offline' in info
