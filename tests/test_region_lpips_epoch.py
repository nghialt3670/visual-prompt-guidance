"""LPIPS mask correctness and epoch aggregation regression tests."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visual_prompt_guidance.training import (
    MaskedLpipsRegionLoss, ImageLossConfig, ImageObjective, TrainingExample,
)
from visual_prompt_guidance import Request, Point
from visual_prompt_guidance.experiments import save_training_reports


class FakeSpatialLPIPS(torch.nn.Module):
    def __init__(self, *, net, spatial):
        super().__init__()
        assert net == 'alex' and spatial is True

    def forward(self, image, target):
        # Mock LPIPS spatial output (not a perceptual model). Verify how its
        # spatial scores are masked and normalized without downloading weights.
        return (image - target).abs().mean(1, keepdim=True)


def test_lpips_mask_reduction_is_not_diluted_by_background(monkeypatch):
    monkeypatch.setitem(sys.modules, 'lpips', types.SimpleNamespace(LPIPS=FakeSpatialLPIPS))
    objective = MaskedLpipsRegionLoss()
    image = torch.zeros(1, 3, 16, 16, requires_grad=True)
    target = torch.zeros_like(image)
    mask_small = torch.zeros(1, 1, 16, 16)
    mask_small[:, :, 3:5, 3:5] = 1
    mask_big = torch.zeros(1, 1, 16, 16)
    mask_big[:, :, 3:11, 3:11] = 1
    # Same edit error per selected pixel: area should not shrink the mean.
    noisy_small = (image + mask_small * .25)
    noisy_big = (image + mask_big * .25)
    value_small = objective(noisy_small, target, mask_small)
    value_big = objective(noisy_big, target, mask_big)
    assert float(value_small.detach()) == pytest.approx(.5)
    assert float(value_big.detach()) == pytest.approx(.5)
    value_small.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_lpips_context_not_replaced_with_zeros(monkeypatch):
    class VerifyInputs(FakeSpatialLPIPS):
        def forward(self, image, target):
            assert image[0, :, 0, 0].abs().sum() > 0
            assert target[0, :, 0, 0].abs().sum() > 0
            return super().forward(image, target)
    monkeypatch.setitem(sys.modules, 'lpips', types.SimpleNamespace(LPIPS=VerifyInputs))
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., 6:10, 6:10] = 1
    pred = torch.full((1, 3, 16, 16), .1, requires_grad=True)
    tgt = torch.full_like(pred, .2)
    score = MaskedLpipsRegionLoss()(pred, tgt, mask)
    assert float(score.detach()) == pytest.approx(.2, abs=1e-6)


def test_region_evaluator_no_backprop_required(monkeypatch):
    monkeypatch.setitem(sys.modules, 'lpips', types.SimpleNamespace(LPIPS=FakeSpatialLPIPS))
    ex = TrainingExample(Request({}), Point(0, 0),
                         Image.fromarray(np.full((16,16,3), 80, dtype=np.uint8)))
    objective = ImageObjective(ImageLossConfig(edit_loss='lpips', background_loss='l1'))
    pred = torch.ones(1,3,16,16) * .4
    mask = torch.ones(1,1,16,16)
    score, terms = objective.evaluate(pred, ex, mask)
    assert torch.isfinite(score) and 'edit/lpips' in terms
    with pytest.raises(RuntimeError, match='without an autograd graph'):
        objective(pred, ex, mask)


def test_epoch_average_reports_only_completed_epochs(tmp_path):
    history = [
        {'epoch':1,'sample':0,'loss':0.1},
        {'epoch':1,'sample':1,'loss':0.7,'epoch_mean_loss':0.4},
        {'epoch':2,'sample':1,'loss':0.2},
        {'epoch':2,'sample':0,'loss':0.4,'epoch_mean_loss':0.3},
    ]
    save_training_reports(tmp_path, history, ['L0:H00'], [1.])
    lines = (tmp_path/'epoch_losses.csv').read_text().splitlines()
    assert float(lines[1].split(',')[1]) == pytest.approx(.4)
    assert float(lines[2].split(',')[1]) == pytest.approx(.3)
    assert len((tmp_path/'sample_losses.csv').read_text().splitlines()) == 5


def test_ui_epoch_mean_points(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
    import ui_training
    records = [
        {'step':1, 'epoch':1, 'sample':0, 'loss':.1},
        {'step':2, 'epoch':1, 'sample':1, 'loss':.5, 'epoch_mean_loss':.3},
        {'step':3, 'epoch':2, 'sample':0, 'loss':.2},
    ]
    df = ui_training.loss_plot_data(records)
    assert len(df[df.series.str.startswith('Image ')]) == 3
    average = df[df.series == 'Epoch mean']
    assert len(average) == 1
    assert float(average.iloc[0]['loss']) == pytest.approx(.3)
    assert int(average.iloc[0]['step']) == 2
