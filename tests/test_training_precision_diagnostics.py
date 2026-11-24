"""Regression coverage for automatic MPS training precision and NaN diagnosis."""
from pathlib import Path
import json
import pytest
import torch
import numpy as np
from PIL import Image

from visual_prompt_guidance.runtime import resolve_training_precision, train_config
from visual_prompt_guidance.config import DEFAULT_SETTINGS
from visual_prompt_guidance import (
    Engine, Guidance, Objective, Loss, Learned, HeadTrainer, TrainConfig, Point, losses,
)
from test_learning import DifferentiableTinyAdapter


def test_auto_mps_training_uses_fp32(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert resolve_training_precision({'dtype': 'auto', 'device': 'auto'})['dtype'] == 'fp32'
    assert resolve_training_precision({'dtype': 'auto', 'device': 'mps'})['dtype'] == 'fp32'
    assert resolve_training_precision({'dtype': 'fp16', 'device': 'mps'})['dtype'] == 'fp16'
    assert resolve_training_precision({'dtype': 'auto', 'device': 'cpu'})['dtype'] == 'auto'
    assert resolve_training_precision({'dtype': 'auto', 'device': 'cuda'})['dtype'] == 'auto'
    assert resolve_training_precision({'dtype': 'auto', 'device': 'cuda:1'})['dtype'] == 'auto'


def test_auto_training_on_cuda_host_keeps_auto_precision(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert resolve_training_precision({'dtype': 'auto', 'device': 'auto'})['dtype'] == 'auto'


def test_anomaly_mode_part_of_shared_train_settings(tmp_path):
    settings = dict(DEFAULT_SETTINGS)
    settings['train_debug_autograd'] = True
    cfg, _ = train_config(settings, tmp_path / 'heads.json', tmp_path / 'history.json')
    assert cfg.detect_anomaly is True
    settings['train_debug_autograd'] = False
    cfg, _ = train_config(settings, tmp_path / 'heads.json', tmp_path / 'history.json')
    assert cfg.detect_anomaly is False


def test_anomaly_detects_finite_forward_nonfinite_backward(tmp_path):
    # The same failure signature as the reported run: finite RGB/loss, NaN derivative.
    class BadSecondDerivative(DifferentiableTinyAdapter):
        def decode_tensor(self, state):
            output = super().decode_tensor(state)
            z = state['latent']
            # sqrt(0) has a finite forward but invalid derivative.
            return output + torch.sqrt((z - z).square()).mean()

    policy = Learned()
    engine = Engine(
        BadSecondDerivative(), Objective(Loss('coverage', losses.coverage)),
        policy=policy, guidance=Guidance(strength=.02, max_steps=1)
    )
    Image.fromarray(np.full((16, 16, 3), 80, dtype=np.uint8)).save(tmp_path / 'source.png')
    Image.fromarray(np.full((16, 16, 3), 190, dtype=np.uint8)).save(tmp_path / 'target.png')
    (tmp_path / 'manifest.json').write_text(json.dumps([{
        'input_image': 'source.png', 'target_image': 'target.png',
        'prompt': 'Add a dog', 'object_phrase': 'dog',
        'visual': {'type': 'point', 'x': 4, 'y': 4},
    }]))
    trainer = HeadTrainer(engine, config=TrainConfig(
        epochs=1, detect_anomaly=True,
        checkpoint=str(tmp_path / 'heads.json'), history=str(tmp_path / 'history.json')
    ))
    with pytest.raises(RuntimeError, match='nan|NaN|SqrtBackward'):
        trainer.fit(tmp_path)
