"""Checkpoint and uniform evaluation use the same heads, input and seed."""
import csv
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import compare_heads

from visual_prompt_guidance.heads import Learned


def test_compare_settings_same_head_subset(tmp_path):
    policy = Learned(selected=('D2:H00', 'D2:H02'))
    policy.weights(('D2:H00', 'D2:H01', 'D2:H02'), device=torch.device('cpu'))
    path = tmp_path/'learned_heads.json'
    policy.save(path)
    uniform, learned = compare_heads.build_comparison_settings(
        {'steps': 2, 'guide_steps': 1, 'use_propagation': False}, str(path))
    assert uniform['selected_layers'] == learned['selected_layers'] == ['D2']
    assert uniform['selected_heads'] == learned['selected_heads'] == ['D2:H00', 'D2:H02']
    assert uniform['steps'] == learned['steps'] == 2
    assert uniform['head_mode'] == 'Uniform' and learned['head_mode'] == 'Learned'


def test_compare_main_one_sample_mock_backend(tmp_path, monkeypatch):
    from test_learning import DifferentiableTinyAdapter
    import json
    class MockBackend(DifferentiableTinyAdapter):
        def decode(self, state):
            pixels = (self.decode_tensor(state)[0].detach().clamp(0,1)*255).byte()
            return Image.fromarray(pixels.permute(1,2,0).cpu().numpy())

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare_heads, 'build_backend', lambda settings: MockBackend())
    Image.fromarray(np.full((16,16,3), 100, dtype=np.uint8)).save('input.png')
    Image.fromarray(np.full((16,16,3), 200, dtype=np.uint8)).save('target.png')
    (tmp_path/'manifest.json').write_text(json.dumps([{
        'input_image': 'input.png', 'target_image': 'target.png',
        'prompt': 'add a dog', 'object_phrase': 'dog',
        'visual': {'type':'point', 'x': 5, 'y': 5},
    }]))
    policy = Learned(selected=('L0:H00', 'L0:H01'))
    policy.weights(('L0:H00','L0:H01'),device=torch.device('cpu'))
    with torch.no_grad():
        policy.logits[:] = torch.tensor([1.0,-1.0])
    weights = tmp_path/'learned_heads.json'
    policy.save(weights)
    summary = compare_heads.main(['--checkpoint',str(weights),
                                  '--dataset',str(tmp_path/'manifest.json'),
                                  '--max-samples','1', '--steps','2','--guided-steps','1'])
    assert summary['samples'] == 1
    folder = next((tmp_path/'outputs'/'compare').iterdir())
    assert (folder/'000_uniform.png').is_file()
    assert (folder/'000_learned.png').is_file()
    assert (folder/'paired_losses.csv').is_file()
    with (folder/'paired_losses.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert int(rows[0]['seed']) == 42
    assert float(rows[0]['uniform_loss']) >= 0
    assert float(rows[0]['learned_loss']) >= 0
