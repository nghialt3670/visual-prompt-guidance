"""Standard inference: no attention, no visual prompt, and compact UI."""
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
import gradio as gr

from visual_prompt_guidance import FullImage, Request
from visual_prompt_guidance.config import profile_settings
from visual_prompt_guidance.runtime import build_standard_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import ui_inference
import ui_common
from test_learning import DifferentiableTinyAdapter


class TrackingBackend(DifferentiableTinyAdapter):
    def __init__(self):
        super().__init__()
        self.attention_requests = []

    def forward(self, state, step, *, attention):
        self.attention_requests.append(attention)
        return super().forward(state, step, attention=attention)

    def decode(self, state):
        img = self.decode_tensor(state)[0].detach().clamp(0, 1).mul(255).byte()
        return Image.fromarray(img.permute(1, 2, 0).cpu().numpy())


def test_standard_engine_uses_native_backend_without_attention():
    backend = TrackingBackend()
    settings = profile_settings('generate')
    # A learned policy with no checkpoint would fail normal guidance; in
    # standard mode the engine must never attempt to load it.
    settings.update(head_mode='Learned', learned_checkpoint='', guide_steps=10,
                    use_propagation=True, steps=2, max_side=128, strength=.1)
    engine = build_standard_engine(settings, backend=backend)
    source = Image.new('RGB', (16, 16), 'gray')
    request = Request({'image': source, 'prompt': 'add a dog',
                       'object_phrase': 'add a dog'}, {'steps': 2, 'max_side': 128})
    result = engine.run(request, FullImage())
    assert result.output.size == source.size
    assert len(result.events) == 2
    assert backend.attention_requests == [None, None]
    assert all(not event.weights and not event.losses for event in result.events)
    assert result.target.edit_mask.shape[-2:] == result.target.density.shape[-2:]
    assert bool((result.target.edit_mask == 1).all())


def test_standard_inference_without_point_box_scribble(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('VPG_CONFIG', str(tmp_path / 'project.json'))
    backend = TrackingBackend()
    monkeypatch.setattr(ui_inference, 'backend_for', lambda settings: backend)
    settings = profile_settings('generate')
    settings.update(inference_mode='standard', head_mode='Learned',
                    learned_checkpoint='', tool='Scribble',
                    prompt='add a dog', object_name='',
                    selected_heads=[], steps=2, max_side=128, save_maps=False)

    class FakePanel:
        def settings(self, *args): return settings
        def save(self, *args): pass

    img = Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8))
    output, report = ui_inference.inference_run(None, [], panel=FakePanel(), standard_image=img)
    assert isinstance(output, Image.Image)
    assert output.size == img.size
    assert 'Standard inference' in report
    assert backend.attention_requests == [None, None]


def test_standard_inference_saves_attention_maps_when_requested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('VPG_CONFIG', str(tmp_path / 'project.json'))
    backend = TrackingBackend()
    monkeypatch.setattr(ui_inference, 'backend_for', lambda settings: backend)
    settings = profile_settings('generate')
    settings.update(inference_mode='standard', prompt='add a dog', object_name='',
                    steps=2, max_side=128, save_maps=True, observer_interval=1)

    class FakePanel:
        def settings(self, *args): return settings
        def save(self, *args): pass

    image = Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8))
    ui_inference.inference_run(None, [], panel=FakePanel(), standard_image=image)
    run = next((tmp_path / 'outputs' / 'generate').iterdir())
    assert backend.attention_requests[0] is not None
    assert (run / 'head_maps' / 'step_000' / 'head_000.npy').is_file()
    assert (run / 'head_maps' / 'step_000' / 'head_000.png').is_file()


def test_standard_mode_can_save_without_head_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv('VPG_CONFIG', str(tmp_path / 'project.json'))
    with gr.Blocks():
        panel = ui_common.Panel('generate', training=False)
    vals=[]
    for key, component in panel.parts.items():
        if key == 'selected_heads': vals.append([])
        else: vals.append(component.value)
    vals.append('standard')
    panel.add('inference_mode', gr.State('standard'))
    settings = panel.settings(*vals)
    assert settings['inference_mode'] == 'standard'
    assert settings['selected_heads'] == []


def test_ui_has_one_clear_button_and_equal_height_image_pair(tmp_path, monkeypatch):
    monkeypatch.setenv('VPG_CONFIG', str(tmp_path / 'project.json'))
    demo = ui_inference.build_demo()
    components = demo.config['components']
    labels = {c['props'].get('label'): c['props'] for c in components
              if c.get('props', {}).get('label')}
    assert labels['Input canvas']['height'] == labels['Edited output']['height'] == 500
    assert labels['Input image']['height'] == 500
    buttons = [c['props'].get('value') for c in components if c['type'] == 'button']
    assert 'Clear annotations' in buttons
    assert 'Clear point / box' not in buttons
    assert 'Clear all annotations' not in buttons
    assert labels['Mode']['value'] == 'guided'


def test_mode_switch_copies_source_without_annotations(tmp_path, monkeypatch):
    monkeypatch.setenv('VPG_CONFIG', str(tmp_path / 'project.json'))
    demo = ui_inference.build_demo()
    callbacks = [f.fn for f in demo.fns.values() if getattr(f.fn, '__name__', '') == 'switch_mode']
    assert len(callbacks) == 1
    switch = callbacks[0]
    image = Image.new('RGB', (36, 24), (110, 120, 130))
    to_standard = switch('standard', None, {'background': image, 'layers': [image]})
    assert to_standard[0]['value'].size == (36, 24)
    assert to_standard[0]['visible'] is True
    assert to_standard[1]['visible'] is False
    assert to_standard[2]['visible'] is False
    assert to_standard[6]['value'] is None
    to_guided = switch('guided', image, None)
    assert to_guided[1]['value']['background'].size == (36, 24)
    assert to_guided[1]['value']['layers'] == []
    assert to_guided[0]['visible'] is False
    assert to_guided[1]['visible'] is True
    assert to_guided[2]['visible'] is True


def test_saved_standard_mode_hides_visual_guidance_ui(tmp_path, monkeypatch):
    from visual_prompt_guidance.config import save_profile
    config = tmp_path / 'project.json'
    save_profile('generate', {'inference_mode':'standard', 'head_mode':'Learned'}, path=config)
    monkeypatch.setenv('VPG_CONFIG', str(config))
    demo=ui_inference.build_demo()
    components=demo.config['components']
    modes=[c for c in components if c.get('props',{}).get('label') == 'Mode']
    assert len(modes) == 1 and modes[0]['props']['value'] == 'standard'
    assert next(c for c in components if c.get('props',{}).get('label') == 'Input image')['props']['visible'] is True
