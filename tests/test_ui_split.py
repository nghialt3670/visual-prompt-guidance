"""Regression coverage for the separate training/inference applications."""
from pathlib import Path
import json
import sys
import numpy as np
import pytest
from PIL import Image

from visual_prompt_guidance.config import profile_settings, save_profile
from visual_prompt_guidance.backends.attention_catalog import flux_catalog, labels_from_catalog
from visual_prompt_guidance.backends.flux2 import Flux2Klein

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))


def labelled_components(demo):
    return {x.get('props',{}).get('label'): x.get('props',{})
            for x in demo.config['components'] if x.get('props',{}).get('label')}


def test_head_metadata_does_not_load_model_weights(tmp_path):
    model=tmp_path/'klein'
    (model/'transformer').mkdir(parents=True)
    (model/'transformer/config.json').write_text(json.dumps({
        'num_layers':5,'num_single_layers':20,'num_attention_heads':24,
    }))
    backend=Flux2Klein(model_id=str(model))
    before=backend._pipe
    catalog=backend.attention_catalog()
    assert len(catalog['layers'])==25
    assert len(labels_from_catalog(catalog,catalog['layers']))==600
    assert backend._pipe is before is None
    import ui_common
    metadata, layers, heads, affinity,info=ui_common.head_refresh('FLUX.2 Klein','auto','auto',str(model),[],[],[])
    assert metadata['head_counts']['D0']==24
    assert 'no weights downloaded' in info


def test_split_apps_restore_independent_profiles_and_do_not_mix_controls(tmp_path,monkeypatch):
    pytest.importorskip('gradio')
    import ui_inference,ui_training
    store=tmp_path/'project.json'
    save_profile('generate',{'steps':37,'head_mode':'Learned','learned_checkpoint':'/tmp/w.json'},path=store)
    save_profile('train',{'steps':4,'training_epochs':9,'learning_rate':0.004,
                         'selected_layers':['D2'],'selected_heads':['D2:H03']},path=store)
    monkeypatch.setenv('VPG_CONFIG',str(store))
    infer=labelled_components(ui_inference.build_demo())
    train=labelled_components(ui_training.build_demo())
    assert infer['Denoising steps']['value']==37
    assert train['Denoising steps']['value']==4
    assert train['Epochs']['value']==9
    assert train['Learning rate']['value']==pytest.approx(.004)
    assert train['Guidance attention layers']['value']==['D2']
    assert train['Include specific heads (optional; empty = all in selected layers)']['value']==['D2:H03']
    assert 'Epochs' not in infer
    assert 'Edit instruction' not in train
    assert 'Head policy' in infer
    assert 'Head policy' not in train


def test_streaming_training_updates_weights_and_writes_reports(tmp_path,monkeypatch):
    pytest.importorskip('gradio')
    import ui_training
    from test_learning import DifferentiableTinyAdapter
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('VPG_CONFIG',str(tmp_path/'configs'/'project.json'))
    Image.fromarray(np.full((16,16,3),110,dtype=np.uint8)).save('input.png')
    Image.fromarray(np.full((16,16,3),200,dtype=np.uint8)).save('target.png')
    (tmp_path/'manifest.json').write_text(json.dumps([{
        'input_image':'input.png','target_image':'target.png',
        'prompt':'Add a dog','object_phrase':'dog',
        'visual':{'type':'point','x':4,'y':4},
    }]))
    settings=profile_settings('train')
    settings.update(model='InstructPix2Pix',steps=2,guide_steps=1,max_side=128,
                    strength=.02,training_epochs=2,selected_layers=[],selected_heads=[],
                    dataset_path=str(tmp_path),learning_rate=.02,optimizer_name='adamw')
    class FakePanel:
        def settings(self,*vals): return settings.copy()
        def save(self,*vals): pass
    monkeypatch.setattr(ui_training,'backend_for',lambda cfg:DifferentiableTinyAdapter())
    outputs=list(ui_training.train_stream(None,panel=FakePanel()))
    assert len(outputs)==4 # initial + two optimizer updates + final
    df=outputs[-1][0]
    assert len(df)==2 and list(df['step'])==[1,2]
    assert float(df['max_head_weight'].max())>0
    checkpoint=Path(outputs[-1][-1])
    assert checkpoint.is_file()
    for file in ('history.json','config.json','loss_curve.png','head_weights.csv'):
        assert (checkpoint.parent/file).is_file()


def test_inference_can_load_checkpoint_and_generate(tmp_path,monkeypatch):
    pytest.importorskip('gradio')
    import ui_inference
    from test_learning import DifferentiableTinyAdapter
    from visual_prompt_guidance.heads import Learned
    import torch
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('VPG_CONFIG',str(tmp_path/'configs'/'project.json'))
    policy=Learned()
    policy.weights(('L0:H00','L0:H01'),device=torch.device('cpu'))
    weights=tmp_path/'weights.json'
    policy.save(weights)
    mode,layers,heads,text=ui_inference.choose_checkpoint(str(weights))
    assert mode['value']=='Learned' and layers['value']==[] and heads['value']==[]
    assert profile_settings('generate')['learned_checkpoint']==str(weights)

    class MockBackend(DifferentiableTinyAdapter):
        def decode(self, state):
            pixels=(self.decode_tensor(state)[0].detach().clamp(0,1)*255).byte()
            return Image.fromarray(pixels.permute(1,2,0).cpu().numpy())
    monkeypatch.setattr(ui_inference,'backend_for',lambda settings: MockBackend())
    settings=profile_settings('generate')
    settings.update(model='InstructPix2Pix',tool='Point',prompt='Add a dog',object_name='dog',
        steps=2,guide_steps=1,max_side=128,strength=.02,selected_losses=['coverage'],
        selected_heads=[],selected_layers=[],save_maps=False)
    class FakePanel:
        def settings(self,*values): return settings
        def save(self,*values): pass
    source=Image.fromarray(np.full((16,16,3),120,dtype=np.uint8))
    result,report=ui_inference.inference_run({'background':source,'layers':[]},[(4,4)],panel=FakePanel())
    assert isinstance(result,Image.Image)
    assert 'Saved' in report
    folder=next((tmp_path/'outputs'/'generate').iterdir())
    for name in ('input.png','result.png','config.json','heads.csv'):
        assert (folder/name).is_file()
