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
    metadata, table, info = ui_common.auto_load_heads('FLUX.2 Klein', str(model))
    assert metadata['head_counts']['D0']==24
    assert len(table['value']) == 600
    assert all(row[0] for row in table['value'])
    assert '600 / 600 heads' in info


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
    assert 'Guidance attention layers' not in train
    head_rows=train['Attention heads']['value']['data']
    assert head_rows==[[True,'D2','H03']]
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
    assert len(outputs)>4  # includes model-loading, denoising and backward stages
    assert outputs[0][3].startswith('Preparing training configuration')
    assert any('backpropagating through denoising' in item[3] for item in outputs)
    assert any('Denoising step 1/2 completed' in item[3] for item in outputs)
    df=outputs[-1][0]
    assert len(df[df['series']=='Image 1'])==2
    assert list(df[df['series']=='Image 1']['step'])==[1,2]
    assert len(df[df['series']=='Epoch mean'])==2
    assert (checkpoint_log := next((tmp_path/'outputs'/'train').glob('*/training.log'))).is_file()
    assert 'Update 2/2' in checkpoint_log.read_text()
    assert float(outputs[-1][2]['max_head_weight'].max())>0
    checkpoint=Path(outputs[-1][-1])
    assert checkpoint.is_file()
    for file in ('history.json','config.json','loss_curve.png','head_weights.csv',
                 'sample_losses.csv','epoch_losses.csv'):
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
    mode,heads,text=ui_inference.choose_checkpoint(str(weights))
    assert mode['value']=='Learned'
    assert heads['value']==[[True,'L0','H00'],[True,'L0','H01']]
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


def test_attention_table_has_all_layers_and_restores_saved_checkmarks(tmp_path):
    import ui_common
    model=tmp_path/'klein'
    (model/'transformer').mkdir(parents=True)
    (model/'transformer/config.json').write_text(json.dumps({
        'num_layers':2,'num_single_layers':1,'num_attention_heads':3,
    }))
    cat,table,info=ui_common.auto_load_heads('FLUX.2 Klein',str(model))
    assert cat['layers']==['D0','D1','S0']
    assert len(table['value'])==9 and all(row[0] for row in table['value'])
    assert '9 / 9 heads' in info
    # The startup callback uses saved checkmarks; rows still include all heads.
    partial=ui_common.table_rows_from_saved(['D1:H01','S0:H02'])
    _,restored,_=ui_common.auto_load_heads('FLUX.2 Klein',str(model),partial,restore=True)
    assert len(restored['value'])==9
    assert ui_common.selected_heads_from_table(restored['value'])==['D1:H01','S0:H02']
    # Changing backend starts with all heads regardless of previous model.
    _,switched,_=ui_common.auto_load_heads('FLUX.2 Klein',str(model),partial,restore=False)
    assert len(ui_common.selected_heads_from_table(switched['value']))==9
    assert ui_common.table_summary(switched['value'],cat).startswith('**9 / 9 heads**')


def test_ui_automatically_loads_head_metadata_without_restore_button(tmp_path,monkeypatch):
    import ui_inference,ui_training
    monkeypatch.setenv('VPG_CONFIG',str(tmp_path/'project.json'))
    for app in [ui_inference,ui_training]:
        demo=app.build_demo()
        assert any(any(target[1]=='load' for target in d['targets']) for d in demo.config['dependencies'])
        labels=labelled_components(demo)
        assert 'Attention heads' in labels
        assert 'Target layers' in labels
        assert 'Guidance attention layers' not in labels
        assert 'Model' in labels and 'Denoising steps' in labels
        button_values=[c['props'].get('value') for c in demo.config['components']
                       if c['type']=='button']
        assert 'Restore inference settings' not in button_values
        assert 'Restore training settings' not in button_values


def test_layer_selection_checks_only_heads_in_targeted_layers():
    import ui_common
    rows=[[False,'D0','H00'],[True,'D1','H00'],[True,'D1','H01'],[True,'S0','H00']]
    updated=ui_common.select_heads_in_layers(rows,['D0','S0'])
    assert updated['value']==[
        [True,'D0','H00'],[False,'D1','H00'],[False,'D1','H01'],[True,'S0','H00']]


def test_table_selection_derived_engine_layers_and_no_implicit_fallback(tmp_path,monkeypatch):
    pytest.importorskip('gradio')
    import gradio as gr
    import ui_common
    monkeypatch.setenv('VPG_CONFIG',str(tmp_path/'project.json'))
    with gr.Blocks():
        panel=ui_common.Panel('train',training=True)
    vals=[]
    for key,widget in panel.parts.items():
        vals.append([[True,'D1','H01'],[False,'D0','H00'],[True,'S2','H02']]
                    if key=='selected_heads' else widget.value)
    settings=panel.settings(*vals)
    assert settings['selected_heads']==['D1:H01','S2:H02']
    assert settings['selected_layers']==['D1','S2']
    assert settings['head_mode']=='Learned'
    vals[panel.keys.index('selected_heads')]=[[False,'D1','H01']]
    with pytest.raises(ValueError,match='at least one attention head'):
        panel.settings(*vals)


def test_checkpoint_selection_keeps_all_backend_heads_visible(tmp_path,monkeypatch):
    import torch
    import ui_inference,ui_common
    from visual_prompt_guidance.heads import Learned
    labels=('D0:H00','D0:H01','D1:H00','D1:H01')
    learned=Learned(selected=('D0:H00',))
    learned.weights(labels,device=torch.device('cpu'))
    weight_path=tmp_path/'learned_heads.json'
    learned.save(weight_path)
    monkeypatch.setenv('VPG_CONFIG',str(tmp_path/'project.json'))
    catalog={'layers':['D0','D1','S0'],
             'head_counts':{'D0':2,'D1':2,'S0':2},
             'default_layers':['D0'],'default_affinity_layers':['D0']}
    mode,heads,text=ui_inference.choose_checkpoint(str(weight_path),catalog)
    assert mode['value']=='Learned'
    rows=heads['value']
    assert len(rows)==6
    assert ui_common.selected_heads_from_table(rows)==['D0:H00']
    assert '2 layers' in text
    # Checkpoint captured D0 and D1, even though it optimizes only a D0 head.
    with __import__('gradio').Blocks():
        panel=ui_common.Panel('generate',training=False)
    values=[]
    for key,component in panel.parts.items():
        if key=='selected_heads':
            values.append(rows)
        elif key=='head_mode':
            values.append('Learned')
        elif key=='learned_checkpoint':
            values.append(str(weight_path))
        else:
            values.append(component.value)
    settings=panel.settings(*values)
    assert settings['selected_heads']==['D0:H00']
    assert settings['selected_layers']==['D0','D1']


def test_gradio_training_callback_is_streaming_generator(tmp_path, monkeypatch):
    """Regression: wrapping train_stream with a lambda returns one generator object."""
    import inspect
    import ui_training
    monkeypatch.setenv('VPG_CONFIG', str(tmp_path / 'project.json'))
    demo = ui_training.build_demo()
    handlers = [dependency.fn for dependency in demo.fns.values()]
    callbacks = [fn for fn in handlers if getattr(fn, '__name__', None) == 'run_training']
    assert len(callbacks) == 1
    assert inspect.isgeneratorfunction(callbacks[0])
    assert len(demo.config['dependencies']) > 0


def test_propagation_dropdown_can_preprocess_before_async_catalog_finishes(tmp_path, monkeypatch):
    """Regression for Gradio 'value not in choices: []' on Train click."""
    import gradio as gr
    import ui_common
    monkeypatch.setenv('VPG_CONFIG', str(tmp_path / 'project.json'))
    with gr.Blocks():
        panel = ui_common.Panel('train', training=True)
    dropdown = panel.parts['propagation_layers']
    # The default profile has no choices until the async metadata request ends.
    # A valid layer returned by the browser must not crash preprocessing.
    label = 'down_blocks.1.attentions.0.transformer_blocks.0.attn2'
    assert dropdown.preprocess([label]) == [label]


def test_propagation_choices_exist_immediately_for_local_model(tmp_path, monkeypatch):
    """Lightweight local model metadata populates dropdown synchronously."""
    import gradio as gr
    import ui_common
    model = tmp_path / 'klein'
    (model / 'transformer').mkdir(parents=True)
    (model / 'transformer' / 'config.json').write_text(json.dumps({
        'num_layers': 2, 'num_single_layers': 1, 'num_attention_heads': 3,
    }))
    config = tmp_path / 'project.json'
    save_profile('train', {
        'model': 'FLUX.2 Klein', 'model_id': str(model),
        'propagation_layers': ['D1'], 'selected_heads': ['D0:H01'],
    }, path=config)
    monkeypatch.setenv('VPG_CONFIG', str(config))
    with gr.Blocks():
        panel = ui_common.Panel('train', training=True)
    dropdown = panel.parts['propagation_layers']
    assert [choice[1] for choice in dropdown.choices] == ['D0', 'D1', 'S0']
    assert dropdown.preprocess(['D1']) == ['D1']
    assert len(panel.parts['selected_heads'].value['data']) == 9
    assert ui_common.selected_heads_from_table(panel.parts['selected_heads'].value) == ['D0:H01']


def test_train_click_acknowledged_without_waiting_for_queue_or_training_inputs(tmp_path, monkeypatch):
    """A slow metadata load or invalid dropdown must not hide the click feedback."""
    import ui_training
    monkeypatch.setenv('VPG_CONFIG', str(tmp_path / 'project.json'))
    demo = ui_training.build_demo()
    acknowledgement = [fn for fn in demo.fns.values()
                       if getattr(fn.fn, '__name__', '') == 'acknowledge_training']
    assert len(acknowledgement) == 1
    assert len(acknowledgement[0].inputs) == 0
    assert 'Training requested' in acknowledgement[0].fn()[0]
    ack_id = next(key for key, fn in demo.fns.items()
                  if getattr(fn.fn, '__name__', '') == 'acknowledge_training')
    events = [event for event in demo.config['dependencies']
              if event.get('id') == ack_id]
    assert len(events) == 1 and events[0]['queue'] is False
    # The actual generator remains the chained job, not the click handler.
    jobs = [fn for fn in demo.fns.values()
            if getattr(fn.fn, '__name__', '') == 'run_training']
    assert len(jobs) == 1


def test_training_generator_reports_setup_failure_without_silent_exception():
    import ui_training
    class InvalidPanel:
        def settings(self, *values):
            raise ValueError('bad attention selection')
    stream = ui_training.train_stream(None, panel=InvalidPanel())
    assert 'Preparing training configuration' in next(stream)[3]
    assert 'bad attention selection' in next(stream)[3]
