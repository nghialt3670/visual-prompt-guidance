"""Inference-only Gradio UI. No training controls or training dependencies.

Launch: uv run --extra all python examples/ui_inference.py --port 7861
"""
from __future__ import annotations

from pathlib import Path
import gradio as gr

from ui_common import (Panel, CANVAS_HEIGHT, serve, canvas_click, canvas_clear,
                       parse_editor, canvas_visual, backend_for)
from visual_prompt_guidance import Request, FileObserver
from visual_prompt_guidance.config import profile_settings, save_profile, config_path
from visual_prompt_guidance.runtime import build_engine, model_options
from visual_prompt_guidance.experiments import create_run_dir, save_run_config
from visual_prompt_guidance.heads import Learned


def checkpoints():
    paths = sorted(Path('outputs/train').glob('*/learned_heads.json'), reverse=True)
    return [str(path) for path in paths]


def choose_checkpoint(path):
    if not path or not Path(path).is_file():
        raise gr.Error('Choose a valid learned_heads.json file')
    policy = Learned.load(path)
    labels=policy.selected or policy.labels or ()
    layers=tuple(dict.fromkeys(head.rsplit(':H',1)[0] for head in policy.labels))
    # Identity/order comes from the checkpoint; no manually chosen layers are needed.
    save_profile('generate',{'head_mode':'Learned','learned_checkpoint':str(path),
                             'selected_layers':[],'selected_heads':[]})
    return (gr.update(value='Learned'),gr.update(choices=[],value=[]),gr.update(choices=[],value=[]),
            f'Loaded {len(labels)} trained head weights from {len(layers)} layers. '
            'This checkpoint will be used during inference.')


def inference_run(editor, points, *values, panel: Panel):
    try:
        settings=panel.settings(*values)
        img,_=parse_editor(editor)
        visual=canvas_visual(editor,settings['tool'],points)
        # The checkpoint setting is a normal part of the same generate profile.
        panel.save(*values)
        folder=create_run_dir('generate')
        options=model_options(settings)
        save_run_config(folder,{'workflow':'generate','source':'ui_inference',
                                'config_source':str(config_path()),'settings':settings,
                                'model_options':options,'points':points or []})
        img.save(folder/'input.png')
        if settings['tool']=='Scribble' and getattr(visual,'mask',None) is not None:
            visual.mask.save(folder/'visual_prompt.png')
        observer=FileObserver(folder,maps=bool(settings['save_maps']),
                              images=bool(settings['save_maps']),interval=3)
        engine=build_engine(settings,backend=backend_for(settings),observers=[observer])
        request=Request(data={'image':img,'prompt':settings['prompt'],
                              'object_phrase':settings['object_name']},options=options)
        result=engine.run(request,visual)
        result.output.save(folder/'result.png')
        return result.output,f'Saved `{folder}` · {len(result.events)} denoising steps'
    except Exception as e:
        raise gr.Error(f'Inference failed: {e}') from e


def build_demo():
    init=profile_settings('generate')
    with gr.Blocks(title='Visual Prompt Guidance · Inference') as demo:
        gr.Markdown('## Visual Prompt Guidance · Inference\n'
                    'Draw a scribble or click to select a point / box. The model runs with the saved **Generate** profile.')
        with gr.Row():
            save_btn=gr.Button('Save inference settings')
            load_btn=gr.Button('Restore inference settings')
        status=gr.Markdown(f'Config: `{config_path()}`')
        points=gr.State([])
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                editor=gr.ImageEditor(label='Input canvas',type='pil',image_mode='RGBA',
                    brush=gr.Brush(default_size=10,colors=['#ff0000'],default_color='#ff0000',color_mode='fixed'),
                    layers=False,transforms=(),height=CANVAS_HEIGHT)
                tool=gr.Radio(['Scribble','Point','Box'],value=init['tool'],label='Visual prompt tool')
                with gr.Row():
                    clear_shape=gr.Button('Clear point / box')
                    clear_all=gr.Button('Clear all annotations')
            with gr.Column(scale=1):
                output=gr.Image(label='Edited output',height=CANVAS_HEIGHT,type='pil')
                run_status=gr.Markdown('')
        with gr.Row():
            prompt=gr.Textbox(value=init['prompt'],label='Edit instruction',scale=2)
            object_name=gr.Textbox(value=init['object_name'],label='Target object phrase',scale=1)
        panel=Panel('generate',training=False)
        panel.add('tool',tool)
        panel.add('prompt',prompt)
        panel.add('object_name',object_name)
        with gr.Accordion('Trained head weights',open=init['head_mode']=='Learned'):
            gr.Markdown('Select a checkpoint from Training. Choose **Learned** under Head policy to use it.')
            available=checkpoints()
            previous=init['learned_checkpoint']
            choices=list(dict.fromkeys([*([previous] if previous else []),*available]))
            checkpoint=panel.add('learned_checkpoint',gr.Dropdown(
                choices=choices,value=previous or None,allow_custom_value=True,
                label='Head-weight checkpoint (learned_heads.json)'))
            with gr.Row():
                refresh=gr.Button('Refresh training runs')
                load_weights=gr.Button('Load selected checkpoint',variant='secondary')
            ckpt_status=gr.Markdown('')
        panel.add('save_maps',gr.Checkbox(value=init['save_maps'],label='Save per-head heatmaps'))
        generate=gr.Button('Run inference',variant='primary')

        # No training widgets in this app.
        panel.connect_presets(save_button=save_btn,load_button=load_btn,status=status)
        editor.select(canvas_click,[editor,tool,points],[points,editor])
        clear_shape.click(lambda ed,t: canvas_clear(ed,t),[editor,tool],[points,editor])
        clear_all.click(lambda ed,t: canvas_clear(ed,t,clear_brush=True),[editor,tool],[points,editor])
        tool.change(lambda: [],outputs=[points],queue=False)
        refresh.click(lambda current: gr.update(choices=list(dict.fromkeys(
            [*([current] if current else []),*checkpoints()])),value=current),[checkpoint],[checkpoint],queue=False)
        load_weights.click(choose_checkpoint,[checkpoint],
             [panel.parts['head_mode'],panel.parts['selected_layers'],panel.parts['selected_heads'],ckpt_status],queue=False)
        generate.click(lambda ed,pts,*vals: inference_run(ed,pts,*vals,panel=panel),
                       [editor,points,*panel.components],[output,run_status],concurrency_limit=1)
    return demo


if __name__=='__main__':
    serve(build_demo,default_port=7861)
