"""Shared *application* widgets and interactions; never imported by the core library.

Both apps persist their own profile in configs/project.json. Each backend
provides a lightweight attention catalog; the selector never loads weights.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from visual_prompt_guidance import losses
from visual_prompt_guidance.config import config_path, profile_settings, save_profile, normalize_value
from visual_prompt_guidance.runtime import build_backend
from visual_prompt_guidance.backends.attention_catalog import labels_from_catalog
from visual_prompt_guidance.prompts import Point, Box, Scribble

CANVAS_HEIGHT = 500


def serve(demo, *, default_port: int):
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=default_port)
    p.add_argument('--config', default=None, help='Shared configs/project.json path')
    args = p.parse_args()
    if args.config:
        import os
        os.environ['VPG_CONFIG'] = args.config
    # Config must be set before build_demo; callers pass its factory to this function.
    demo().queue().launch(server_name=args.host, server_port=args.port)


@lru_cache(maxsize=4)
def cached_backend(model, device, dtype, model_id):
    return build_backend({'model': model, 'device': device, 'dtype': dtype, 'model_id': model_id})


def backend_for(settings):
    return cached_backend(settings['model'], settings['device'], settings['dtype'], settings.get('model_id'))


def to_pil_rgb(value):
    return value.convert('RGB') if isinstance(value, Image.Image) else Image.fromarray(np.asarray(value)).convert('RGB')


def to_pil_rgba(value):
    return value.convert('RGBA') if isinstance(value, Image.Image) else Image.fromarray(np.asarray(value)).convert('RGBA')


def parse_editor(editor):
    if not editor or editor.get('background') is None:
        raise ValueError('Upload a source image first')
    bg = to_pil_rgb(editor['background'])
    return bg, [to_pil_rgba(layer) for layer in (editor.get('layers') or [])]


def draw_canvas(background, layers, tool, points):
    composited = background.convert('RGBA')
    for layer in layers:
        composited = Image.alpha_composite(composited, layer.resize(background.size))
    overlay = Image.new('RGBA', background.size)
    draw = ImageDraw.Draw(overlay)
    if tool == 'Point' and points:
        x,y=points[-1]
        draw.ellipse((x-6,y-6,x+6,y+6),outline='red',width=3)
        draw.line((x-12,y,x+12,y),fill='red',width=2)
        draw.line((x,y-12,x,y+12),fill='red',width=2)
    elif tool == 'Box' and len(points) == 1:
        x,y=points[0]
        draw.ellipse((x-5,y-5,x+5,y+5),fill='orange')
    elif tool == 'Box' and len(points) >= 2:
        x0,y0=points[-2]; x1,y1=points[-1]
        draw.rectangle((min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)),outline='orange',width=4)
    return {'background': background, 'layers': layers, 'composite': Image.alpha_composite(composited,overlay)}


def canvas_click(editor, tool, points, evt: gr.SelectData):
    bg,layers=parse_editor(editor)
    points=list(points or [])
    if tool != 'Scribble':
        if not isinstance(evt.index, (tuple,list)) or len(evt.index)<2:
            return points, draw_canvas(bg,layers,tool,points)
        point=tuple(float(v) for v in evt.index[:2])
        points = [point] if tool == 'Point' or len(points)>=2 else [*points, point]
    return points, draw_canvas(bg,layers,tool,points)


def canvas_clear(editor, tool, *, clear_brush: bool = False):
    bg,layers=parse_editor(editor)
    if clear_brush:
        layers=[]
    return [], draw_canvas(bg,layers,tool,[])


def canvas_visual(editor, tool, points):
    bg,layers = parse_editor(editor)
    if tool == 'Point':
        if not points: raise ValueError('Click the input canvas to choose a point')
        return Point(*points[-1])
    if tool == 'Box':
        if len(points)<2: raise ValueError('Click two corners on the input canvas')
        return Box(*points[-2],*points[-1],margin=16)
    mask=Image.new('L',bg.size,0)
    for layer in layers:
        mask=Image.fromarray(np.maximum(np.asarray(mask),np.asarray(layer.getchannel('A'))),'L')
    if not np.asarray(mask).any():
        raise ValueError('Draw a scribble on the input canvas')
    return Scribble(mask=mask)


def save_settings(profile: str, keys, *values):
    values = {k: normalize_value(v) for k,v in zip(keys,values)}
    save_profile(profile,values)
    return values


def restore_settings(profile, keys):
    saved=profile_settings(profile)
    result=[]
    for key in keys:
        value=saved[key]
        if key in {'selected_layers','selected_heads','propagation_layers'}:
            result.append(gr.update(choices=value,value=value))
        else:
            result.append(gr.update(value=value))
    return result


def get_attention_catalog(model, device, dtype, model_id):
    backend = cached_backend(model,device,dtype,model_id or None)
    # This calls only component_config() + metadata construction, never _load().
    return backend.attention_catalog()


def head_refresh(model,device,dtype,model_id, selected_layers, selected_heads, affinity_layers):
    try:
        catalog=get_attention_catalog(model,device,dtype,model_id)
        layers=list(catalog['layers'])
        chosen=[x for x in (selected_layers or []) if x in layers]
        if not chosen:
            chosen=list(catalog['default_layers'])
        labels=labels_from_catalog(catalog, chosen)
        keep=[x for x in (selected_heads or []) if x in labels]
        affinity=[x for x in (affinity_layers or catalog['default_affinity_layers']) if x in layers]
        return (catalog, gr.update(choices=layers,value=chosen),
                gr.update(choices=labels,value=keep),
                gr.update(choices=layers,value=affinity),
                f'{len(layers)} layers, {len(labels)} individual heads available. Metadata only; no weights downloaded.')
    except Exception as error:
        return (None,gr.update(),gr.update(),gr.update(),f'Cannot load head metadata: {error}')


def heads_for_layers(catalog,selected_layers, selected_heads):
    if not catalog:
        return gr.update(), 'Load the available layers first.'
    labels=labels_from_catalog(catalog,list(selected_layers or []))
    keep=[x for x in (selected_heads or []) if x in labels]
    return gr.update(choices=labels,value=keep), f'{len(labels)} individual heads in selected layers.'


def manual_rows_from_selection(selected, old_rows):
    old={str(row[0]): float(row[1]) for row in (old_rows.values.tolist() if hasattr(old_rows,'values') else old_rows or [])
         if row is not None and len(row)>=2 and str(row[0]).strip()}
    return [[head,old.get(head,1.0)] for head in (selected or [])]


class Panel:
    def __init__(self, profile: str, training: bool):
        self.profile=profile
        self.training=training
        self.parts={}
        self.catalog=None
        self._create()

    def add(self, key, component):
        self.parts[key]=component
        return component

    @property
    def keys(self):
        return list(self.parts)

    @property
    def components(self):
        return list(self.parts.values())

    def settings(self, *values):
        base=profile_settings(self.profile)
        base.update({k:normalize_value(v) for k,v in zip(self.keys,values)})
        if self.training:
            base['head_mode']='Learned'
        return base

    def save(self, *values):
        return save_settings(self.profile,self.keys,*values)

    def _create(self):
        init=profile_settings(self.profile)
        def add(k,comp): return self.add(k,comp)
        with gr.Row():
            add('model',gr.Dropdown(['InstructPix2Pix','FLUX.2 Klein'],value=init['model'],label='Backend'))
            add('device',gr.Dropdown(['auto','cuda','mps','cpu'],value=init['device'],label='Device'))
            add('dtype',gr.Dropdown(['auto','fp32','fp16','bf16'],value=init['dtype'],label='Model dtype'))
            add('model_id',gr.Textbox(value=init['model_id'] or '',label='Model ID / local path',placeholder='Default checkpoint'))
        with gr.Accordion('Sampling and guidance', open=True):
            with gr.Row():
                add('steps',gr.Slider(1,60,value=init['steps'],step=1,label='Denoising steps'))
                add('guide_steps',gr.Slider(0,60,value=init['guide_steps'],step=1,label='Guided steps'))
                add('max_side',gr.Slider(64,1024,value=init['max_side'],step=32,label='Max image side (px)'))
            with gr.Row():
                add('strength',gr.Number(value=init['strength'],label='Guidance strength'))
                add('seed',gr.Number(value=init['seed'],precision=0,label='Seed'))
            add('selected_losses',gr.CheckboxGroup(list(losses.BUILTIN_LOSSES),value=init['selected_losses'],label='Guidance losses'))
            with gr.Row():
                for k in ('coverage_w','focal_w','moment_w'):
                    add(k,gr.Number(value=init[k],label=k.removesuffix('_w')))
            with gr.Row():
                for k in ('centerline_w','centroid_w','outside_w'):
                    add(k,gr.Number(value=init[k],label=k.removesuffix('_w')))
        with gr.Accordion('Backend-specific sampling', open=False):
            flux=gr.Group(visible=init['model']=='FLUX.2 Klein')
            with flux:
                with gr.Row():
                    add('flux_mode',gr.Dropdown(['base','inpaint'],value=init['flux_mode'],label='FLUX mode'))
                    add('flux_inpaint_strength',gr.Slider(.1,1,value=init['flux_inpaint_strength'],step=.05,label='Inpaint strength'))
                    add('flux_cfg',gr.Number(value=init['flux_cfg'],label='FLUX CFG'))
            pix=gr.Group(visible=init['model']=='InstructPix2Pix')
            with pix:
                with gr.Row():
                    add('pix_text_cfg',gr.Number(value=init['pix_text_cfg'],label='Pix2Pix text CFG'))
                    add('pix_image_cfg',gr.Number(value=init['pix_image_cfg'],label='Pix2Pix image CFG'))
        self.parts['model'].change(lambda model:(gr.update(visible=model=='FLUX.2 Klein'),gr.update(visible=model=='InstructPix2Pix')),
                                   self.parts['model'],[flux,pix])
        with gr.Accordion('Attention heads',open=False):
            self.catalog=gr.State(None)
            load_heads=gr.Button('Load layers and heads (metadata only)')
            info=gr.Markdown('Select a backend, then load its attention metadata.')
            add('selected_layers',gr.CheckboxGroup(choices=init['selected_layers'],value=init['selected_layers'],label='Guidance attention layers'))
            add('selected_heads',gr.CheckboxGroup(choices=init['selected_heads'],value=init['selected_heads'],label='Include specific heads (optional; empty = all in selected layers)'))
            if not self.training:
                add('head_mode',gr.Radio(['Uniform','Adaptive','Manual','Learned'],value=init['head_mode'],label='Head policy'))
                manual=gr.Group(visible=init['head_mode']=='Manual')
                with manual:
                    add('manual_rows',gr.Dataframe(headers=['head','weight'],datatype=['str','number'],
                        row_count=(len(init['manual_rows']),'dynamic'),col_count=(2,'fixed'),
                        value=init['manual_rows'],interactive=True,label='Manual head weights'))
                adaptive=gr.Group(visible=init['head_mode']=='Adaptive')
                with adaptive:
                    with gr.Row():
                        for k in ('adaptive_temperature','adaptive_ema','adaptive_mix'):
                            add(k,gr.Number(value=init[k],label=k.replace('adaptive_','')))
                # Checkpoint selection is loaded in the inference app; no training UI here.
                self.parts['head_mode'].change(lambda mode:(gr.update(visible=mode=='Manual'),gr.update(visible=mode=='Adaptive')),
                    self.parts['head_mode'],[manual,adaptive])
                self.parts['selected_heads'].change(manual_rows_from_selection,
                    [self.parts['selected_heads'],self.parts['manual_rows']],[self.parts['manual_rows']])
            else:
                gr.Markdown('Individual heads selected here are optimized by the trainer. Leave empty to train all heads of the selected layers.')
                add('learned_temperature',gr.Number(value=init['learned_temperature'],label='Learned head softmax temperature'))
            add('propagation_layers',gr.CheckboxGroup(choices=init['propagation_layers'],value=init['propagation_layers'],label='Self-attention affinity layers'))
            load_heads.click(head_refresh,
                [self.parts[k] for k in ('model','device','dtype','model_id','selected_layers','selected_heads','propagation_layers')],
                [self.catalog,self.parts['selected_layers'],self.parts['selected_heads'],self.parts['propagation_layers'],info])
            self.parts['selected_layers'].change(heads_for_layers,
                [self.catalog,self.parts['selected_layers'],self.parts['selected_heads']],
                [self.parts['selected_heads'],info])
            with gr.Row():
                all_layers=gr.Button('All layers')
                default_layers=gr.Button('Backend default layers')
                all_heads=gr.Button('All heads explicitly')
                clear_heads=gr.Button('Clear head filter')
            all_layers.click(lambda cat: gr.update(value=list(cat['layers'])) if cat else gr.update(),
                             [self.catalog],[self.parts['selected_layers']],queue=False)
            default_layers.click(lambda cat: gr.update(value=list(cat['default_layers'])) if cat else gr.update(),
                                 [self.catalog],[self.parts['selected_layers']],queue=False)
            all_heads.click(lambda cat,layers: gr.update(value=labels_from_catalog(cat,list(layers or []))) if cat else gr.update(),
                            [self.catalog,self.parts['selected_layers']],[self.parts['selected_heads']],queue=False)
            clear_heads.click(lambda: [],outputs=[self.parts['selected_heads']],queue=False)
            def clear_catalog():
                return None,gr.update(choices=[],value=[]),gr.update(choices=[],value=[]),gr.update(choices=[],value=[])
            self.parts['model'].input(clear_catalog,outputs=[self.catalog,self.parts['selected_layers'],
                self.parts['selected_heads'],self.parts['propagation_layers']],queue=False)
        with gr.Accordion('Propagation',open=False):
            add('use_propagation',gr.Checkbox(value=init['use_propagation'],label='Enable affinity propagation'))
            with gr.Row():
                for k in ('propagation_start','propagation_stop','propagation_limit'):
                    add(k,gr.Number(value=init[k],precision=0,label=k.replace('propagation_','')))
            with gr.Row():
                for k in ('propagation_threshold','propagation_max_fraction'):
                    add(k,gr.Number(value=init[k],label=k.replace('propagation_','')))

    def connect_presets(self, *, save_button, load_button, status):
        save_button.click(lambda *vals:(self.save(*vals),f'Saved {self.profile} settings to {config_path()}')[1],
                          self.components,[status],queue=False)
        outputs=[*self.components,status,self.catalog]
        def restore():
            return (*restore_settings(self.profile,self.keys),f'Restored {self.profile} settings.',None)
        load_button.click(restore,outputs=outputs,queue=False)
