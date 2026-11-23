"""Inference-only Gradio UI. No training controls or training dependencies.

Launch: uv run --extra all python examples/ui_inference.py --port 7861
"""
from __future__ import annotations

import gradio as gr

from ui_common import (
    Panel, CANVAS_HEIGHT, serve, canvas_click, canvas_clear,
    parse_editor, canvas_visual, backend_for, checkpoint_choices,
    table_rows_from_labels,
)
from visual_prompt_guidance.backends.attention_catalog import labels_from_catalog
from visual_prompt_guidance import Request
from visual_prompt_guidance.config import save_profile, config_path, profile_settings
from visual_prompt_guidance.runtime import build_engine, model_options, build_observers
from visual_prompt_guidance.experiments import create_run_dir, save_run_config
from visual_prompt_guidance.heads import Learned


def choose_checkpoint(path, catalog=None):
    if not path:
        raise gr.Error('Choose a learned_heads.json file first')
    policy = Learned.load(path)
    labels = list(policy.labels or ())
    selected = list(policy.selected or policy.labels or ())
    layers = list(dict.fromkeys(head.rsplit(':H', 1)[0] for head in labels))
    # Keep the full backend catalog in the table; a checkpoint controls
    # weights, not which layers exist in the model.
    if catalog is not None:
        available = labels_from_catalog(catalog, list(catalog['layers']))
        if set(labels) - set(available):
            raise gr.Error('This checkpoint does not match the selected backend')
    else:
        available = labels
    table = table_rows_from_labels(available, selected)
    save_profile('generate', {
        'head_mode': 'Learned',
        'learned_checkpoint': str(path),
        'selected_layers': layers,
        'selected_heads': selected,
    })
    return (
        gr.update(value='Learned'),
        gr.update(value=table),
        f'Loaded {len(selected)} trained heads across {len(layers)} layers.'
    )


def inference_run(editor, points, *values, panel: Panel):
    try:
        settings = panel.settings(*values)
        img, _ = parse_editor(editor)
        visual = canvas_visual(editor, settings['tool'], points)
        panel.save(*values)
        folder = create_run_dir('generate')
        options = model_options(settings)
        save_run_config(folder, {
            'workflow': 'generate', 'source': 'ui_inference',
            'config_source': str(config_path()), 'settings': settings,
            'model_options': options, 'points': points or []
        })
        img.save(folder / 'input.png')
        if settings['tool'] == 'Scribble' and getattr(visual, 'mask', None) is not None:
            visual.mask.save(folder / 'visual_prompt.png')
        engine = build_engine(settings, backend=backend_for(settings),
                              observers=build_observers(settings, folder))
        request = Request(data={'image': img, 'prompt': settings['prompt'],
                                'object_phrase': settings['object_name']}, options=options)
        result = engine.run(request, visual)
        result.output.save(folder / 'result.png')
        return result.output, f'Saved `{folder}` · {len(result.events)} denoising steps'
    except Exception as e:
        raise gr.Error(f'Inference failed: {e}') from e


def build_demo():
    init = profile_settings('generate')
    with gr.Blocks(title='Visual Prompt Guidance · Inference') as demo:
        gr.Markdown('## Visual Prompt Guidance · Inference\n'
                    'Draw a scribble or click to select a point / box. The model runs with the saved **Generate** profile.')
        save_btn = gr.Button('Save inference settings', size='sm')
        status = gr.Markdown(f'Config: `{config_path()}`')
        points = gr.State([])
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                editor = gr.ImageEditor(label='Input canvas', type='pil', image_mode='RGBA',
                    brush=gr.Brush(default_size=10, colors=['#ff0000'], default_color='#ff0000', color_mode='fixed'),
                    layers=False, transforms=(), height=CANVAS_HEIGHT)
                tool = gr.Radio(['Scribble', 'Point', 'Box'], value=init['tool'], label='Visual prompt tool')
                with gr.Row():
                    clear_shape = gr.Button('Clear point / box')
                    clear_all = gr.Button('Clear all annotations')
            with gr.Column(scale=1):
                output = gr.Image(label='Edited output', height=CANVAS_HEIGHT, type='pil')
                run_status = gr.Markdown('')
        with gr.Row():
            prompt = gr.Textbox(value=init['prompt'], label='Edit instruction', scale=2)
            object_name = gr.Textbox(value=init['object_name'], label='Target object phrase', scale=1)
        panel = Panel('generate', training=False)
        panel.add('tool', tool)
        panel.add('prompt', prompt)
        panel.add('object_name', object_name)
        ckpt_status = gr.Markdown('')
        with gr.Accordion('Diagnostics', open=False):
            panel.add('save_maps', gr.Checkbox(value=init['save_maps'], label='Save attention maps'))
            panel.add('observer_interval', gr.Number(value=init['observer_interval'],
                                                     precision=0, label='Export every N steps'))
        generate = gr.Button('Run inference', variant='primary')

        panel.connect_save(save_btn, status)
        panel.connect_autoload(demo)
        editor.select(canvas_click, [editor, tool, points], [points, editor])
        clear_shape.click(lambda ed, t: canvas_clear(ed, t), [editor, tool], [points, editor])
        clear_all.click(lambda ed, t: canvas_clear(ed, t, clear_brush=True), [editor, tool], [points, editor])
        tool.change(lambda: [], outputs=[points], queue=False)
        panel.refresh_checkpoints.click(
            lambda current: gr.update(choices=list(dict.fromkeys(
                [*([current] if current else []), *checkpoint_choices()])), value=current),
            panel.parts['learned_checkpoint'], panel.parts['learned_checkpoint'], queue=False)
        panel.load_checkpoint.click(
            choose_checkpoint, [panel.parts['learned_checkpoint'], panel.catalog],
            [panel.parts['head_mode'], panel.parts['selected_heads'], ckpt_status], queue=False)
        generate.click(lambda ed, pts, *vals: inference_run(ed, pts, *vals, panel=panel),
                       [editor, points, *panel.components], [output, run_status], concurrency_limit=1)
    return demo


if __name__ == '__main__':
    serve(build_demo, default_port=7861)
