"""Inference-only Gradio UI. No training controls or training dependencies.

Launch: uv run --extra all python examples/ui_inference.py --port 7861
"""
from __future__ import annotations

import gradio as gr

from ui_common import (
    Panel, CANVAS_HEIGHT, serve, canvas_click, canvas_clear,
    parse_editor, canvas_visual, to_pil_rgb, backend_for, checkpoint_choices,
    table_rows_from_labels,
)
from visual_prompt_guidance.backends.attention_catalog import labels_from_catalog
from visual_prompt_guidance import Request, FullImage
from visual_prompt_guidance.config import save_profile, config_path, profile_settings
from visual_prompt_guidance.runtime import build_engine, build_standard_engine, model_options, build_observers
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


def inference_run(editor, points, *values, panel: Panel, standard_image=None):
    try:
        settings = panel.settings(*values)
        standard = settings.get('inference_mode') == 'standard'
        if standard and standard_image is not None:
            img = to_pil_rgb(standard_image)
        else:
            img, _ = parse_editor(editor)
        visual = FullImage() if standard else canvas_visual(editor, settings['tool'], points)
        if not str(settings['prompt']).strip():
            raise ValueError('Enter an edit instruction')
        panel.save(*values)
        folder = create_run_dir('generate')
        options = model_options(settings)
        if standard and not settings.get('save_maps', False):
            options['skip_attention_tokens'] = True
        save_run_config(folder, {
            'workflow': 'generate', 'source': 'ui_inference',
            'config_source': str(config_path()), 'settings': settings,
            'model_options': options, 'points': points or []
        })
        img.save(folder / 'input.png')
        if not standard and settings['tool'] == 'Scribble' and getattr(visual, 'mask', None) is not None:
            visual.mask.save(folder / 'visual_prompt.png')
        backend = backend_for(settings)
        observers = build_observers(settings, folder)
        engine = (build_standard_engine(settings, backend=backend, observers=observers)
                  if standard else build_engine(settings, backend=backend, observers=observers))
        request = Request(data={'image': img, 'prompt': settings['prompt'],
                                'object_phrase': settings['prompt'] if standard else settings['object_name']},
                          options=options)
        result = engine.run(request, visual)
        result.output.save(folder / 'result.png')
        return result.output, f'{"Standard" if standard else "Guided"} inference · Saved `{folder}` · {len(result.events)} denoising steps'
    except Exception as e:
        raise gr.Error(f'Inference failed: {e}') from e


def build_demo():
    init = profile_settings('generate')
    is_guided = init['inference_mode'] == 'guided'
    with gr.Blocks(title='Visual Prompt Guidance · Inference') as demo:
        gr.Markdown('## Visual Prompt Guidance · Inference')
        with gr.Row():
            mode = gr.Radio([('Standard (no guidance)', 'standard'),
                             ('Visual guidance', 'guided')],
                            value=init['inference_mode'], label='Mode', scale=4)
            save_btn = gr.Button('Save settings', size='sm', scale=1)
        status = gr.Markdown(f'Config: `{config_path()}`')
        points = gr.State([])
        # Only image inputs and output live in this equal-height row.
        # The tool/clear controls are BELOW it; they must not stretch a column.
        with gr.Row(equal_height=True, elem_classes=['vpg-image-pair']):
            with gr.Column(scale=1, min_width=0):
                editor = gr.ImageEditor(
                    label='Input canvas', type='pil', image_mode='RGBA',
                    brush=gr.Brush(default_size=10, colors=['#ff0000'],
                                   default_color='#ff0000', color_mode='fixed'),
                    layers=False, transforms=(), height=CANVAS_HEIGHT,
                    elem_id='vpg-input-editor', visible=is_guided)
                standard_image = gr.Image(
                    label='Input image', type='pil', height=CANVAS_HEIGHT,
                    elem_id='vpg-standard-input', visible=not is_guided)
            with gr.Column(scale=1, min_width=0):
                output = gr.Image(label='Edited output', height=CANVAS_HEIGHT,
                                  type='pil', elem_id='vpg-output-image')
        with gr.Row(visible=is_guided) as visual_controls:
            tool = gr.Radio(['Scribble', 'Point', 'Box'], value=init['tool'],
                            label='Visual prompt', scale=4)
            clear_marks = gr.Button('Clear annotations', size='sm', scale=1)
        run_status = gr.Markdown('')
        with gr.Row():
            prompt = gr.Textbox(value=init['prompt'], label='Edit instruction', scale=3)
            object_group = gr.Group(visible=is_guided)
            with object_group:
                object_name = gr.Textbox(value=init['object_name'],
                                         label='Target object phrase', scale=1)
        panel = Panel('generate', training=False)
        panel.add('inference_mode', mode)
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
        clear_marks.click(lambda ed, t: canvas_clear(ed, t, clear_brush=True),
                          [editor, tool], [points, editor], queue=False)
        tool.change(lambda: [], outputs=[points], queue=False)
        def switch_mode(choice, plain, marked):
            guided = choice == 'guided'
            plain_update = gr.update(visible=not guided)
            editor_update = gr.update(visible=guided)
            # Retain the source while switching input tools. Drawn marks are
            # discarded on entry to Guided, and ignored entirely in Standard.
            if guided and plain is not None:
                img = to_pil_rgb(plain)
                editor_update = gr.update(visible=True, value={
                    'background': img, 'layers': [], 'composite': img.convert('RGBA')})
            elif not guided and marked and marked.get('background') is not None:
                plain_update = gr.update(visible=True, value=to_pil_rgb(marked['background']))
            return (plain_update, editor_update,
                    gr.update(visible=guided), gr.update(visible=guided),
                    gr.update(visible=guided), gr.update(visible=guided),
                    gr.update(value=None), '')

        mode.change(
            switch_mode, [mode, standard_image, editor],
            [standard_image, editor, visual_controls, object_group,
             panel.guidance_panel, panel.propagation_panel, output, run_status],
            queue=False)
        panel.refresh_checkpoints.click(
            lambda current: gr.update(choices=list(dict.fromkeys(
                [*([current] if current else []), *checkpoint_choices()])), value=current),
            panel.parts['learned_checkpoint'], panel.parts['learned_checkpoint'], queue=False)
        panel.load_checkpoint.click(
            choose_checkpoint, [panel.parts['learned_checkpoint'], panel.catalog],
            [panel.parts['head_mode'], panel.parts['selected_heads'], ckpt_status], queue=False)
        generate.click(
            lambda ed, plain, pts, *vals: inference_run(
                ed, pts, *vals, panel=panel, standard_image=plain),
            [editor, standard_image, points, *panel.components],
            [output, run_status], concurrency_limit=1)
    return demo


if __name__ == '__main__':
    serve(build_demo, default_port=7861)
