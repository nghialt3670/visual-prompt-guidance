"""Shared *application* widgets and interactions; never imported by the core library.

Both apps persist their own profile in configs/project.json. Each backend
provides a lightweight attention catalog; the selector never loads weights.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from visual_prompt_guidance import losses
from visual_prompt_guidance.config import config_path, profile_settings, save_profile, normalize_value
from visual_prompt_guidance.runtime import build_backend
from visual_prompt_guidance.devices import device_choices
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
    demo().queue().launch(server_name=args.host, server_port=args.port)


@lru_cache(maxsize=4)
def cached_backend(model, device, dtype, model_id):
    return build_backend({'model': model, 'device': device, 'dtype': dtype, 'model_id': model_id})


def backend_for(settings):
    return cached_backend(settings['model'], settings['device'], settings['dtype'], settings.get('model_id'))


def checkpoint_choices():
    return [str(path) for path in sorted(Path('outputs/train').glob('*/learned_heads.json'), reverse=True)]


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
        x, y = points[-1]
        draw.ellipse((x-6, y-6, x+6, y+6), outline='red', width=3)
        draw.line((x-12, y, x+12, y), fill='red', width=2)
        draw.line((x, y-12, x, y+12), fill='red', width=2)
    elif tool == 'Box' and len(points) == 1:
        x, y = points[0]
        draw.ellipse((x-5, y-5, x+5, y+5), fill='orange')
    elif tool == 'Box' and len(points) >= 2:
        x0, y0 = points[-2]
        x1, y1 = points[-1]
        draw.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), outline='orange', width=4)
    return {'background': background, 'layers': layers, 'composite': Image.alpha_composite(composited, overlay)}


def canvas_click(editor, tool, points, evt: gr.SelectData):
    bg, layers = parse_editor(editor)
    points = list(points or [])
    if tool != 'Scribble':
        if not isinstance(evt.index, (tuple, list)) or len(evt.index) < 2:
            return points, draw_canvas(bg, layers, tool, points)
        point = tuple(float(v) for v in evt.index[:2])
        points = [point] if tool == 'Point' or len(points) >= 2 else [*points, point]
    return points, draw_canvas(bg, layers, tool, points)


def canvas_clear(editor, tool, *, clear_brush: bool = False):
    """Reset point, box, and scribble marks without removing the input image."""
    if not editor or editor.get('background') is None:
        return [], gr.update()
    bg, layers = parse_editor(editor)
    if clear_brush:
        layers = []
    return [], draw_canvas(bg, layers, tool, [])


def canvas_visual(editor, tool, points):
    bg, layers = parse_editor(editor)
    if tool == 'Point':
        if not points:
            raise ValueError('Click the input canvas to choose a point')
        return Point(*points[-1])
    if tool == 'Box':
        if len(points) < 2:
            raise ValueError('Click two corners on the input canvas')
        return Box(*points[-2], *points[-1], margin=16)
    mask = Image.new('L', bg.size, 0)
    for layer in layers:
        mask = Image.fromarray(np.maximum(np.asarray(mask), np.asarray(layer.getchannel('A'))), 'L')
    if not np.asarray(mask).any():
        raise ValueError('Draw a scribble on the input canvas')
    return Scribble(mask=mask)


def split_head_label(label: str):
    layer, marker, head = str(label).rpartition(':H')
    return (layer, f'H{head}') if marker else (str(label), '')


def _table_rows(value):
    if hasattr(value, 'values') and hasattr(value.values, 'tolist'):
        return value.values.tolist()
    if isinstance(value, dict) and 'data' in value:
        return value['data']
    return value or []


def selected_heads_from_table(table_rows):
    selected = []
    for row in _table_rows(table_rows):
        if row is None or len(row) < 3:
            continue
        if bool(row[0]) and str(row[1]).strip() and str(row[2]).strip():
            selected.append(f'{row[1]}:{row[2]}')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate head identifiers in the attention table')
    return selected


def table_rows_from_labels(labels, selected_heads=None, *, select_all_if_empty=False):
    labels = list(labels or [])
    selected = set(selected_heads or [])
    all_checked = select_all_if_empty and not selected
    return [[bool(all_checked or label in selected), *split_head_label(label)] for label in labels]


def table_rows_from_saved(selected_heads):
    return table_rows_from_labels(selected_heads or [], selected_heads or [])


def _derived_layers(selected_heads):
    return list(dict.fromkeys(label.rsplit(':H', 1)[0] for label in selected_heads))


def get_attention_catalog(model, model_id):
    # Catalog needs only the model architecture. Device and dtype cannot change it.
    backend = cached_backend(model, 'auto', 'auto', model_id or None)
    return backend.attention_catalog()


def _head_summary(catalog, selected):
    total = sum(int(catalog['head_counts'][l]) for l in catalog['layers'])
    return f'**{len(selected)} / {total} heads** · **{len(_derived_layers(selected))} / {len(catalog["layers"])} layers**'


def auto_load_heads(model, model_id, saved_rows=None, *, restore=False):
    """Build the full table from architecture metadata. Never load generator weights.

    First page load retains saved head choices; changing backends starts with all
    heads selected, avoiding accidental reuse of another model's head labels.
    """
    try:
        catalog = get_attention_catalog(model, model_id)
        labels = labels_from_catalog(catalog, list(catalog['layers']))
        previous = selected_heads_from_table(saved_rows) if restore else []
        valid = [head for head in previous if head in set(labels)]
        if restore and previous and not valid:
            selected = labels
            note = ' · Saved heads were unavailable for this model; selected all.'
        elif restore and previous:
            selected = valid
            note = ''
        else:
            selected = labels
            note = ''
        rows = table_rows_from_labels(labels, selected)
        return catalog, gr.update(value=rows), _head_summary(catalog, selected) + note
    except Exception as error:
        # Keep saved visible rows if the metadata source is temporarily offline.
        return None, gr.update() if restore else gr.update(value=[]), f'**Could not read attention metadata:** {error}'


def switched_catalog(model, model_id):
    """Backend switch: reset affinity and layer choices from the new model's metadata."""
    catalog, table, info = auto_load_heads(model, model_id, restore=False)
    if catalog:
        affinity = list(catalog['default_affinity_layers'])
        return (catalog, table, gr.update(choices=catalog['layers'], value=affinity),
                gr.update(choices=catalog['layers'], value=list(catalog['layers'])), info)
    return (catalog, table, gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]), info)


def table_summary(table_rows, catalog):
    selected = selected_heads_from_table(table_rows)
    if catalog:
        return _head_summary(catalog, selected)
    return f'**{len(selected)} heads selected** · Metadata not loaded.'


def set_all_rows_selected(table_rows, selected=True):
    return gr.update(value=[[bool(selected), str(row[1]), str(row[2])]
                            for row in _table_rows(table_rows)
                            if row is not None and len(row) >= 3])


def select_heads_in_layers(table_rows, layers):
    selected_layers = set(layers or [])
    return gr.update(value=[[str(row[1]) in selected_layers, str(row[1]), str(row[2])]
                            for row in _table_rows(table_rows)
                            if row is not None and len(row) >= 3])


def _checkpoint_selection(checkpoint, table_rows):
    from visual_prompt_guidance.heads import Learned
    policy = Learned.load(checkpoint)
    available = [f'{row[1]}:{row[2]}' for row in _table_rows(table_rows) if row and len(row) >= 3]
    if set(policy.labels or ()) != set(available):
        # Keep the table intact until user explicitly loads the checkpoint.
        return None
    return list(policy.selected or policy.labels or ())


def apply_head_mode(mode, table_rows, checkpoint, old_manual_rows):
    manual_visible = mode == 'Manual'
    adaptive_visible = mode == 'Adaptive'
    learned_visible = mode == 'Learned'
    if mode in {'Uniform', 'Adaptive', 'Learned'}:
        # A learned checkpoint fixes its own trained head selection.
        try:
            specific = _checkpoint_selection(checkpoint, table_rows) if learned_visible and checkpoint else None
        except (OSError, ValueError):
            specific = None
        available = [f'{row[1]}:{row[2]}' for row in _table_rows(table_rows) if row and len(row) >= 3]
        head_update = gr.update(value=table_rows_from_labels(available, specific or available))
    else:
        head_update = gr.update()
    manual_rows = manual_rows_from_selection(head_update.get('value', table_rows), old_manual_rows)
    return (gr.update(visible=manual_visible), gr.update(visible=adaptive_visible),
            gr.update(visible=learned_visible), head_update, gr.update(value=manual_rows))


def manual_rows_from_selection(head_table, old_rows):
    selected = selected_heads_from_table(head_table)
    old = {
        str(row[0]): float(row[1])
        for row in _table_rows(old_rows)
        if row is not None and len(row) >= 2 and str(row[0]).strip()
    }
    return [[head, old.get(head, 1.0)] for head in selected]

class Panel:
    """Backend/sampling/guidance/propagation editor shared by both workflows.

    The persisted schema remains flat for backward-compatible CLI usage; only
    the UI groups settings by the corresponding engine component.
    """

    def __init__(self, profile: str, training: bool):
        self.profile = profile
        self.training = training
        self.parts = {}
        self.catalog = None
        self._create()

    def add(self, key, component):
        self.parts[key] = component
        return component

    @property
    def keys(self):
        return list(self.parts)

    @property
    def components(self):
        return list(self.parts.values())

    def _updated_settings(self, *values):
        updates = {key: normalize_value(value) for key, value in zip(self.keys, values)}
        # All available heads appear in the UI, but only checked ones are active.
        # demo.load() populates remote metadata asynchronously. A user can click
        # Save/Train before the table appears; an *empty* table is therefore an
        # uninitialized selector, not the user's choice to deselect every head.
        rows = _table_rows(updates.pop('selected_heads'))
        chosen = selected_heads_from_table(rows)
        standard = not self.training and updates.get('inference_mode') == 'standard'
        if standard:
            # Unguided inference never needs a catalog/checkpoint. Keep the
            # previous head selection so returning to Guided is nondestructive.
            if chosen:
                updates['selected_heads'] = chosen
                updates['selected_layers'] = _derived_layers(chosen)
            else:
                previous = profile_settings(self.profile)
                updates['selected_heads'] = list(previous['selected_heads'])
                updates['selected_layers'] = list(previous['selected_layers'])
            return updates
        if not rows:
            try:
                catalog = get_attention_catalog(updates['model'], updates.get('model_id') or '')
                chosen = labels_from_catalog(catalog, list(catalog['layers']))
            except Exception as exc:
                raise ValueError(f'Could not initialize attention heads for {updates["model"]}: {exc}') from exc
        if not chosen:
            raise ValueError('Select at least one attention head in the table')
        updates['selected_heads'] = chosen
        updates['selected_layers'] = _derived_layers(chosen)
        if updates.get('head_mode') == 'Manual':
            weights = {str(row[0]): float(row[1]) for row in _table_rows(updates.get('manual_rows'))
                       if row is not None and len(row) >= 2 and str(row[0]).strip()}
            updates['manual_rows'] = [[head, weights.get(head, 1.0)] for head in chosen]
        if self.training:
            updates['head_mode'] = 'Learned'
        elif updates.get('head_mode') == 'Learned' and updates.get('learned_checkpoint'):
            from visual_prompt_guidance.heads import Learned
            policy = Learned.load(updates['learned_checkpoint'])
            expected = set(policy.selected or policy.labels or ())
            if set(chosen) != expected:
                raise ValueError('Selected heads differ from the learned checkpoint; click Use checkpoint to restore its selection')
            # A checkpoint preserves the exact captured layer set, which can
            # include layers with zero selected heads.
            updates['selected_layers'] = _derived_layers(policy.labels or ())
        return updates

    def settings(self, *values):
        resolved = profile_settings(self.profile)
        resolved.update(self._updated_settings(*values))
        return resolved

    def save(self, *values):
        return save_profile(self.profile, self._updated_settings(*values))

    def _create(self):
        init = profile_settings(self.profile)
        add = self.add

        # Initialize metadata *before rendering* any input component.  Gradio
        # validates every input against its choices during preprocessing, even
        # when the component lives in a collapsed/hidden accordion.  If choices
        # are populated only by demo.load(), clicking Train/Run before that
        # asynchronous callback finishes raises "value is not in choices: []".
        saved_heads = list(init['selected_heads'])
        if (not self.training and init.get('head_mode') == 'Learned'
                and init.get('learned_checkpoint') and not saved_heads):
            try:
                from visual_prompt_guidance.heads import Learned
                checkpoint = Learned.load(init['learned_checkpoint'])
                saved_heads = list(checkpoint.selected or checkpoint.labels or ())
            except (OSError, ValueError):
                pass
        starting_rows = table_rows_from_saved(saved_heads)
        # A custom *local* model is safe to inspect synchronously (config.json
        # only). For remote/default models, defer discovery to demo.load() so
        # app construction never waits on Hugging Face or downloads weights.
        local_model = init.get('model_id') and Path(init['model_id']).expanduser().is_dir()
        if local_model:
            initial_catalog, table_update, initial_info = auto_load_heads(
                init['model'], init['model_id'], starting_rows, restore=True)
            initial_rows = table_update.get('value', starting_rows)
        else:
            initial_catalog, initial_rows = None, starting_rows
            initial_info = 'Loading attention metadata automatically…'
        if initial_catalog:
            affinity_choices = list(initial_catalog['layers'])
            saved_affinity = [x for x in init['propagation_layers'] if x in affinity_choices]
            initial_affinity = saved_affinity or list(initial_catalog['default_affinity_layers'])
            initial_layer_choices = affinity_choices
        else:
            # Valid inputs must be available before demo.load() finishes.
            # Gradio pre-validates even hidden or collapsed controls.
            affinity_choices = list(dict.fromkeys(init['propagation_layers']))
            initial_affinity = list(init['propagation_layers'])
            initial_layer_choices = _derived_layers(selected_heads_from_table(initial_rows))
        initial_layer_value = _derived_layers(selected_heads_from_table(initial_rows))

        with gr.Accordion('Backend', open=True):
            with gr.Row():
                add('model', gr.Dropdown(['InstructPix2Pix', 'FLUX.2 Klein'], value=init['model'], label='Model'))
                add('model_id', gr.Textbox(value=init['model_id'] or '', label='Model ID / local path', placeholder='Default checkpoint'))
            with gr.Row():
                device_options = list(dict.fromkeys([*device_choices(), init['device']]))
                add('device', gr.Dropdown(device_options, value=init['device'], label='Device'))
                add('dtype', gr.Dropdown(['auto', 'fp32', 'fp16', 'bf16'], value=init['dtype'], label='Dtype'))
            flux = gr.Group(visible=init['model'] == 'FLUX.2 Klein')
            with flux:
                with gr.Row():
                    add('flux_mode', gr.Dropdown(['base', 'inpaint'], value=init['flux_mode'], label='FLUX mode'))
                    add('flux_inpaint_strength', gr.Slider(.1, 1, value=init['flux_inpaint_strength'], step=.05, label='Inpaint strength'))
                    add('flux_cfg', gr.Number(value=init['flux_cfg'], label='CFG'))
            pix = gr.Group(visible=init['model'] == 'InstructPix2Pix')
            with pix:
                with gr.Row():
                    add('pix_text_cfg', gr.Number(value=init['pix_text_cfg'], label='Text CFG'))
                    add('pix_image_cfg', gr.Number(value=init['pix_image_cfg'], label='Image CFG'))
            self.parts['model'].change(
                lambda model: (gr.update(visible=model == 'FLUX.2 Klein'),
                               gr.update(visible=model == 'InstructPix2Pix')),
                self.parts['model'], [flux, pix], queue=False)

        with gr.Accordion('Sampling', open=True):
            with gr.Row():
                add('steps', gr.Slider(1, 60, value=init['steps'], step=1, label='Denoising steps'))
                add('max_side', gr.Slider(64, 1024, value=init['max_side'], step=32, label='Max side (px)'))
                add('seed', gr.Number(value=init['seed'], precision=0, label='Seed'))

        with gr.Accordion('Guidance', open=True,
                          visible=self.training or init['inference_mode'] == 'guided') as self.guidance_panel:
            with gr.Row():
                add('guide_steps', gr.Slider(0, 60, value=init['guide_steps'], step=1, label='Guided steps'))
                add('strength', gr.Number(value=init['strength'], label='Guidance strength'))
                add('normalize_gradient', gr.Checkbox(value=init['normalize_gradient'], label='Normalize gradient'))
            with gr.Accordion('Losses', open=False):
                add('selected_losses', gr.CheckboxGroup(list(losses.BUILTIN_LOSSES), value=init['selected_losses'], label='Guidance loss terms'))
                with gr.Row():
                    for k in ('coverage_w', 'focal_w', 'moment_w'):
                        add(k, gr.Number(value=init[k], label=k.removesuffix('_w')))
                with gr.Row():
                    for k in ('centerline_w', 'centroid_w', 'outside_w'):
                        add(k, gr.Number(value=init[k], label=k.removesuffix('_w')))
            with gr.Accordion('Attention policy', open=True):
                if not self.training:
                    add('head_mode', gr.Radio(['Uniform', 'Adaptive', 'Manual', 'Learned'], value=init['head_mode'], label='Head policy'))
                else:
                    gr.Markdown('**Learned policy** · Trainable logits for the selected heads.')
                    add('learned_temperature', gr.Number(value=init['learned_temperature'], label='Softmax temperature'))

                self.catalog = gr.State(initial_catalog)
                self.head_info = gr.Markdown(initial_info)
                add('selected_heads', gr.Dataframe(
                    headers=['Use', 'Layer', 'Head'],
                    datatype=['bool', 'str', 'str'],
                    value=initial_rows,
                    interactive=True, row_count=max(1, len(initial_rows)),
                    column_count=3,
                    static_columns=[1, 2], show_search='filter',
                    label='Attention heads', max_height=320, wrap=False,
                    column_widths=['80px', '125px', '100px'],
                ))
                with gr.Row():
                    head_layers = gr.Dropdown(
                        choices=initial_layer_choices, value=initial_layer_value,
                        multiselect=True, label='Target layers')
                    select_layers = gr.Button('Select all heads in layers', size='sm')
                self.head_layers = head_layers
                select_layers.click(select_heads_in_layers,
                                    [self.parts['selected_heads'], head_layers],
                                    [self.parts['selected_heads']], queue=False)
                with gr.Row():
                    select_all = gr.Button('Select all heads', size='sm')
                    clear_all = gr.Button('Clear selection', size='sm')
                select_all.click(lambda rows: set_all_rows_selected(rows, True),
                                 self.parts['selected_heads'], self.parts['selected_heads'], queue=False)
                clear_all.click(lambda rows: set_all_rows_selected(rows, False),
                                self.parts['selected_heads'], self.parts['selected_heads'], queue=False)
                self.parts['selected_heads'].change(table_summary,
                    [self.parts['selected_heads'], self.catalog], [self.head_info], queue=False)

                if not self.training:
                    adaptive = gr.Group(visible=init['head_mode'] == 'Adaptive')
                    with adaptive:
                        with gr.Row():
                            for k in ('adaptive_temperature', 'adaptive_ema', 'adaptive_mix'):
                                add(k, gr.Number(value=init[k], label=k.replace('adaptive_', '')))
                    manual = gr.Group(visible=init['head_mode'] == 'Manual')
                    with manual:
                        add('manual_rows', gr.Dataframe(
                            headers=['Head', 'Weight'], datatype=['str', 'number'],
                            row_count=max(1, len(init['manual_rows'])),
                            column_count=2,
                            value=init['manual_rows'], interactive=True,
                            label='Manual weights', max_height=220))
                    learned = gr.Group(visible=init['head_mode'] == 'Learned')
                    with learned:
                        options = list(dict.fromkeys([*([init['learned_checkpoint']] if init['learned_checkpoint'] else []), *checkpoint_choices()]))
                        add('learned_checkpoint', gr.Dropdown(
                            choices=options, value=init['learned_checkpoint'] or None,
                            allow_custom_value=True, label='Trained weights (learned_heads.json)'))
                        with gr.Row():
                            self.refresh_checkpoints = gr.Button('Refresh runs', size='sm')
                            self.load_checkpoint = gr.Button('Use checkpoint', size='sm')
                    self.parts['head_mode'].change(
                        apply_head_mode,
                        [self.parts['head_mode'], self.parts['selected_heads'],
                         self.parts['learned_checkpoint'], self.parts['manual_rows']],
                        [manual, adaptive, learned, self.parts['selected_heads'], self.parts['manual_rows']],
                        queue=False)
                    self.parts['selected_heads'].change(manual_rows_from_selection,
                        [self.parts['selected_heads'], self.parts['manual_rows']],
                        [self.parts['manual_rows']], queue=False)

            with gr.Accordion('Advanced guidance', open=False):
                if not self.training:
                    add('scoring_losses', gr.CheckboxGroup(
                        list(losses.BUILTIN_LOSSES), value=init['scoring_losses'],
                        label='Separate adaptive scoring losses (empty = guidance losses)'))
                add('updater', gr.State(init['updater']))  # Only built-in updater; injectable engines remain supported.

        with gr.Accordion('Propagation', open=False,
                          visible=self.training or init['inference_mode'] == 'guided') as self.propagation_panel:
            enabled = add('use_propagation', gr.Checkbox(
                value=init['use_propagation'], label='Enable self-attention propagation'))
            propagation_fields = gr.Group(visible=init['use_propagation'])
            with propagation_fields:
                add('propagation_layers', gr.Dropdown(
                    choices=affinity_choices, value=initial_affinity,
                    multiselect=True, allow_custom_value=True,
                    label='Self-attention affinity layers'))
                with gr.Row():
                    for k in ('propagation_start', 'propagation_stop', 'propagation_limit'):
                        add(k, gr.Number(value=init[k], precision=0, label=k.replace('propagation_', '')))
                with gr.Row():
                    for k in ('propagation_threshold', 'propagation_max_fraction'):
                        add(k, gr.Number(value=init[k], label=k.replace('propagation_', '')))
                # The current engine has one built-in transform. Custom transforms
                # remain injectable through Engine.__init__ directly.
                add('propagation_algorithm', gr.State(init['propagation_algorithm']))
            enabled.change(lambda flag: gr.update(visible=flag), enabled, propagation_fields, queue=False)

        # Load the complete attention table at page open and every backend switch.
        # Keeping this work on client-session load avoids network/model requests
        # during app construction and avoids loading multi-GB model weights.
        def initial_catalog(model, model_id, current):
            saved = current
            # Upgrade older profiles that stored a checkpoint but cleared
            # selected_heads. The checkpoint provides the correct saved subset.
            if (not self.training and init.get('head_mode') == 'Learned'
                    and init.get('learned_checkpoint')
                    and not selected_heads_from_table(current)):
                try:
                    from visual_prompt_guidance.heads import Learned
                    ckpt = Learned.load(init['learned_checkpoint'])
                    saved = table_rows_from_saved(ckpt.selected or ckpt.labels)
                except (OSError, ValueError):
                    pass
            catalog, table, info = auto_load_heads(model, model_id, saved, restore=True)
            if catalog:
                previous_affinity = init.get('propagation_layers') or []
                valid_affinity = [x for x in previous_affinity if x in catalog['layers']]
                affinity = valid_affinity or list(catalog['default_affinity_layers'])
                layer_value = _derived_layers(selected_heads_from_table(table['value']))
                return (catalog, table, gr.update(choices=catalog['layers'], value=affinity),
                        gr.update(choices=catalog['layers'], value=layer_value), info)
            return catalog, table, gr.update(), gr.update(), info

        def switched_model(model):
            # Previous backend's custom checkpoint must not leak to a new model.
            cat, rows, affinity, layers, info = switched_catalog(model, '')
            return gr.update(value=''), cat, rows, affinity, layers, info

        self._initial_catalog = initial_catalog
        self._switched_catalog = switched_catalog
        self._switched_model = switched_model
        self.parts['model'].input(
            self._switched_model, [self.parts['model']],
            [self.parts['model_id'], self.catalog, self.parts['selected_heads'],
             self.parts['propagation_layers'], self.head_layers, self.head_info], queue=True)
        self.parts['model_id'].blur(
            self._switched_catalog,
            [self.parts['model'], self.parts['model_id']],
            [self.catalog, self.parts['selected_heads'], self.parts['propagation_layers'],
             self.head_layers, self.head_info],
            queue=True)

    def connect_autoload(self, demo):
        demo.load(self._initial_catalog,
            [self.parts['model'], self.parts['model_id'], self.parts['selected_heads']],
            [self.catalog, self.parts['selected_heads'], self.parts['propagation_layers'],
             self.head_layers, self.head_info],
            show_progress='minimal')

    def connect_save(self, button, status):
        def save(*values):
            try:
                self.save(*values)
                return f'Saved {self.profile} settings to `{config_path()}`'
            except ValueError as e:
                raise gr.Error(str(e)) from e
        button.click(save, self.components, [status], queue=False)
