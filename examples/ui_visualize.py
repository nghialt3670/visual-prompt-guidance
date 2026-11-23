"""Browse saved inference and training runs.

Launch: uv run --extra all python examples/ui_visualize.py --port 7863
"""
from __future__ import annotations

import functools
import json
import os
from math import isfinite
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


OUTPUTS_ROOT = Path(os.environ.get('VPG_OUTPUTS', 'outputs'))
WORKFLOWS = {'generate', 'train'}
METRIC_COLUMNS = ['step', 'series', 'value']
ALL_LAYERS = 'All layers'


def run_choices(workflow, root=OUTPUTS_ROOT):
    if workflow not in WORKFLOWS:
        return []
    base = Path(root) / workflow
    if not base.is_dir():
        return []
    return [path.relative_to(root).as_posix()
            for path in sorted(base.iterdir(), key=lambda item: item.name, reverse=True)
            if path.is_dir()]


def resolve_run(workflow, selected, root=OUTPUTS_ROOT):
    if workflow not in WORKFLOWS or not selected:
        return None
    parts = Path(str(selected)).parts
    if len(parts) != 2 or parts[0] != workflow or parts[1] in {'', '.', '..'}:
        return None
    root = Path(root).resolve()
    run = (root / parts[0] / parts[1]).resolve()
    return run if run.parent == (root / workflow).resolve() and run.is_dir() else None


def _read_json(path, fallback):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback


def _read_csv(path):
    try:
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_image(path):
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            return image.copy()
    except (OSError, ValueError):
        return None


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _metric_rows(steps, heads, samples, epochs):
    rows = []
    for event in steps:
        step = _number(event.get('step'))
        if step is None:
            continue
        grad = _number(event.get('grad_rms'))
        if grad is not None:
            rows.append({'step': step, 'series': 'Gradient RMS', 'value': grad})
        for name, value in (event.get('losses') or {}).items():
            number = _number(value)
            if number is not None:
                rows.append({'step': step, 'series': f'Loss · {name}', 'value': number})

    if not heads.empty and 'step' in heads:
        for column, title in (('mean_attention', 'Mean attention'),
                              ('max_attention', 'Peak attention')):
            if column not in heads:
                continue
            grouped = pd.to_numeric(heads[column], errors='coerce')
            for step, value in grouped.groupby(heads['step']).mean().items():
                number = _number(value)
                if number is not None:
                    rows.append({'step': float(step), 'series': title, 'value': number})

    if not samples.empty and {'update', 'loss'}.issubset(samples.columns):
        for _, record in samples.iterrows():
            step = _number(record.get('update'))
            loss = _number(record.get('loss'))
            if step is None or loss is None:
                continue
            sample = int(_number(record.get('sample')) or 0) + 1
            rows.append({'step': step, 'series': f'Image {sample}', 'value': loss})
        if not epochs.empty and {'epoch', 'mean_loss'}.issubset(epochs.columns):
            last_updates = samples.groupby('epoch')['update'].max().to_dict()
            for _, record in epochs.iterrows():
                update = last_updates.get(record.get('epoch'), _number(record.get('epoch')))
                mean = _number(record.get('mean_loss'))
                if update is not None and mean is not None:
                    rows.append({'step': float(update), 'series': 'Epoch mean', 'value': mean})
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _map_steps(run):
    folder = run / 'head_maps'
    if not folder.is_dir():
        return []
    return sorted((path.name for path in folder.glob('step_*') if path.is_dir()),
                  key=lambda value: int(value.removeprefix('step_')))


GRID_BG = (15, 23, 42)
GRID_TEXT = (226, 232, 240)
GRID_MISSING = (51, 65, 85)
GRID_ROWS_PER_PAGE = 40
GRID_SPACING = 3
GRID_LABEL_WIDTH = 92
GRID_HEADER_HEIGHT = 30
MAP_PER_HEAD = 'per_head'
MAP_AVERAGE = 'average'
VIEW_BY_PAGE = 'by_page'
VIEW_FULL = 'full'
GRID_SCROLL_HEIGHT = 600


def _head_labels(run):
    labels = _read_json(run / 'head_labels.json', [])
    return [str(label) for label in labels] if isinstance(labels, list) else []


def _layer_names(run):
    return list(dict.fromkeys(label.rsplit(':H', 1)[0] for label in _head_labels(run)))


def _map_path(run, step, stem):
    return run / 'head_maps' / step / f'{stem}.npy'


@functools.lru_cache(maxsize=512)
def _load_map(path, mtime_ns):  # mtime_ns is part of the cache key so rewritten maps reload
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2 or not values.size:
        return None
    return values.astype(np.float32)


def _read_cell(run, step, stem):
    path = _map_path(run, step, stem)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    return _load_map(str(path), mtime)


def _row_cells(run, steps, stem):
    return [_read_cell(run, step, stem) for step in steps]


def _grid_units(run, selected_layers=None, mode=MAP_PER_HEAD):
    steps, heads = _select_heads(run, selected_layers)
    if mode != MAP_AVERAGE:
        return steps, [(label, [stem]) for label, stem in heads]
    groups = {}
    for label, stem in heads:
        groups.setdefault(label.rsplit(':H', 1)[0], []).append(stem)
    return steps, list(groups.items())


def _unit_cells(run, steps, mode, stems):
    if mode != MAP_AVERAGE:
        return _row_cells(run, steps, stems[0])
    cells = []
    for step in steps:
        maps = [cell for cell in (_read_cell(run, step, stem) for stem in stems) if cell is not None]
        if maps:
            shape = maps[0].shape
            cells.append(np.mean([cell for cell in maps if cell.shape == shape], axis=0))
        else:
            cells.append(None)
    return cells


def _select_heads(run, selected_layers=None):
    """Return saved steps and (label, file stem) pairs for the selected layers."""
    selection = list(selected_layers or [ALL_LAYERS])
    active = set(_layer_names(run) if ALL_LAYERS in selection or not selection else selection)
    steps = _map_steps(run)
    heads = []
    for index, label in enumerate(_head_labels(run)):
        if label.rsplit(':H', 1)[0] not in active:
            continue
        stem = f'head_{index:03d}'
        if any(_map_path(run, step, stem).is_file() for step in steps):
            heads.append((label, stem))
    return steps, heads


def _page_count(heads):
    return max(1, -(-len(heads) // GRID_ROWS_PER_PAGE))


def _clamp_page(page, heads):
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    return min(max(page, 1), _page_count(heads))


def _page_heads(heads, page):
    start = (page - 1) * GRID_ROWS_PER_PAGE
    return heads[start:start + GRID_ROWS_PER_PAGE]


def _visible_units(units, page, map_view=VIEW_BY_PAGE):
    if map_view == VIEW_FULL:
        return list(units)
    return _page_heads(units, _clamp_page(page, units))


def _step_headers(run, steps):
    """Return (step index, timestep) text for each saved step column."""
    events = _read_json(run / 'steps.json', [])
    timesteps = {}
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and 'step' in event:
                timesteps[str(event['step'])] = str(event.get('timestep', ''))[:7]
    headers = []
    for name in steps:
        index = int(name.removeprefix('step_'))
        headers.append((str(index), timesteps.get(str(index), '')))
    return headers


def _row_range(cells):
    finite = [cell[np.isfinite(cell)] for cell in cells if cell is not None]
    finite = [values for values in finite if values.size]
    if not finite:
        return None
    values = np.concatenate(finite)
    return float(values.min()), float(values.max())


def _colorize(values, lo, hi):
    from matplotlib import colormaps

    span = max(hi - lo, 1e-9)
    normalized = np.clip(np.nan_to_num((values - lo) / span, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    rgb = (colormaps['inferno'](normalized)[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(values)] = GRID_MISSING
    return rgb


def _grid_metrics(rows, cols):
    thumb = max(24, min(48, 2400 // max(cols, 1)))
    step = thumb + GRID_SPACING
    top = GRID_HEADER_HEIGHT + 6
    return {
        'thumb': thumb,
        'step': step,
        'left': GRID_LABEL_WIDTH,
        'top': top,
        'width': GRID_LABEL_WIDTH + cols * step + 4,
        'height': top + rows * step + 4,
    }


def _cell_position(rows, cols, x, y):
    if rows <= 0 or cols <= 0:
        return None
    layout = _grid_metrics(rows, cols)
    step, thumb = layout['step'], layout['thumb']
    dx, dy = x - layout['left'], y - layout['top']
    if dx < 0 or dy < 0 or dx % step >= thumb or dy % step >= thumb:
        return None
    row, column = dy // step, dx // step
    if row >= rows or column >= cols:
        return None
    return int(row), int(column)


def _message_image(text):
    image = Image.new('RGB', (760, 180), GRID_BG)
    ImageDraw.Draw(image).text((28, 78), text, fill=GRID_TEXT, font=ImageFont.load_default())
    return image


def _render_attention_grid(labels, headers, cells):
    layout = _grid_metrics(len(labels), len(headers))
    thumb, step = layout['thumb'], layout['step']
    image = Image.new('RGB', (layout['width'], layout['height']), GRID_BG)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for column, (index, timestep) in enumerate(headers):
        x0 = layout['left'] + column * step
        draw.text((x0 + 2, 4), index, fill=GRID_TEXT, font=font)
        draw.text((x0 + 2, 16), timestep, fill=GRID_TEXT, font=font)
    for row, (label, row_cells) in enumerate(zip(labels, cells)):
        y0 = layout['top'] + row * step
        draw.text((8, y0 + thumb // 2 - 5), label, fill=GRID_TEXT, font=font)
        bounds = _row_range(row_cells)
        for column, cell in enumerate(row_cells):
            x0 = layout['left'] + column * step
            if cell is None or bounds is None:
                tile = Image.new('RGB', (thumb, thumb), GRID_MISSING)
            else:
                tile = Image.fromarray(_colorize(cell, *bounds)).resize(
                    (thumb, thumb), Image.Resampling.BILINEAR)
            image.paste(tile, (x0, y0))
    return image


def _map_detail(run, selected_layers, page, x, y, mode=MAP_PER_HEAD, map_view=VIEW_BY_PAGE):
    steps, units = _grid_units(run, selected_layers, mode)
    if not steps or not units:
        return None
    page_units = _visible_units(units, page, map_view)
    position = _cell_position(len(page_units), len(steps), x, y)
    if position is None:
        return None
    row, column = position
    label, stems = page_units[row]
    cells = _unit_cells(run, steps, mode, stems)
    values, bounds = cells[column], _row_range(cells)
    if values is None or bounds is None:
        return None
    height, width = values.shape
    scale = max(1, 512 // max(height, width))
    art = Image.fromarray(_colorize(values, *bounds)).resize(
        (width * scale, height * scale), Image.Resampling.NEAREST)
    index, timestep = _step_headers(run, steps)[column]
    name = f'{label} · avg of {len(stems)} heads' if mode == MAP_AVERAGE else label
    image = Image.new('RGB', (art.width, art.height + 22), GRID_BG)
    ImageDraw.Draw(image).text((6, 5), f'{name} · step {index} · t={timestep}',
                               fill=GRID_TEXT, font=ImageFont.load_default())
    image.paste(art, (0, 22))
    return image


def build_attention_grid(run, selected_layers=None, page=1, mode=MAP_PER_HEAD, map_view=VIEW_BY_PAGE):
    steps, units = _grid_units(run, selected_layers, mode)
    if not steps or not units:
        text = 'No saved attention maps for the selected layers.'
        return _message_image(text), text, 1, 1
    pages = _page_count(units)
    page = _clamp_page(page, units)
    page_units = _visible_units(units, page, map_view)
    cells = [_unit_cells(run, steps, mode, stems) for _, stems in page_units]
    if not any(cell is not None for row in cells for cell in row):
        text = 'Saved attention maps contain no readable values.'
        return _message_image(text), text, page, pages
    image = _render_attention_grid([label for label, _ in page_units], _step_headers(run, steps), cells)
    head_count = sum(len(stems) for _, stems in units)
    location = (f'all {len(page_units)} rows (scrolling)' if map_view == VIEW_FULL
                else f'page {page}/{pages}')
    if mode == MAP_AVERAGE:
        status = (f'{len(units)} layers · {head_count} heads averaged · {len(steps)} saved steps · '
                  f'{location} · heatmap per layer and step, each row scaled to its own range')
    else:
        status = (f'{head_count} heads · {len(steps)} saved steps · {location} · '
                  'heatmap per head and step, each row scaled to its own range')
    return image, status, page, pages


def load_run(workflow, selected, *, root=OUTPUTS_ROOT, mode=MAP_PER_HEAD, map_view=VIEW_BY_PAGE):
    run = resolve_run(workflow, selected, root)
    if run is None:
        return {
            'selected': None, 'summary': 'No saved runs found for this workflow.', 'config': {},
            'source': None, 'result': None, 'visual_prompt': None, 'metrics': pd.DataFrame(columns=METRIC_COLUMNS),
            'steps': pd.DataFrame(), 'heads': pd.DataFrame(), 'loss_curve': None,
            'samples': pd.DataFrame(), 'epochs': pd.DataFrame(), 'weights': pd.DataFrame(),
            'log': '', 'map_layers': [], 'map_grid': None,
            'map_status': 'No attention maps saved.',
        }

    config = _read_json(run / 'config.json', {})
    settings = config.get('settings') or {}
    steps = _read_json(run / 'steps.json', [])
    steps = steps if isinstance(steps, list) else []
    step_rows = []
    for event in steps:
        row = {key: event.get(key) for key in ('step', 'timestep', 'grad_rms')}
        row.update(event.get('losses') or {})
        row.update({f'extra.{key}': value for key, value in (event.get('extra') or {}).items()})
        step_rows.append(row)
    step_table = pd.DataFrame.from_records(step_rows)

    heads = _read_csv(run / 'heads.csv')
    samples = _read_csv(run / 'sample_losses.csv')
    epochs = _read_csv(run / 'epoch_losses.csv')
    history = _read_json(run / 'history.json', [])
    if samples.empty and isinstance(history, list):
        samples = pd.DataFrame([
            {'update': index, 'epoch': item.get('epoch', 1),
             'sample': item.get('sample', index - 1), 'loss': item.get('loss')}
            for index, item in enumerate(history, start=1) if isinstance(item, dict)
        ])
    if epochs.empty and not samples.empty and {'epoch', 'loss'}.issubset(samples.columns):
        epochs = (samples.groupby('epoch', as_index=False).agg(
            mean_loss=('loss', 'mean'), samples=('loss', 'count')))
    weights = _read_csv(run / 'head_weights.csv')
    if weights.empty and workflow == 'train':
        weights = _read_csv(run / 'heads.csv')
    if weights.empty:
        weights = pd.DataFrame(columns=['head', 'weight'])

    map_layers = _layer_names(run)
    map_grid, map_status, _, _ = build_attention_grid(run, [ALL_LAYERS], 1, mode, map_view)
    model = settings.get('model', '')
    summary = f'{workflow.title()} · {run.name}'
    if model:
        summary += f' · {model}'
    if steps:
        summary += f' · {len(steps)} denoising steps'
    elif not samples.empty:
        summary += f' · {len(samples)} training samples'
    return {
        'selected': selected, 'summary': summary, 'config': config,
        'source': _read_image(run / 'input.png'),
        'result': _read_image(run / 'result.png'),
        'visual_prompt': _read_image(run / 'visual_prompt.png'),
        'metrics': _metric_rows(steps, heads, samples, epochs),
        'steps': step_table, 'heads': heads,
        'loss_curve': _read_image(run / 'loss_curve.png'),
        'samples': samples, 'epochs': epochs, 'weights': weights,
        'log': (run / 'training.log').read_text(encoding='utf-8', errors='replace')
        if (run / 'training.log').is_file() else '',
        'map_layers': map_layers, 'map_grid': map_grid,
        'map_status': map_status,
    }


def render_run(workflow, selected, *, root=OUTPUTS_ROOT, mode=MAP_PER_HEAD, map_view=VIEW_BY_PAGE):
    choices = run_choices(workflow, root)
    if selected not in choices:
        selected = choices[0] if choices else None
    view = load_run(workflow, selected, root=root, mode=mode, map_view=map_view)
    return (
        gr.update(choices=choices, value=selected), view['summary'], view['config'],
        view['source'], view['result'], view['visual_prompt'],
        view['metrics'], view['steps'], view['heads'],
        view['metrics'], view['steps'], view['heads'],
        view['loss_curve'], view['samples'],
        view['epochs'], view['weights'], view['weights'], view['log'],
        gr.update(choices=[ALL_LAYERS, *view['map_layers']], value=[ALL_LAYERS], interactive=True),
        view['map_grid'], view['map_status'], gr.update(value=1), None,
    )


def build_demo(root=OUTPUTS_ROOT):
    initial_workflow = 'generate'
    initial_runs = run_choices('generate', root)
    initial_run = resolve_run(initial_workflow, initial_runs[0], root) if initial_runs else None
    initial_layers = _layer_names(initial_run) if initial_run else []
    initial_map_layer_choices = [ALL_LAYERS, *initial_layers]
    with gr.Blocks(title='Visual Prompt Guidance · Run Explorer') as demo:
        gr.Markdown('## Saved run explorer')
        with gr.Row():
            workflow = gr.Dropdown(
                choices=[('Inference', 'generate'), ('Training', 'train')],
                value=initial_workflow, label='Workflow', scale=1)
            run = gr.Dropdown(choices=initial_runs,
                              value=initial_runs[0] if initial_runs else None,
                              allow_custom_value=True, label='Saved run', scale=4)
            refresh = gr.Button('Refresh runs', size='sm')
        summary = gr.Markdown('')
        with gr.Accordion('Saved configuration', open=False):
            config_view = gr.JSON(label='Saved configuration')
        with gr.Group(visible=initial_workflow == 'generate', elem_id='vpg-inference-views') as inference_views:
            with gr.Tabs():
                with gr.Tab('Images'):
                    with gr.Row():
                        source = gr.Image(label='Input', type='pil', interactive=False)
                        result = gr.Image(label='Result', type='pil', interactive=False)
                        visual_prompt = gr.Image(label='Visual prompt', type='pil', interactive=False)
                with gr.Tab('Attention maps'):
                    with gr.Row():
                        map_layers = gr.Dropdown(
                            choices=initial_map_layer_choices, value=[ALL_LAYERS], multiselect=True,
                            interactive=True, allow_custom_value=True, label='Layers to show', scale=4)
                        map_mode = gr.Dropdown(
                            choices=[('Per head', MAP_PER_HEAD), ('Average per layer', MAP_AVERAGE)],
                            value=MAP_PER_HEAD, interactive=True, label='Display mode', scale=2)
                        map_view = gr.Dropdown(
                            choices=[('By page', VIEW_BY_PAGE), ('Full (scrolling)', VIEW_FULL)],
                            value=VIEW_BY_PAGE, interactive=True, label='View', scale=2)
                        map_page = gr.Number(value=1, precision=0, minimum=1, label='Page', interactive=True, scale=1)
                        all_layers_btn = gr.Button('Show all layers', size='sm', scale=1)
                    map_status = gr.Markdown('')
                    with gr.Row() as map_scroll:
                        map_plot = gr.Image(
                            label='Attention maps by denoising step (columns) and layer:head (rows)',
                            type='pil', interactive=False, buttons=['download'])
                    map_detail = gr.Image(
                        label='Selected map (click a cell above)', type='pil', interactive=False,
                        buttons=['download'])
                with gr.Tab('Statistics'):
                    inference_metrics = gr.LinePlot(
                        x='step', y='value', color='series', title='Saved run metrics',
                        x_title='Step / update', y_title='Value', height=360)
                    with gr.Row():
                        inference_steps = gr.Dataframe(label='Step statistics', interactive=False, wrap=True)
                        inference_heads = gr.Dataframe(label='Per-head statistics', interactive=False, wrap=True)

        with gr.Group(visible=initial_workflow == 'train', elem_id='vpg-training-views') as training_views:
            with gr.Tabs():
                with gr.Tab('Statistics'):
                    training_metrics = gr.LinePlot(
                        x='step', y='value', color='series', title='Training metrics',
                        x_title='Step / update', y_title='Value', height=360)
                    with gr.Row():
                        training_steps = gr.Dataframe(label='Step statistics', interactive=False, wrap=True)
                        training_heads = gr.Dataframe(label='Per-head statistics', interactive=False, wrap=True)
                with gr.Tab('Training reports'):
                    loss_curve = gr.Image(label='Training loss curve', type='pil', interactive=False)
                    with gr.Row():
                        samples = gr.Dataframe(label='Per-sample losses', interactive=False)
                        epochs = gr.Dataframe(label='Epoch losses', interactive=False)
                    with gr.Row():
                        weight_plot = gr.BarPlot(x='head', y='weight', title='Learned head weights', height=320)
                        weights = gr.Dataframe(label='Head weights', interactive=False)
                    log = gr.Textbox(label='Training log', lines=12, interactive=False)

        outputs = [run, summary, config_view, source, result, visual_prompt,
                   inference_metrics, inference_steps, inference_heads,
                   training_metrics, training_steps, training_heads,
                   loss_curve, samples, epochs, weight_plot, weights, log,
                   map_layers, map_plot, map_status, map_page, map_detail]

        def load_latest(workflow, mode, map_view):
            return render_run(workflow, None, root=root, mode=mode, map_view=map_view)

        def load_selected(workflow, selected, mode, map_view):
            return render_run(workflow, selected, root=root, mode=mode, map_view=map_view)

        def update_attention_grid(workflow, selected, layers, page, mode=MAP_PER_HEAD, map_view=VIEW_BY_PAGE):
            run_path = resolve_run(workflow, selected, root)
            if run_path is None:
                return gr.update(), _message_image('No attention maps saved.'), 'No attention maps saved.', gr.update()
            selected_layers = list(layers or [])
            choices = [ALL_LAYERS, *_layer_names(run_path)]
            if ALL_LAYERS in selected_layers and len(selected_layers) > 1:
                if selected_layers[-1] == ALL_LAYERS:
                    selected_layers = [ALL_LAYERS]
                else:
                    selected_layers = [layer for layer in selected_layers if layer != ALL_LAYERS]
            if not selected_layers:
                selected_layers = [ALL_LAYERS]
            requested = int(page or 1)
            figure, status, shown, _ = build_attention_grid(run_path, selected_layers, requested, mode, map_view)
            layer_update = (gr.update(choices=choices, interactive=True)
                            if selected_layers == list(layers or [])
                            else gr.update(choices=choices, value=selected_layers, interactive=True))
            page_update = gr.update() if shown == requested else gr.update(value=shown)
            return layer_update, figure, status, page_update

        def on_layers_change(workflow, selected, layers, mode=MAP_PER_HEAD, map_view=VIEW_BY_PAGE):
            layer_update, figure, status, _ = update_attention_grid(workflow, selected, layers, 1, mode, map_view)
            return layer_update, figure, status, gr.update(value=1)

        def on_view_change(workflow, selected, layers, mode, map_view):
            _, figure, status, _ = update_attention_grid(workflow, selected, layers, 1, mode, map_view)
            full = map_view == VIEW_FULL
            return (figure, status, gr.update(value=1, visible=not full),
                    gr.update(height=GRID_SCROLL_HEIGHT if full else None))

        def on_map_plot_select(workflow, selected, layers, page, mode, map_view, evt: gr.SelectData):
            run_path = resolve_run(workflow, selected, root)
            if run_path is None or not evt or not isinstance(evt.index, (tuple, list)):
                return gr.update()
            detail = _map_detail(run_path, list(layers or [ALL_LAYERS]), int(page or 1),
                                 int(evt.index[0]), int(evt.index[1]), mode, map_view)
            return gr.update() if detail is None else detail

        def update_workflow_tabs(selected):
            inference = selected == 'generate'
            return gr.update(visible=inference), gr.update(visible=not inference)

        workflow.change(load_latest, [workflow, map_mode, map_view], outputs, queue=False)
        workflow.change(update_workflow_tabs, workflow, [inference_views, training_views], queue=False)
        run.change(load_selected, [workflow, run, map_mode, map_view], outputs, queue=False)
        refresh.click(load_latest, [workflow, map_mode, map_view], outputs, queue=False)
        demo.load(load_selected, [workflow, run, map_mode, map_view], outputs, show_progress='minimal')
        map_layers.change(on_layers_change, [workflow, run, map_layers, map_mode, map_view],
                          [map_layers, map_plot, map_status, map_page], queue=False)
        map_mode.change(on_layers_change, [workflow, run, map_layers, map_mode, map_view],
                        [map_layers, map_plot, map_status, map_page], queue=False)
        map_view.change(on_view_change, [workflow, run, map_layers, map_mode, map_view],
                        [map_plot, map_status, map_page, map_scroll], queue=False)
        map_page.change(update_attention_grid, [workflow, run, map_layers, map_page, map_mode, map_view],
                        [map_layers, map_plot, map_status, map_page], queue=False)
        map_plot.select(on_map_plot_select, [workflow, run, map_layers, map_page, map_mode, map_view],
                        [map_detail], queue=False)
        all_layers_btn.click(lambda w, s, m, v: on_layers_change(w, s, [ALL_LAYERS], m, v),
                             [workflow, run, map_mode, map_view], [map_layers, map_plot, map_status, map_page], queue=False)
    return demo


if __name__ == '__main__':
    from ui_common import serve
    serve(build_demo, default_port=7863)