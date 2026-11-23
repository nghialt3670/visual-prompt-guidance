import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import ui_visualize


def test_visualizer_loads_inference_images_stats_and_attention_map(tmp_path):
    run = tmp_path / 'generate' / 'run-a'
    (run / 'head_maps' / 'step_000').mkdir(parents=True)
    (run / 'head_maps' / 'step_001').mkdir()
    (run / 'config.json').write_text(json.dumps({
        'workflow': 'generate', 'settings': {'model': 'Test Model', 'prompt': 'edit'},
    }))
    (run / 'steps.json').write_text(json.dumps([
        {'step': 0, 'timestep': 1, 'grad_rms': 0.25, 'losses': {'total': 1.5}},
    ]))
    (run / 'heads.csv').write_text(
        'step,head,weight,mean_attention,max_attention\n'
        '0,D0:H00,,0.4,0.9\n0,D0:H01,,0.6,0.8\n0,D1:H00,,0.2,0.7\n')
    (run / 'head_labels.json').write_text(json.dumps(['D0:H00', 'D0:H01', 'D1:H00']))
    Image.new('RGB', (8, 8), 'white').save(run / 'input.png')
    Image.new('RGB', (8, 8), 'black').save(run / 'result.png')
    np.save(run / 'head_maps' / 'step_000' / 'head_000.npy',
            np.asarray([[[0.0, 0.5], [0.75, 1.0]]], dtype=np.float32))
    np.save(run / 'head_maps' / 'step_000' / 'head_001.npy',
            np.asarray([[[1.0, 0.5], [0.25, 0.0]]], dtype=np.float32))
    np.save(run / 'head_maps' / 'step_000' / 'head_002.npy',
            np.full((1, 2, 2), 8.0, dtype=np.float32))
    np.save(run / 'head_maps' / 'step_001' / 'head_000.npy',
            np.asarray([[[0.25, 0.5], [0.75, 1.0]]], dtype=np.float32))
    np.save(run / 'head_maps' / 'step_001' / 'head_001.npy',
            np.asarray([[[1.0, 0.75], [0.5, 0.25]]], dtype=np.float32))
    np.save(run / 'head_maps' / 'step_001' / 'head_002.npy',
            np.full((1, 2, 2), 4.0, dtype=np.float32))

    view = ui_visualize.load_run('generate', 'generate/run-a', root=tmp_path)
    assert view['source'].size == view['result'].size == (8, 8)
    steps, heads = ui_visualize._select_heads(run, ['All layers'])
    labels = [label for label, _ in heads]
    assert labels == ['D0:H00', 'D0:H01', 'D1:H00']
    assert steps == ['step_000', 'step_001']
    cells = [ui_visualize._row_cells(run, steps, stem) for _, stem in heads]
    assert sum(cell is not None for row in cells for cell in row) == 6
    assert isinstance(view['map_grid'], Image.Image)
    assert view['map_grid'].size[0] > 100 and view['map_grid'].size[1] > 40
    assert '3 heads · 2 saved steps' in view['map_status']
    assert {'Gradient RMS', 'Loss · total', 'Mean attention', 'Peak attention'} <= set(view['metrics']['series'])
    assert ui_visualize.run_choices('generate', tmp_path) == ['generate/run-a']
    assert ui_visualize.resolve_run('generate', '../outside', tmp_path) is None
    d0_steps, d0_heads = ui_visualize._select_heads(run, ['D0'])
    d0_grid, d0_status, _, _ = ui_visualize.build_attention_grid(run, ['D0'])
    assert [label for label, _ in d0_heads] == ['D0:H00', 'D0:H01']
    assert d0_steps == ['step_000', 'step_001']
    assert '2 heads · 2 saved steps' in d0_status
    assert d0_grid.size[1] < view['map_grid'].size[1]
    rendered = ui_visualize.render_run('generate', 'generate/run-a', root=tmp_path)
    assert rendered[18]['value'] == ['All layers']


def test_visualizer_loads_training_reports_and_metrics(tmp_path):
    run = tmp_path / 'train' / 'run-b'
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({
        'workflow': 'train', 'settings': {'model': 'Test Model'},
    }))
    (run / 'sample_losses.csv').write_text(
        'update,epoch,sample,loss\n1,1,0,2.0\n2,1,1,1.0\n')
    (run / 'epoch_losses.csv').write_text('epoch,mean_loss,samples\n1,1.5,2\n')
    (run / 'head_weights.csv').write_text('head,weight\nD0:H00,0.7\nD0:H01,0.3\n')
    Image.new('RGB', (12, 6), 'white').save(run / 'loss_curve.png')
    (run / 'training.log').write_text('training complete\n')

    view = ui_visualize.load_run('train', 'train/run-b', root=tmp_path)
    assert len(view['samples']) == 2
    assert len(view['epochs']) == 1
    assert len(view['weights']) == 2
    assert view['loss_curve'].size == (12, 6)
    assert 'Epoch mean' in set(view['metrics']['series'])
    assert view['log'] == 'training complete\n'


def test_visualizer_gradio_app_builds_without_saved_runs(tmp_path):
    demo = ui_visualize.build_demo(root=tmp_path)
    labels = {component.get('props', {}).get('label')
              for component in demo.config['components']}
    assert {'Workflow', 'Saved run', 'Layers to show',
            'Attention maps by denoising step (columns) and layer:head (rows)', 'Per-sample losses'} <= labels
    groups = {component['props'].get('elem_id'): component['props'].get('visible')
            for component in demo.config['components']
              if component.get('props', {}).get('elem_id') in {
                  'vpg-inference-views', 'vpg-training-views'}}
    assert groups == {'vpg-inference-views': True, 'vpg-training-views': False}
    callbacks = {callback.fn.__name__: callback.fn for callback in demo.fns.values()}
    tab_visibility = callbacks['update_workflow_tabs']
    assert [item['visible'] for item in tab_visibility('generate')] == [True, False]
    assert [item['visible'] for item in tab_visibility('train')] == [False, True]
    outputs = ui_visualize.render_run('generate', None, root=tmp_path)
    assert len(outputs) == 23
    assert outputs[18]['value'] == ['All layers']


def test_update_attention_grid_avoids_infinite_loop_and_normalizes(tmp_path):
    run = tmp_path / 'generate' / 'run-c'
    (run / 'head_maps' / 'step_000').mkdir(parents=True)
    (run / 'head_labels.json').write_text(json.dumps(['D0:H00', 'D1:H00']))
    np.save(run / 'head_maps' / 'step_000' / 'head_000.npy', np.ones((1, 2, 2), dtype=np.float32))
    np.save(run / 'head_maps' / 'step_000' / 'head_001.npy', np.ones((1, 2, 2), dtype=np.float32))

    demo = ui_visualize.build_demo(root=tmp_path)
    callbacks = {callback.fn.__name__: callback.fn for callback in demo.fns.values()}
    update_fn = callbacks['update_attention_grid']

    # When layers is already normalized to ['All layers'], must NOT return a value update
    layer_up, img, status, page_up = update_fn('generate', 'generate/run-c', ['All layers'], 1)
    assert 'value' not in layer_up
    assert 'value' not in page_up
    assert isinstance(img, Image.Image)
    assert '2 heads' in status

    # When user selects another layer alongside 'All layers', 'All layers' is removed
    layer_up, img, status, _ = update_fn('generate', 'generate/run-c', ['All layers', 'D0'], 1)
    assert layer_up.get('value') == ['D0']
    assert '1 heads' in status

    # When layers is already normalized to ['D0'], must NOT return a value update
    layer_up, img, status, _ = update_fn('generate', 'generate/run-c', ['D0'], 1)
    assert 'value' not in layer_up

    # When user selects 'All layers' while 'D0' is active, it switches back to ['All layers']
    layer_up, img, status, _ = update_fn('generate', 'generate/run-c', ['D0', 'All layers'], 1)
    assert layer_up.get('value') == ['All layers']
    assert '2 heads' in status

    # Out-of-range pages are clamped to the last page
    _, _, status, page_up = update_fn('generate', 'generate/run-c', ['All layers'], 5)
    assert page_up.get('value') == 1

    # Clicking a cell shows its full map; clicking the gap outside the grid does nothing
    detail_fn = callbacks['on_map_plot_select']
    class DummySelectData:
        index = (100, 40)
    detail = detail_fn('generate', 'generate/run-c', ['All layers'], 1, 'per_head', 'by_page', DummySelectData())
    assert isinstance(detail, Image.Image)
    class OutsideSelectData:
        index = (5, 5)
    assert detail_fn('generate', 'generate/run-c', ['All layers'], 1, 'per_head', 'by_page', OutsideSelectData()) == {
        '__type__': 'update'}


def test_attention_grid_marks_missing_cells_and_scales_rows(tmp_path):
    run = tmp_path / 'generate' / 'run-d'
    for step in ('step_000', 'step_001'):
        (run / 'head_maps' / step).mkdir(parents=True)
    (run / 'head_labels.json').write_text(json.dumps(['D0:H00', 'D0:H01']))
    np.save(run / 'head_maps' / 'step_000' / 'head_000.npy', np.zeros((1, 2, 2), np.float32))
    np.save(run / 'head_maps' / 'step_001' / 'head_000.npy', np.ones((1, 2, 2), np.float32))
    np.save(run / 'head_maps' / 'step_000' / 'head_001.npy', np.full((1, 2, 2), 5.0, np.float32))

    steps, heads = ui_visualize._select_heads(run, ['All layers'])
    cells = [ui_visualize._row_cells(run, steps, stem) for _, stem in heads]
    assert cells[1][1] is None
    assert ui_visualize._row_range(cells[0]) == (0.0, 1.0)
    # Grid origin is (92, 36) with 48 px tiles on a 51 px pitch
    assert ui_visualize._cell_position(2, 2, 144, 37) == (0, 1)
    assert ui_visualize._cell_position(2, 2, 141, 37) is None


def test_attention_grid_paginates_rows(tmp_path):
    run = tmp_path / 'generate' / 'run-e'
    (run / 'head_maps' / 'step_000').mkdir(parents=True)
    (run / 'head_labels.json').write_text(json.dumps([f'L0:H{i:02d}' for i in range(41)]))
    for i in range(41):
        np.save(run / 'head_maps' / 'step_000' / f'head_{i:03d}.npy', np.full((1, 2, 2), float(i), np.float32))

    _, heads = ui_visualize._select_heads(run, ['All layers'])
    assert len(heads) == 41
    _, status, page, pages = ui_visualize.build_attention_grid(run, ['All layers'], 9)
    assert (page, pages) == (2, 2)
    assert '41 heads' in status and 'page 2/2' in status
    assert len(ui_visualize._page_heads(heads, 2)) == 1


def test_attention_average_mode_averages_heads_per_layer(tmp_path):
    run = tmp_path / 'generate' / 'run-f'
    for step in ('step_000', 'step_001'):
        (run / 'head_maps' / step).mkdir(parents=True)
    (run / 'head_labels.json').write_text(json.dumps(['D0:H00', 'D0:H01', 'D1:H00']))
    np.save(run / 'head_maps' / 'step_000' / 'head_000.npy', np.zeros((1, 2, 2), np.float32))
    np.save(run / 'head_maps' / 'step_000' / 'head_001.npy', np.full((1, 2, 2), 2.0, np.float32))
    np.save(run / 'head_maps' / 'step_000' / 'head_002.npy', np.full((1, 2, 2), 4.0, np.float32))
    np.save(run / 'head_maps' / 'step_001' / 'head_000.npy', np.full((1, 2, 2), 3.0, np.float32))

    steps, units = ui_visualize._grid_units(run, ['All layers'], ui_visualize.MAP_AVERAGE)
    assert [label for label, _ in units] == ['D0', 'D1']
    d0_stems = dict(units)['D0']
    cells = ui_visualize._unit_cells(run, steps, ui_visualize.MAP_AVERAGE, d0_stems)
    assert np.allclose(cells[0], 1.0)
    # Head 001 has no map at step_001, so the layer average uses head 000 alone
    assert np.allclose(cells[1], 3.0)
    d1_cells = ui_visualize._unit_cells(run, steps, ui_visualize.MAP_AVERAGE, dict(units)['D1'])
    assert np.allclose(d1_cells[0], 4.0) and d1_cells[1] is None

    _, status, page, pages = ui_visualize.build_attention_grid(
        run, ['All layers'], 1, ui_visualize.MAP_AVERAGE)
    assert (page, pages) == (1, 1)
    assert '2 layers · 3 heads averaged' in status

    detail = ui_visualize._map_detail(run, ['All layers'], 1, 144, 37, ui_visualize.MAP_AVERAGE)
    assert isinstance(detail, Image.Image)


def test_attention_average_mode_paginates_layers(tmp_path):
    run = tmp_path / 'generate' / 'run-g'
    (run / 'head_maps' / 'step_000').mkdir(parents=True)
    (run / 'head_labels.json').write_text(json.dumps([f'L{i}:H00' for i in range(41)]))
    for i in range(41):
        np.save(run / 'head_maps' / 'step_000' / f'head_{i:03d}.npy', np.full((1, 2, 2), float(i), np.float32))

    _, status, page, pages = ui_visualize.build_attention_grid(
        run, ['All layers'], 9, ui_visualize.MAP_AVERAGE)
    assert (page, pages) == (2, 2)
    assert '41 layers · 41 heads averaged' in status and 'page 2/2' in status


def test_attention_full_view_renders_all_rows_and_reaches_later_rows(tmp_path):
    run = tmp_path / 'generate' / 'run-h'
    (run / 'head_maps' / 'step_000').mkdir(parents=True)
    (run / 'head_labels.json').write_text(json.dumps([f'L0:H{i:02d}' for i in range(41)]))
    for i in range(41):
        np.save(run / 'head_maps' / 'step_000' / f'head_{i:03d}.npy', np.full((1, 2, 2), float(i), np.float32))

    paged, _, _, _ = ui_visualize.build_attention_grid(run, ['All layers'], 2)
    full, status, page, pages = ui_visualize.build_attention_grid(
        run, ['All layers'], 2, ui_visualize.MAP_PER_HEAD, ui_visualize.VIEW_FULL)
    assert full.size[1] > paged.size[1]
    assert 'all 41 rows (scrolling)' in status and 'page' not in status
    assert (page, pages) == (2, 2)

    # Row 40 (the 41st row) sits below the first page and is only clickable in full view
    assert ui_visualize._cell_position(41, 1, 93, 2077) == (40, 0)
    assert ui_visualize._map_detail(
        run, ['All layers'], 1, 93, 2077, ui_visualize.MAP_PER_HEAD, ui_visualize.VIEW_FULL) is not None
    assert ui_visualize._map_detail(
        run, ['All layers'], 1, 93, 2077, ui_visualize.MAP_PER_HEAD, ui_visualize.VIEW_BY_PAGE) is None


def test_attention_view_switch_toggles_page_input_and_scroll_height(tmp_path):
    demo = ui_visualize.build_demo(root=tmp_path)
    callbacks = {callback.fn.__name__: callback.fn for callback in demo.fns.values()}
    on_view_change = callbacks['on_view_change']

    _, _, page_up, scroll_up = on_view_change('generate', None, ['All layers'], 'per_head', 'full')
    assert page_up.get('visible') is False
    assert scroll_up.get('height') == ui_visualize.GRID_SCROLL_HEIGHT

    _, _, page_up, scroll_up = on_view_change('generate', None, ['All layers'], 'per_head', 'by_page')
    assert page_up.get('visible') is True
    assert scroll_up.get('height') is None

