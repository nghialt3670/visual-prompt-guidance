"""Training-only Gradio UI with per-sample live loss/gradient/weight plots.

Launch: uv run --extra all python examples/ui_training.py --port 7862
"""
from __future__ import annotations

from dataclasses import asdict
from math import log
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
import traceback
import time

import gradio as gr
import pandas as pd

from ui_common import Panel, backend_for, serve
from visual_prompt_guidance import HeadTrainer
from visual_prompt_guidance.config import profile_settings, config_path, save_profile
from visual_prompt_guidance.runtime import build_engine, model_options, train_config, resolve_training_precision
from visual_prompt_guidance.experiments import create_run_dir, save_run_config, save_training_reports


class _StepStatusObserver:
    """Progress-only observer: never copies maps or tensors to the UI."""
    wants_maps = False
    wants_scores = False

    def __init__(self, notify, *, steps: int):
        self.notify = notify
        self.steps = steps

    def step(self, event):
        self.notify(f"Denoising step {event.index + 1}/{self.steps} completed")

    def end(self, result):
        self.notify("Denoising and image decoding completed")


def train_stream(dataset_file, *values, panel: Panel):
    """Stream stage updates, denoising progress and optimizer metrics.

    A long model load or second-order backward must never leave the UI silent.
    The worker owns compute; this generator owns only scalar UI updates.
    """
    started = time.monotonic()
    untouched = (gr.update(), gr.update(), gr.update())
    print('[VPG training] Train requested; validating configuration', flush=True)
    # Yield before synchronous setup so Gradio can acknowledge the button click.
    yield (*untouched, 'Preparing training configuration…', '')

    try:
        settings = panel.settings(*values)
        requested_dtype = settings["dtype"]
        settings = resolve_training_precision(settings)
        location = str(dataset_file or settings['dataset_path'] or '').strip()
        if not location:
            raise ValueError('Provide a dataset ZIP, folder, or manifest.json')
        settings['dataset_path'] = location
        settings['head_mode'] = 'Learned'
        if settings['guide_steps'] < 1:
            raise ValueError('Training requires at least one guided step')
        panel.save(*values)
        if not dataset_file:
            save_profile('train', {'dataset_path': location})
        folder = create_run_dir('train')
        path = folder / 'learned_heads.json'
        history = folder / 'history.json'
        train_cfg, image_cfg = train_config(settings, path, history)
        options = model_options(settings)
        save_run_config(folder, {
            'workflow': 'train', 'source': 'ui_training',
            'config_source': str(config_path()), 'settings': settings,
            'model_options': options, 'train_config': asdict(train_cfg),
            'image_loss': asdict(image_cfg), 'dataset': location,
        })
    except Exception as exc:
        print(f'[VPG training] Configuration failed: {exc}', flush=True)
        yield (*untouched, f'**Training setup failed:** {exc}', '')
        return

    events = Queue()  # Unbounded scalar-only queue: never block the GPU on UI rendering.
    records = []
    log_file = folder / 'training.log'

    def notify(message):
        print(f'[VPG training] {message}', flush=True)
        with log_file.open('a', encoding='utf-8') as handle:
            handle.write(f'{message}\n')
        events.put(('status', message))

    def worker():
        try:
            notify(f'Run folder: {folder}')
            if requested_dtype != settings["dtype"]:
                notify(f'Training dtype auto resolved to {settings["dtype"]} on MPS (higher-order gradients)')
            if train_cfg.detect_anomaly:
                notify("Autograd anomaly detection enabled (slow): tracing the first invalid backward op")
            notify('Creating backend and engine (model weights load on first sample)')
            engine = build_engine(settings, training=True, backend=backend_for(settings))
            engine.observers = (*engine.observers, _StepStatusObserver(notify, steps=int(settings['steps'])))
            trainer = HeadTrainer(engine, config=train_cfg, image_loss=image_cfg)

            def on_progress(count, total, event):
                policy = trainer.engine.policy
                import torch
                with torch.no_grad():
                    weights = policy.weights(policy.labels, device=policy.logits.device).detach().float()
                    positive = weights[weights > 0]
                    peak = float(positive.max().cpu())
                    entropy = float((-positive * positive.log()).sum().cpu())
                payload = {
                    'step': count, 'total': total,
                    'loss': float(event['loss']),
                    'grad_norm': float(event['grad_norm']),
                    'max_head_weight': peak,
                    'head_entropy': entropy,
                    'epoch': event['epoch'],
                }
                gpu_note = (f" CUDA peak={event['cuda_peak_allocated_gib']:.2f} GiB"
                            if 'cuda_peak_allocated_gib' in event else '')
                print(f"[VPG training] Update {count}/{total}: "
                      f"loss={payload['loss']:.6f} grad={payload['grad_norm']:.5g} "
                      f"max_weight={peak:.4f}{gpu_note}", flush=True)
                with log_file.open('a', encoding='utf-8') as handle:
                    handle.write(f"Update {count}/{total}: loss={payload['loss']:.6f} "
                                 f"grad={payload['grad_norm']:.5g} max_weight={peak:.4f}\n")
                events.put(('progress', payload))

            result = trainer.fit(location, options=options, progress=on_progress, stage=notify)
            notify('Preparing training plots and reports')
            save_training_reports(folder, result.history, result.head_labels, result.weights)
            save_profile('train', {'learned_checkpoint': result.checkpoint})
            events.put(('done', result))
        except BaseException as exc:
            message = f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}'
            print(f'[VPG training] FAILED: {message}', flush=True)
            with log_file.open('a', encoding='utf-8') as handle:
                handle.write(f'FAILED: {message}\n')
            events.put(('error', message))

    thread = Thread(target=worker, name='vpg-training', daemon=True)
    thread.start()
    last_status = f'Run started: `{folder}` · initializing'
    yield (*untouched, last_status, '')

    # There may be no optimizer update for minutes during model loading/backward.
    # Timed reads make progress visible even when training is computationally busy.
    while True:
        try:
            kind, payload = events.get(timeout=5)
        except Empty:
            elapsed = int(time.monotonic() - started)
            if not thread.is_alive():
                message = 'Worker stopped without reporting a result; check training.log'
                print(f'[VPG training] {message}', flush=True)
                yield (*untouched, f'**Training stopped:** {message} · `{folder}`', '')
                break
            print(f'[VPG training] Still running: {last_status} ({elapsed}s elapsed)', flush=True)
            yield (*untouched, f'{last_status} · {elapsed}s elapsed (working)', '')
            continue

        if kind == 'status':
            last_status = str(payload)
            yield (*untouched, f'{last_status} · `{folder}`', '')
        elif kind == 'progress':
            records.append(payload)
            df = pd.DataFrame.from_records(records)
            last_status = (f"Update {payload['step']}/{payload['total']} · epoch {payload['epoch']} · "
                           f"image loss {payload['loss']:.6f} · max head weight {payload['max_head_weight']:.4f}")
            yield df, df, df, last_status, ''
        elif kind == 'done':
            result = payload
            df = pd.DataFrame.from_records(records)
            last_loss = result.history[-1]['loss'] if result.history else float('nan')
            notify_done = (f'Completed: {len(result.history)} optimizer updates; '
                           f'final loss {last_loss:.6f}; checkpoint {result.checkpoint}')
            print(f'[VPG training] {notify_done}', flush=True)
            yield df, df, df, f'**{notify_done}** · `{folder}`', result.checkpoint
            break
        elif kind == 'error':
            yield (*untouched, f'**Training failed:**\n```text\n{payload}\n```\nSee `{log_file}`', '')
            break


def build_demo():
    init=profile_settings('train')
    with gr.Blocks(title='Visual Prompt Guidance · Training') as demo:
        gr.Markdown('## Visual Prompt Guidance · Training\n'
                    'Train head weights against ground-truth edits. The pretrained generator is frozen. '
                    'Plots update **after each training sample**.')
        save_btn=gr.Button('Save training settings', size='sm')
        status=gr.Markdown(f'Config: `{config_path()}` · Train profile')
        with gr.Row():
            dataset_file=gr.File(label='Dataset ZIP (optional)',file_types=['.zip'],type='filepath')
            dataset_path=gr.Textbox(label='Dataset folder or manifest.json on server',
                                    value=init['dataset_path'])
        with gr.Accordion('Training optimizer and image-level loss',open=True):
            # Training options are part of the same shared settings schema.
            train_components={}
            with gr.Row():
                train_components['training_epochs']=gr.Number(value=init['training_epochs'],precision=0,label='Epochs')
                train_components['learning_rate']=gr.Number(value=init['learning_rate'],label='Learning rate')
                train_components['optimizer_name']=gr.Dropdown(['adamw','adam','sgd'],value=init['optimizer_name'],label='Optimizer')
                train_components['weight_decay']=gr.Number(value=init['weight_decay'],label='Weight decay')
            with gr.Row():
                train_components['edit_loss_name']=gr.Dropdown(['l1','mse','charbonnier','smooth_l1','lpips'],
                    value=init['edit_loss_name'],label='Edit loss')
                train_components['edit_loss_weight']=gr.Number(value=init['edit_loss_weight'],label='Edit weight')
                train_components['background_loss_name']=gr.Dropdown(['l1','mse','charbonnier','smooth_l1','lpips'],
                    value=init['background_loss_name'],label='Background loss')
                train_components['background_loss_weight']=gr.Number(value=init['background_loss_weight'],label='Background weight')
                train_components['grad_clip']=gr.Number(value=init['grad_clip'],label='Gradient clipping')
            with gr.Accordion('FLUX memory optimization (exact gradients)', open=False):
                with gr.Row():
                    train_components['train_checkpoint_transformer'] = gr.Checkbox(
                        value=init['train_checkpoint_transformer'], label='Checkpoint transformer')
                    train_components['train_checkpoint_attention'] = gr.Checkbox(
                        value=init['train_checkpoint_attention'], label='Checkpoint attention chunks')
                    train_components['train_checkpoint_vae'] = gr.Checkbox(
                        value=init['train_checkpoint_vae'], label='Checkpoint VAE decoder')
                    train_components['train_query_chunk'] = gr.Number(
                        value=init['train_query_chunk'], precision=0, label='Query chunk size')
        panel=Panel('train',training=True)
        panel.add('dataset_path',dataset_path)
        for key,comp in train_components.items():
            panel.add(key,comp)
        panel.connect_save(save_btn,status)
        panel.connect_autoload(demo)
        train_button=gr.Button('Train head weights',variant='primary')
        progress_status=gr.Markdown('')
        checkpoint=gr.Textbox(label='Saved checkpoint (usable in Inference)',interactive=False)
        with gr.Row():
            loss_plot=gr.LinePlot(x='step',y='loss',title='Image-level training loss',
                                  x_title='Optimizer update',y_title='Loss',height=300)
            gradient_plot=gr.LinePlot(x='step',y='grad_norm',title='Head-logit gradient norm',
                                     x_title='Optimizer update',y_title='Norm',height=300)
        weight_plot=gr.LinePlot(x='step',y='max_head_weight',title='Largest individual head weight',
                                x_title='Optimizer update',y_title='Weight',height=250)
        # Gradio must receive a generator *function* for streaming. A lambda
        # returning train_stream(...) is not recognized as a generator callback;
        # Gradio instead treats the generator object as a single output.
        def run_training(file, *vals):
            yield from train_stream(file, *vals, panel=panel)

        # A no-input, non-queued acknowledgement updates the UI immediately,
        # even if another queued Gradio event (e.g. metadata discovery) is slow.
        def acknowledge_training():
            print('[VPG training] Train button clicked; submitting job', flush=True)
            return 'Training requested · starting worker…', ''

        accepted = train_button.click(
            acknowledge_training, inputs=[], outputs=[progress_status, checkpoint],
            queue=False, show_progress='hidden')
        accepted.then(
            run_training, [dataset_file, *panel.components],
            [loss_plot, gradient_plot, weight_plot, progress_status, checkpoint],
            concurrency_limit=1, show_progress='minimal')
    return demo


if __name__=='__main__':
    serve(build_demo,default_port=7862)
