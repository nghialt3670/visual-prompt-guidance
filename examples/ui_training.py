"""Training-only Gradio UI with per-sample live loss/gradient/weight plots.

Launch: uv run --extra all python examples/ui_training.py --port 7862
"""
from __future__ import annotations

from dataclasses import asdict
from math import log
from pathlib import Path
from queue import Queue
from threading import Thread
import traceback

import gradio as gr
import pandas as pd

from ui_common import Panel, backend_for, serve
from visual_prompt_guidance import HeadTrainer
from visual_prompt_guidance.config import profile_settings, config_path, save_profile
from visual_prompt_guidance.runtime import build_engine, model_options, train_config
from visual_prompt_guidance.experiments import create_run_dir, save_run_config, save_training_reports


def train_stream(dataset_file, *values, panel: Panel):
    """A Gradio generator streams events while the synchronous trainer runs in a worker.

    HeadTrainer remains UI-agnostic; it invokes a plain progress callback.
    No model or tensors are sent through Queue, only scalar diagnostic values.
    """
    settings=panel.settings(*values)
    location=str(dataset_file or settings['dataset_path'] or '').strip()
    if not location:
        raise gr.Error('Provide a dataset ZIP, folder, or manifest.json')
    settings['dataset_path']=location
    settings['head_mode']='Learned'
    if settings['guide_steps'] < 1:
        raise gr.Error('Training requires at least one guided step')
    panel.save(*values)
    # The uploaded ZIP path is transient; do not persist it as the permanent dataset path.
    if not dataset_file:
        save_profile('train',{'dataset_path':location})
    folder=create_run_dir('train')
    path=folder/'learned_heads.json'
    history=folder/'history.json'
    train_cfg,image_cfg=train_config(settings,path,history)
    options=model_options(settings)
    save_run_config(folder,{'workflow':'train','source':'ui_training',
        'config_source':str(config_path()),'settings':settings,
        'model_options':options,'train_config':asdict(train_cfg),
        'image_loss':asdict(image_cfg),'dataset':location})
    events: Queue=Queue(maxsize=32)

    def worker():
        try:
            trainer=HeadTrainer(build_engine(settings,training=True,
                                             backend=backend_for(settings)),
                                config=train_cfg,image_loss=image_cfg)
            def on_progress(count,total,event):
                # The trainer callback is invoked after optimizer.step().
                policy=trainer.engine.policy
                import torch
                with torch.no_grad():
                    weights=policy.weights(policy.labels,device=policy.logits.device).detach().float()
                    positive=weights[weights>0]
                    peak=float(positive.max().cpu())
                    entropy=float((-positive*positive.log()).sum().cpu())
                events.put(('progress',{'step':count,'total':total,
                                        'loss':float(event['loss']),
                                        'grad_norm':float(event['grad_norm']),
                                        'max_head_weight':peak,
                                        'head_entropy':entropy,
                                        'epoch':event['epoch']}))
            result=trainer.fit(location,options=options,progress=on_progress)
            save_training_reports(folder,result.history,result.head_labels,result.weights)
            save_profile('train',{'learned_checkpoint':result.checkpoint})
            events.put(('done',result))
        except BaseException as error:
            events.put(('error',f'{type(error).__name__}: {error}\n{traceback.format_exc(limit=4)}'))

    thread=Thread(target=worker,name='vpg-training',daemon=True)
    thread.start()
    records=[]
    empty=pd.DataFrame(columns=['step','loss','grad_norm','max_head_weight'])
    yield empty,empty,empty,f'Run started: `{folder}`. Loading model / dataset…',''
    while True:
        kind,payload=events.get()
        if kind=='progress':
            records.append(payload)
            df=pd.DataFrame.from_records(records)
            status=f"Iteration {payload['step']}/{payload['total']} · epoch {payload['epoch']} · image loss {payload['loss']:.6f} · max head weight {payload['max_head_weight']:.4f}"
            yield df,df,df,status,''
        elif kind=='done':
            result=payload
            status=(f'**Completed:** {len(result.history)} optimizer updates. '
                    f'Final loss: {result.history[-1]["loss"]:.6f}. '
                    f'Run folder: `{folder}`.')
            df=pd.DataFrame.from_records(records) if records else empty
            yield df,df,df,status,result.checkpoint
            break
        else:
            df=pd.DataFrame.from_records(records) if records else empty
            yield df,df,df,f'**Training failed:** {payload}\nRun folder: `{folder}`',''
            break


def build_demo():
    init=profile_settings('train')
    with gr.Blocks(title='Visual Prompt Guidance · Training') as demo:
        gr.Markdown('## Visual Prompt Guidance · Training\n'
                    'Train head weights against ground-truth edits. The pretrained generator is frozen. '
                    'Plots update **after each training sample**.')
        with gr.Row():
            save_btn=gr.Button('Save training settings')
            load_btn=gr.Button('Restore training settings')
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
                train_components['image_loss_name']=gr.Dropdown(['l1','mse','charbonnier','smooth_l1'],
                    value=init['image_loss_name'],label='Image loss')
                train_components['inside_weight']=gr.Number(value=init['inside_weight'],label='Edit-region weight')
                train_components['outside_weight']=gr.Number(value=init['outside_weight'],label='Background weight')
                train_components['lpips_weight']=gr.Number(value=init['lpips_weight'],label='LPIPS weight')
                train_components['grad_clip']=gr.Number(value=init['grad_clip'],label='Gradient clipping')
        panel=Panel('train',training=True)
        panel.add('dataset_path',dataset_path)
        for key,comp in train_components.items():
            panel.add(key,comp)
        panel.connect_presets(save_button=save_btn,load_button=load_btn,status=status)
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
        train_button.click(lambda file,*vals: train_stream(file,*vals,panel=panel),
             [dataset_file,*panel.components],
             [loss_plot,gradient_plot,weight_plot,progress_status,checkpoint],
             concurrency_limit=1,show_progress='minimal')
    return demo


if __name__=='__main__':
    serve(build_demo,default_port=7862)
