"""Optional Gradio application. NOT part of the framework's public/core API.

Run from the project root: uv run --extra all python examples/gradio_app.py
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from visual_prompt_guidance import (
    Engine, Request, Guidance, Point, Box, Scribble,
    Objective, Loss, Uniform, Manual, Adaptive, AffinityGrow, FileObserver, losses,
)
from visual_prompt_guidance.backends import Flux2Klein, InstructPix2Pix


@lru_cache(maxsize=2)
def backend(model: str):
    if model == "InstructPix2Pix":
        return InstructPix2Pix()
    if model == "FLUX.2 Klein":
        return Flux2Klein()
    raise ValueError(f"Unknown model: {model}")


def get_source(editor):
    if not editor or editor.get("background") is None:
        raise ValueError("Upload an image to the editor first")
    bg = editor["background"]
    return bg.convert("RGB") if isinstance(bg, Image.Image) else Image.fromarray(np.asarray(bg)).convert("RGB")


def on_editor_change(editor):
    try:
        return get_source(editor), []
    except ValueError:
        return None, []


def on_click(editor, points, evt: gr.SelectData):
    src = get_source(editor)
    x,y = map(float, evt.index[:2])
    points = list(points or []) + [(x,y)]
    img=src.copy()
    draw=ImageDraw.Draw(img)
    for i,(cx,cy) in enumerate(points):
        draw.ellipse((cx-5,cy-5,cx+5,cy+5),fill="red")
        draw.text((cx+8,cy+4),str(i+1),fill="red")
    return points,img


def extract_scribble(editor, size):
    mask = Image.new("L",size,0)
    for layer in editor.get("layers") or []:
        rgba = layer if isinstance(layer, Image.Image) else Image.fromarray(np.asarray(layer))
        if "A" in rgba.getbands():
            mask = Image.fromarray(np.maximum(np.asarray(mask),np.asarray(rgba.getchannel("A"))),"L")
    if not np.asarray(mask).any():
        raise ValueError("Draw a scribble with the brush")
    return mask


def run(editor, selected, kind, prompt, object_name, selected_losses,
        weights_json, model, mode, inpaint_strength, steps, guide_steps, strength, seed,
        text_cfg, image_cfg, max_side,
        head_mode, head_layers, manual_weights, temperature, propagate, save_maps):
    try:
        original = get_source(editor)
        points = selected or []
        if kind == "Point":
            if not points:
                raise ValueError("Click the preview image to place a point")
            visual=Point(*points[-1])
        elif kind == "Box":
            if len(points)<2:
                raise ValueError("Click two corners of the box on the preview")
            visual=Box(*points[-2],*points[-1],margin=16)
        else:
            visual=Scribble(mask=extract_scribble(editor,original.size))
        weights=json.loads(weights_json or "{}")
        terms=[Loss(name,losses.BUILTIN_LOSSES[name],float(weights.get(name,1.))) for name in selected_losses]
        if not terms:
            raise ValueError("Select at least one loss")
        objective=Objective(*terms)
        if head_mode == "Adaptive":
            policy=Adaptive(temperature=float(temperature))
        elif head_mode == "Manual":
            policy=Manual(json.loads(manual_weights))
        else:
            policy=Uniform()
        cfg=Guidance(strength=float(strength),max_steps=int(guide_steps),
                     layers=tuple(x.strip() for x in head_layers.split(",") if x.strip()) or None)
        folder=Path("outputs") / time.strftime("%Y%m%d_%H%M%S")
        observer=FileObserver(folder,maps=save_maps,images=save_maps,interval=3)
        transform=AffinityGrow(start=3,stop=int(guide_steps),limit=12,threshold=.2) if propagate else None
        model_backend = backend(model)
        runner=Engine(model_backend,objective,guidance=cfg,policy=policy,
                      transform=transform,observers=[observer])
        options={"steps":int(steps), "seed":int(seed), "cfg":float(text_cfg),
                 "max_side":int(max_side)}
        if model == "FLUX.2 Klein":
            options.update(mode=mode,inpaint_strength=float(inpaint_strength))
        else:
            options["image_cfg"] = float(image_cfg)
        req=Request(data={"image":original,"prompt":prompt,"object_phrase":object_name},
                    options=options)
        result=runner.run(req,visual)
        result.output.save(folder/"result.png")
        return result.output, f"Saved {folder} | {len(result.events)} steps"
    except (ValueError,RuntimeError,TypeError,KeyError,json.JSONDecodeError) as error:
        raise gr.Error(str(error)) from error


def build_demo():
    with gr.Blocks(title="Visual Prompt Guidance — Example Application") as demo:
        gr.Markdown("## Visual Prompt Guidance (example app)\nThe framework is independent of this UI. Draw for scribble; click the preview for point or two box corners.")
        with gr.Row():
            with gr.Column():
                editor=gr.ImageEditor(label="Source + scribble brush",type="pil",image_mode="RGBA",
                    brush=gr.Brush(default_size=10,colors=["#ff0000"],default_color="#ff0000",color_mode="fixed"),
                    layers=False,transforms=(),height=380)
                preview=gr.Image(label="Click to place point / box corners",type="pil",interactive=False)
                clicks=gr.State([])
                clear=gr.Button("Clear selected points")
                kind=gr.Dropdown(["Point","Box","Scribble"],value="Scribble",label="Visual prompt")
                prompt=gr.Textbox(label="Text prompt",value="Insert a curved snake following the line")
                object_name=gr.Textbox(label="Target object phrase",value="snake")
                losses_choice=gr.CheckboxGroup(list(losses.BUILTIN_LOSSES),value=["coverage"],label="Losses")
                weights=gr.Textbox(label="Loss weights (JSON; omit for 1.0)",value="{}")
                model=gr.Dropdown(["InstructPix2Pix","FLUX.2 Klein"],value="InstructPix2Pix",
                                  label="Model backend",info="The UI uses the selected backend; model weights download at first run.")
                with gr.Accordion("Sampling",open=True):
                    mode=gr.Dropdown(["base","inpaint"],value="base",label="Klein mode",
                                     info="FLUX.2 only. InstructPix2Pix is image editing without an inpaint mask.")
                    inpaint_strength=gr.Slider(.1,1.,value=.8,step=.05,label="Klein inpaint strength",
                                                info="FLUX.2 inpaint only.")
                    steps=gr.Slider(4,60,value=30,step=1,label="Denoising steps")
                    guided=gr.Slider(0,30,value=10,step=1,label="Guided steps",
                                     info="Number of early steps receiving attention-gradient guidance.")
                    strength=gr.Slider(0,.1,value=.01,step=.001,label="Latent guidance strength")
                    seed=gr.Number(value=42,precision=0,label="Seed")
                    text_cfg=gr.Slider(1,12,value=7.5,step=.5,label="Text guidance scale",
                                       info="InstructPix2Pix default 7.5; FLUX.2 often uses 4. Higher favors the text instruction.")
                    image_cfg=gr.Slider(1,3,value=1.5,step=.1,label="Image guidance scale (InstructPix2Pix)",
                                        info="Higher stays closer to the input image; used only by InstructPix2Pix.")
                    max_side=gr.Slider(128,768,value=256,step=64,label="Maximum image side (px)",
                                       info="Try 256px locally with gradients. 512px needs more memory.")
                with gr.Accordion("Head policy / diagnostics",open=False):
                    head_mode=gr.Dropdown(["Uniform","Adaptive","Manual"],value="Uniform",label="Head policy")
                    head_layers=gr.Textbox(value="",label="Attention layers (optional)",
                                           info="Leave blank to use the selected backend's default layers. Pix2Pix IDs are U-Net module paths.")
                    manual=gr.Textbox(value='{}',label="Manual head weights (JSON)",
                                      info="Use exact labels exported in heads.csv, e.g. {\"down_blocks.1.attentions.0.transformer_blocks.0.attn2:H00\":1}.")
                    temperature=gr.Number(value=.01,label="Adaptive temperature")
                    propagate=gr.Checkbox(value=False,label="Grow target using image affinity")
                    save_maps=gr.Checkbox(value=False,label="Export individual head maps / PNGs")
                submit=gr.Button("Generate",variant="primary")
            with gr.Column():
                output=gr.Image(label="Generated image",type="pil")
                status=gr.Textbox(label="Run information",interactive=False)
        editor.change(on_editor_change,[editor],[preview,clicks])
        preview.select(on_click,[editor,clicks],[clicks,preview])
        clear.click(lambda ed: ([],get_source(ed)),[editor],[clicks,preview])
        submit.click(run,[editor,clicks,kind,prompt,object_name,losses_choice,weights,model,mode,
                          inpaint_strength,steps,guided,strength,seed,text_cfg,image_cfg,max_side,
                          head_mode,head_layers,manual,temperature,propagate,save_maps],
                     [output,status],concurrency_limit=1)
    return demo


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=7861)
    args=parser.parse_args()
    build_demo().queue().launch(server_name=args.host,server_port=args.port)
