"""Optional Gradio application built on top of the framework API.

Run from the project root:
  uv run --extra all python examples/gradio_app.py
"""
from __future__ import annotations

import argparse
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from visual_prompt_guidance import (
    Engine, Request, Guidance, Point, Box, Scribble,
    Objective, Loss, Uniform, Manual, Adaptive, Learned, HeadSubset,
    HeadTrainer, TrainConfig, ImageLossConfig,
    AffinityGrow, FileObserver, losses,
)
from visual_prompt_guidance.backends import Flux2Klein, InstructPix2Pix

CANVAS_HEIGHT = 520
BUILTIN_WEIGHT_DEFAULTS = {
    "coverage": 1.0,
    "focal": 1.0,
    "moment": 1.0,
    "centerline": 1.0,
    "centroid": 1.0,
    "outside": 1.0,
}


@lru_cache(maxsize=4)
def backend(model: str):
    if model == "InstructPix2Pix":
        return InstructPix2Pix()
    if model == "FLUX.2 Klein":
        return Flux2Klein()
    raise ValueError(f"Unknown model: {model}")


def _to_pil_rgb(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.fromarray(np.asarray(value)).convert("RGB")


def _to_pil_rgba(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGBA")
    return Image.fromarray(np.asarray(value)).convert("RGBA")


def parse_editor(editor: dict | None) -> tuple[Image.Image, list[Image.Image]]:
    if not editor or editor.get("background") is None:
        raise ValueError("Upload an image first")
    background = _to_pil_rgb(editor["background"])
    layers = [_to_pil_rgba(layer) for layer in (editor.get("layers") or [])]
    return background, layers


def compose_canvas(background: Image.Image, layers: list[Image.Image], tool: str, points: list[tuple[float, float]]):
    composite = background.convert("RGBA")
    for layer in layers:
        composite = Image.alpha_composite(composite, layer.resize(composite.size))
    overlay = Image.new("RGBA", composite.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if tool == "Point" and points:
        x, y = points[-1]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(255, 40, 40, 255), width=3)
        draw.line((x - 12, y, x + 12, y), fill=(255, 40, 40, 255), width=2)
        draw.line((x, y - 12, x, y + 12), fill=(255, 40, 40, 255), width=2)
    elif tool == "Box" and points:
        for x, y in points[:1]:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 160, 0, 255))
        if len(points) >= 2:
            x0, y0 = points[-2]
            x1, y1 = points[-1]
            draw.rectangle((x0, y0, x1, y1), outline=(255, 160, 0, 255), width=4)
            draw.ellipse((x1 - 5, y1 - 5, x1 + 5, y1 + 5), fill=(255, 160, 0, 255))
    composite = Image.alpha_composite(composite, overlay)
    return {"background": background, "layers": layers, "composite": composite}


def render_editor(editor: dict | None, tool: str, points: list[tuple[float, float]]):
    background, layers = parse_editor(editor)
    return compose_canvas(background, layers, tool, points or [])


def status_for_tool(tool: str, points: list[tuple[float, float]]):
    if tool == "Point":
        return "Point mode: click once on the canvas." if not points else f"Point selected at ({points[-1][0]:.0f}, {points[-1][1]:.0f})."
    if tool == "Box":
        if len(points) == 0:
            return "Box mode: click top-left / first corner."
        if len(points) == 1:
            return "Box mode: click the second corner."
        x0, y0 = points[-2]
        x1, y1 = points[-1]
        return f"Box selected: ({x0:.0f}, {y0:.0f}) → ({x1:.0f}, {y1:.0f})."
    return "Scribble mode: draw directly on the same canvas with the brush."


def on_canvas_change(editor, tool, points):
    try:
        return render_editor(editor, tool, points or []), status_for_tool(tool, points or [])
    except ValueError:
        return None, "Upload an image first."


def on_canvas_select(editor, tool, points, evt: gr.SelectData):
    try:
        current = list(points or [])
        if tool == "Scribble":
            return current, render_editor(editor, tool, current), status_for_tool(tool, current)
        x, y = map(float, evt.index[:2])
        if tool == "Point":
            current = [(x, y)]
        elif tool == "Box":
            if len(current) >= 2:
                current = [(x, y)]
            else:
                current.append((x, y))
        return current, render_editor(editor, tool, current), status_for_tool(tool, current)
    except ValueError as error:
        raise gr.Error(str(error)) from error


def clear_box_point(editor, tool):
    return [], render_editor(editor, tool, []), status_for_tool(tool, [])


def clear_scribble(editor, tool, points):
    background, _ = parse_editor(editor)
    payload = compose_canvas(background, [], tool, points or [])
    return payload, status_for_tool(tool, points or [])


def clear_all(editor, tool):
    background, _ = parse_editor(editor)
    payload = compose_canvas(background, [], tool, [])
    return payload, [], status_for_tool(tool, [])


def extract_scribble_mask(editor, size):
    _, layers = parse_editor(editor)
    mask = Image.new("L", size, 0)
    for layer in layers:
        if "A" in layer.getbands():
            mask = Image.fromarray(np.maximum(np.asarray(mask), np.asarray(layer.getchannel("A"))), "L")
    if not np.asarray(mask).any():
        raise ValueError("Draw a scribble on the canvas.")
    return mask


def visual_prompt_from_canvas(editor, tool, points):
    background, _ = parse_editor(editor)
    if tool == "Point":
        if not points:
            raise ValueError("Select one point on the canvas.")
        return Point(*points[-1])
    if tool == "Box":
        if len(points) < 2:
            raise ValueError("Select two box corners on the canvas.")
        return Box(*points[-2], *points[-1], margin=16)
    return Scribble(mask=extract_scribble_mask(editor, background.size))


def collect_loss_terms(selected_losses, coverage_w, focal_w, moment_w, centerline_w, centroid_w, outside_w):
    weight_values = {
        "coverage": float(coverage_w),
        "focal": float(focal_w),
        "moment": float(moment_w),
        "centerline": float(centerline_w),
        "centroid": float(centroid_w),
        "outside": float(outside_w),
    }
    terms = [Loss(name, losses.BUILTIN_LOSSES[name], weight_values[name]) for name in (selected_losses or [])]
    if not terms:
        raise ValueError("Select at least one loss.")
    return Objective(*terms)


def update_model_panels(model):
    return (
        gr.update(visible=model == "FLUX.2 Klein"),
        gr.update(visible=model == "InstructPix2Pix"),
    )


def _head_count(module) -> int:
    heads = getattr(module, "heads", None)
    if heads is not None:
        return int(heads)
    inner = getattr(module, "inner_dim", None)
    head_dim = getattr(module, "head_dim", None)
    if inner is not None and head_dim:
        return int(inner // head_dim)
    raise ValueError("Could not infer attention head count")


def build_attention_catalog(model_name: str):
    model_backend = backend(model_name)
    if model_name == "InstructPix2Pix":
        from visual_prompt_guidance.backends.instruct_pix2pix_attention import cross_attention_layers, default_layers
        pipe = model_backend._load()
        layers_map = cross_attention_layers(pipe.unet)
        layers = tuple(layers_map.keys())
        defaults = tuple(default_layers(pipe.unet))
        affinity_defaults = defaults
        head_counts = {name: _head_count(module) for name, module in layers_map.items()}
    elif model_name == "FLUX.2 Klein":
        pipe = model_backend._load("base")
        transformer = pipe.transformer
        double = [f"D{i}" for i in range(len(transformer.transformer_blocks))]
        single = [f"S{i}" for i in range(len(transformer.single_transformer_blocks))]
        layers = tuple(double + single)
        defaults = tuple(double)
        affinity_defaults = tuple(double[:2])
        head_counts = {}
        for name in layers:
            if name.startswith("D"):
                module = transformer.transformer_blocks[int(name[1:])].attn
            else:
                module = transformer.single_transformer_blocks[int(name[1:])].attn
            head_counts[name] = _head_count(module)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return {
        "layers": list(layers),
        "default_layers": list(defaults),
        "default_affinity_layers": list(affinity_defaults),
        "head_counts": head_counts,
    }


def labels_from_catalog(catalog: dict, selected_layers: list[str] | None):
    layers = selected_layers or catalog.get("default_layers") or catalog.get("layers") or []
    labels = []
    for layer in layers:
        for idx in range(int(catalog["head_counts"].get(layer, 0))):
            labels.append(f"{layer}:H{idx:02d}")
    return labels


def refresh_attention_choices(model_name, selected_layers, selected_heads):
    try:
        catalog = build_attention_catalog(model_name)
        layer_values = list(selected_layers) if selected_layers else list(catalog["default_layers"])
        head_choices = labels_from_catalog(catalog, layer_values)
        keep_heads = [head for head in (selected_heads or []) if head in head_choices]
        info = f"Loaded {len(catalog['layers'])} layers and {len(head_choices)} heads for the selected layers."
        return (
            catalog,
            gr.update(choices=catalog["layers"], value=layer_values),
            gr.update(choices=catalog["layers"], value=list(catalog["default_affinity_layers"])),
            gr.update(choices=head_choices, value=keep_heads),
            [[head, 1.0] for head in keep_heads],
            info,
        )
    except Exception as error:  # noqa: BLE001 - surface backend/model issues in UI
        return (
            None,
            gr.update(),
            gr.update(),
            gr.update(),
            [],
            f"Failed to load attention choices: {error}",
        )


def update_heads_from_layers(catalog, selected_layers, selected_heads):
    if not catalog:
        return gr.update(), [], "Load attention choices first."
    head_choices = labels_from_catalog(catalog, list(selected_layers or []))
    kept = [head for head in (selected_heads or []) if head in head_choices]
    return gr.update(choices=head_choices, value=kept), [[head, 1.0] for head in kept], f"Showing {len(head_choices)} heads."


def update_manual_weight_table(selected_heads):
    return [[head, 1.0] for head in (selected_heads or [])]


def update_head_mode(mode):
    return (
        gr.update(visible=mode == "Manual"),
        gr.update(visible=mode == "Adaptive"),
        gr.update(visible=mode == "Learned"),
    )


def parse_manual_weights(rows):
    if rows is None:
        return {}
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    elif hasattr(rows, "tolist") and not isinstance(rows, list):
        rows = rows.tolist()
    if not rows:
        return {}
    parsed = {}
    for row in rows:
        if not row or len(row) < 2:
            continue
        label = str(row[0]).strip()
        if not label:
            continue
        weight = float(row[1])
        if weight < 0:
            raise ValueError(f"Weight for {label} must be nonnegative")
        parsed[label] = weight
    return parsed


def run(editor, selected_points, tool, prompt, object_name,
        selected_losses, coverage_w, focal_w, moment_w, centerline_w, centroid_w, outside_w,
        model, steps, guide_steps, strength, seed, max_side,
        flux_mode, flux_inpaint_strength, flux_cfg,
        pix_text_cfg, pix_image_cfg,
        head_mode, selected_layers, selected_heads, manual_rows, adaptive_temperature, adaptive_ema, adaptive_mix,
        learned_checkpoint,
        use_propagation, propagation_layers, propagation_start, propagation_stop, propagation_limit,
        propagation_threshold, propagation_max_fraction,
        save_maps):
    try:
        original, _ = parse_editor(editor)
        visual = visual_prompt_from_canvas(editor, tool, selected_points or [])
        objective = collect_loss_terms(selected_losses, coverage_w, focal_w, moment_w, centerline_w, centroid_w, outside_w)

        if head_mode == "Adaptive":
            policy = Adaptive(temperature=float(adaptive_temperature), ema=float(adaptive_ema), mix=float(adaptive_mix))
        elif head_mode == "Manual":
            policy = Manual(parse_manual_weights(manual_rows))
        elif head_mode == "Learned":
            if not learned_checkpoint or not Path(learned_checkpoint).is_file():
                raise ValueError("Train head weights first, or specify a valid learned-head checkpoint")
            policy = Learned.load(learned_checkpoint)
            expected_selection = set(policy.selected or policy.labels)
            if selected_heads and set(selected_heads) != expected_selection:
                raise ValueError("Head selection differs from the learned checkpoint; select matching heads or clear the checkbox selection")
        else:
            policy = Uniform()
        if selected_heads and head_mode in ("Adaptive", "Uniform"):
            policy = HeadSubset(policy, tuple(selected_heads))

        guidance = Guidance(
            strength=float(strength),
            max_steps=int(guide_steps),
            layers=tuple(selected_layers or []) or None,
            affinity_layers=tuple(propagation_layers or []) or None,
        )
        transform = None
        if use_propagation:
            transform = AffinityGrow(
                start=int(propagation_start),
                stop=int(propagation_stop),
                limit=int(propagation_limit),
                threshold=float(propagation_threshold),
                max_fraction=float(propagation_max_fraction),
            )

        folder = Path("outputs") / time.strftime("%Y%m%d_%H%M%S")
        observer = FileObserver(folder, maps=save_maps, images=save_maps, interval=3)
        model_backend = backend(model)
        runner = Engine(model_backend, objective, guidance=guidance, policy=policy, transform=transform, observers=[observer])

        options = {
            "steps": int(steps),
            "seed": int(seed),
            "max_side": int(max_side),
        }
        if model == "FLUX.2 Klein":
            options.update(mode=flux_mode, inpaint_strength=float(flux_inpaint_strength), cfg=float(flux_cfg))
        else:
            options.update(cfg=float(pix_text_cfg), image_cfg=float(pix_image_cfg), mode="base")

        request = Request(
            data={"image": original, "prompt": prompt, "object_phrase": object_name},
            options=options,
        )
        result = runner.run(request, visual)
        result.output.save(folder / "result.png")
        return result.output, f"Saved {folder} | {len(result.events)} steps"
    except (ValueError, RuntimeError, TypeError, KeyError) as error:
        raise gr.Error(str(error)) from error



def train_heads(dataset_file, dataset_path, model, selected_losses,
                coverage_w, focal_w, moment_w, centerline_w, centroid_w, outside_w,
                steps, guide_steps, strength, seed, max_side,
                flux_mode, flux_inpaint_strength, flux_cfg, pix_text_cfg, pix_image_cfg,
                selected_layers, selected_heads, learned_temperature,
                use_propagation, propagation_layers, propagation_start, propagation_stop,
                propagation_limit, propagation_threshold, propagation_max_fraction,
                epochs, learning_rate, optimizer_name, weight_decay, image_loss_name,
                inside_weight, outside_weight, lpips_weight, grad_clip, checkpoint_file,
                progress=gr.Progress()):
    try:
        location = str(dataset_file or dataset_path or "").strip()
        if not location:
            raise ValueError("Provide a dataset ZIP or a server-side manifest.json / directory")
        objective = collect_loss_terms(selected_losses, coverage_w, focal_w, moment_w,
                                       centerline_w, centroid_w, outside_w)
        policy = Learned(
            selected=tuple(selected_heads) if selected_heads else None,
            temperature=float(learned_temperature),
        )
        transform = AffinityGrow(
            start=int(propagation_start), stop=int(propagation_stop),
            limit=int(propagation_limit), threshold=float(propagation_threshold),
            max_fraction=float(propagation_max_fraction),
        ) if use_propagation else None
        guidance = Guidance(strength=float(strength), max_steps=int(guide_steps),
                            layers=tuple(selected_layers or []) or None,
                            affinity_layers=tuple(propagation_layers or []) or None)
        trainer = HeadTrainer(
            Engine(backend(model), objective, policy=policy,
                   guidance=guidance, transform=transform),
            config=TrainConfig(
                epochs=int(epochs), learning_rate=float(learning_rate),
                optimizer=str(optimizer_name), weight_decay=float(weight_decay), grad_clip=float(grad_clip),
                checkpoint=str(checkpoint_file),
                history=str(Path(checkpoint_file).with_suffix(".history.json")),
            ),
            image_loss=ImageLossConfig(
                pixel=str(image_loss_name), inside_weight=float(inside_weight),
                outside_weight=float(outside_weight), lpips_weight=float(lpips_weight),
            ),
        )
        options = {"steps": int(steps), "seed": int(seed), "max_side": int(max_side)}
        if model == "FLUX.2 Klein":
            options.update(mode=flux_mode, inpaint_strength=float(flux_inpaint_strength),
                           cfg=float(flux_cfg))
        else:
            options.update(cfg=float(pix_text_cfg), image_cfg=float(pix_image_cfg), mode="base")
        def on_progress(done, total, event):
            progress((done, total), desc=f"Epoch {event['epoch']} | image loss {event['loss']:.5f}")
        result = trainer.fit(location, options=options, progress=on_progress)
        weights = sorted(zip(result.head_labels, result.weights), key=lambda x: x[1], reverse=True)
        top = ", ".join(f"{name}: {weight:.3f}" for name, weight in weights[:3])
        return result.checkpoint, (
            f"Training completed: {len(result.history)} examples processed. "
            f"Last image loss: {result.history[-1]['loss']:.5f}. "
            f"Saved: {result.checkpoint}. Top heads: {top}"
        )
    except (ValueError, RuntimeError, TypeError, KeyError, ImportError, OSError) as error:
        raise gr.Error(f"Head-weight training failed: {error}") from error


def build_demo():
    with gr.Blocks(title="Visual Prompt Guidance — Example Application") as demo:
        gr.Markdown(
            "## Visual Prompt Guidance\n"
            "Use **one canvas** for all prompt types. Pick **Point**, **Box**, or **Scribble**; "
            "click on the same canvas for point/box, or draw directly for scribble."
        )
        canvas_points = gr.State([])
        attention_catalog = gr.State(None)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                canvas = gr.ImageEditor(
                    label="Input canvas",
                    type="pil",
                    image_mode="RGBA",
                    brush=gr.Brush(default_size=10, colors=["#ff0000"], default_color="#ff0000", color_mode="fixed"),
                    layers=False,
                    transforms=(),
                    height=CANVAS_HEIGHT,
                )
                canvas_status = gr.Markdown("Upload an image first.")
                with gr.Row():
                    tool = gr.Radio(["Scribble", "Point", "Box"], value="Scribble", label="Active prompt tool")
                    clear_shape = gr.Button("Clear point / box")
                    clear_brush = gr.Button("Clear scribble")
                    clear_all_btn = gr.Button("Clear all")
            with gr.Column(scale=1):
                output = gr.Image(label="Generated output", type="pil", height=CANVAS_HEIGHT)
                run_status = gr.Textbox(label="Run information", interactive=False)

        with gr.Row():
            prompt = gr.Textbox(label="Text prompt", value="Insert a curved snake following the line")
            object_name = gr.Textbox(label="Target object phrase", value="snake")
            model = gr.Dropdown(["InstructPix2Pix", "FLUX.2 Klein"], value="InstructPix2Pix", label="Backend")

        with gr.Accordion("Objective", open=True):
            selected_losses = gr.CheckboxGroup(
                list(losses.BUILTIN_LOSSES), value=["coverage"], label="Enabled losses"
            )
            with gr.Row():
                coverage_w = gr.Number(value=1.0, label="coverage")
                focal_w = gr.Number(value=1.0, label="focal")
                moment_w = gr.Number(value=1.0, label="moment")
            with gr.Row():
                centerline_w = gr.Number(value=1.0, label="centerline")
                centroid_w = gr.Number(value=1.0, label="centroid")
                outside_w = gr.Number(value=1.0, label="outside")

        with gr.Accordion("General guidance sampling", open=True):
            with gr.Row():
                steps = gr.Slider(4, 60, value=30, step=1, label="Denoising steps")
                guide_steps = gr.Slider(0, 30, value=10, step=1, label="Guided steps")
                strength = gr.Slider(0, 0.1, value=0.01, step=0.001, label="Guidance strength")
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="Seed")
                max_side = gr.Slider(128, 768, value=256, step=64, label="Maximum input side (px)")

        with gr.Accordion("Backend-specific configuration", open=False):
            flux_group = gr.Group(visible=False)
            with flux_group:
                gr.Markdown("### FLUX.2 Klein")
                with gr.Row():
                    flux_mode = gr.Dropdown(["base", "inpaint"], value="base", label="Mode")
                    flux_inpaint_strength = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Inpaint strength")
                    flux_cfg = gr.Slider(1.0, 12.0, value=4.0, step=0.5, label="CFG scale")
            pix_group = gr.Group(visible=True)
            with pix_group:
                gr.Markdown("### InstructPix2Pix")
                with gr.Row():
                    pix_text_cfg = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="Text guidance scale")
                    pix_image_cfg = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Image guidance scale")

        with gr.Accordion("Attention heads", open=False):
            with gr.Row():
                refresh_heads = gr.Button("Load available layers / heads")
                attention_info = gr.Markdown("Load attention choices after selecting a backend.")
            selected_layers = gr.CheckboxGroup(label="Guidance attention layers", choices=[])
            selected_heads = gr.CheckboxGroup(
                label="Include specific heads (optional)", choices=[],
                info="Empty = all heads from checked layers. Applies to Uniform, Adaptive and Learned."
            )
            head_mode = gr.Radio(["Uniform", "Adaptive", "Manual", "Learned"], value="Uniform", label="Head policy")
            manual_group = gr.Group(visible=False)
            with manual_group:
                gr.Markdown("Select heads above, then edit the weight values below.")
                manual_rows = gr.Dataframe(
                    headers=["head", "weight"],
                    datatype=["str", "number"],
                    row_count=(0, "dynamic"),
                    col_count=(2, "fixed"),
                    interactive=True,
                    label="Selected heads and weights",
                )
            adaptive_group = gr.Group(visible=False)
            with adaptive_group:
                with gr.Row():
                    adaptive_temperature = gr.Number(value=0.01, label="temperature")
                    adaptive_ema = gr.Number(value=0.5, label="ema")
                    adaptive_mix = gr.Number(value=0.1, label="uniform mix")
            learned_group = gr.Group(visible=False)
            with learned_group:
                gr.Markdown("### Trainable head weights (higher-order gradients)")
                learned_checkpoint = gr.Textbox(
                    value="outputs/learned_heads.json", label="Weights checkpoint (JSON)",
                    info="Train using the controls below, or enter an existing checkpoint for inference."
                )
                with gr.Row():
                    dataset_file = gr.File(label="Upload dataset ZIP (manifest.json + images)",
                                           file_types=[".zip"], type="filepath")
                    dataset_path = gr.Textbox(label="Or dataset folder / manifest.json on server", value="")
                with gr.Row():
                    learned_temperature = gr.Number(value=1.0, label="Learned softmax temperature")
                    training_epochs = gr.Number(value=1, precision=0, label="Epochs")
                    learning_rate = gr.Number(value=0.001, label="Learning rate")
                    optimizer_name = gr.Dropdown(["adamw", "adam", "sgd"], value="adamw", label="Optimizer")
                    weight_decay = gr.Number(value=0.0, label="Weight decay")
                with gr.Row():
                    image_loss_name = gr.Dropdown(["l1", "mse", "charbonnier", "smooth_l1"], value="l1", label="Image pixel loss")
                    inside_weight = gr.Number(value=1.0, label="Edit-region weight")
                    outside_weight = gr.Number(value=1.0, label="Preservation weight")
                    lpips_weight = gr.Number(value=0.0, label="Global LPIPS weight (optional)")
                    grad_clip = gr.Number(value=1.0, label="Gradient clipping norm")
                gr.Markdown("**Note:** Training backpropagates through every denoising step and each guidance gradient. "
                            "Start with 4 denoising steps, 1–2 guided steps, and 128–256px images; "
                            "larger settings can exceed GPU memory.")
                train_button = gr.Button("Train head weights", variant="secondary")
                train_status = gr.Textbox(label="Training status", interactive=False)

        with gr.Accordion("Propagation", open=False):
            use_propagation = gr.Checkbox(value=False, label="Enable affinity-based propagation")
            propagation_layers = gr.CheckboxGroup(label="Affinity layers", choices=[])
            with gr.Row():
                propagation_start = gr.Number(value=3, label="start step")
                propagation_stop = gr.Number(value=10, label="stop step")
                propagation_limit = gr.Number(value=12, label="max new cells / step")
            with gr.Row():
                propagation_threshold = gr.Number(value=0.2, label="similarity threshold")
                propagation_max_fraction = gr.Number(value=0.3, label="max area fraction")

        with gr.Accordion("Diagnostics", open=False):
            save_maps = gr.Checkbox(value=False, label="Save per-head maps / PNGs")

        submit = gr.Button("Generate", variant="primary")

        canvas.change(lambda current_tool, points: status_for_tool(current_tool, points or []), [tool, canvas_points], [canvas_status])
        canvas.select(on_canvas_select, [canvas, tool, canvas_points], [canvas_points, canvas, canvas_status])
        tool.change(on_canvas_change, [canvas, tool, canvas_points], [canvas, canvas_status])
        clear_shape.click(clear_box_point, [canvas, tool], [canvas_points, canvas, canvas_status])
        clear_brush.click(clear_scribble, [canvas, tool, canvas_points], [canvas, canvas_status])
        clear_all_btn.click(clear_all, [canvas, tool], [canvas, canvas_points, canvas_status])

        model.change(update_model_panels, [model], [flux_group, pix_group])
        refresh_heads.click(
            refresh_attention_choices,
            [model, selected_layers, selected_heads],
            [attention_catalog, selected_layers, propagation_layers, selected_heads, manual_rows, attention_info],
        )
        selected_layers.change(update_heads_from_layers, [attention_catalog, selected_layers, selected_heads], [selected_heads, manual_rows, attention_info])
        selected_heads.change(update_manual_weight_table, [selected_heads], [manual_rows])
        head_mode.change(update_head_mode, [head_mode], [manual_group, adaptive_group, learned_group])
        train_button.click(
            train_heads,
            [dataset_file, dataset_path, model, selected_losses,
             coverage_w, focal_w, moment_w, centerline_w, centroid_w, outside_w,
             steps, guide_steps, strength, seed, max_side,
             flux_mode, flux_inpaint_strength, flux_cfg, pix_text_cfg, pix_image_cfg,
             selected_layers, selected_heads, learned_temperature,
             use_propagation, propagation_layers, propagation_start, propagation_stop,
             propagation_limit, propagation_threshold, propagation_max_fraction,
             training_epochs, learning_rate, optimizer_name, weight_decay, image_loss_name,
             inside_weight, outside_weight, lpips_weight, grad_clip, learned_checkpoint],
            [learned_checkpoint, train_status], concurrency_limit=1,
        )

        submit.click(
            run,
            [canvas, canvas_points, tool, prompt, object_name,
             selected_losses, coverage_w, focal_w, moment_w, centerline_w, centroid_w, outside_w,
             model, steps, guide_steps, strength, seed, max_side,
             flux_mode, flux_inpaint_strength, flux_cfg,
             pix_text_cfg, pix_image_cfg,
             head_mode, selected_layers, selected_heads, manual_rows, adaptive_temperature, adaptive_ema, adaptive_mix,
        learned_checkpoint,
             use_propagation, propagation_layers, propagation_start, propagation_stop, propagation_limit,
             propagation_threshold, propagation_max_fraction,
             save_maps],
            [output, run_status],
            concurrency_limit=1,
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    build_demo().queue().launch(server_name=args.host, server_port=args.port)
