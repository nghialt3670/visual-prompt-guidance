"""Generate using the saved Generate profile and optional CLI overrides.

Example:
  uv run --extra pix2pix python examples/generate.py --dataset data/compbench-vpg/add/manifest_scribble.json
  uv run --extra flux2 python examples/generate.py --backend flux2 --dataset data/compbench-vpg/add/manifest_scribble.json --checkpoint outputs/train/RUN/learned_heads.json

Input source: --dataset [--sample-index] OR --image, --prompt, --object, and visual annotation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
from visual_prompt_guidance.config import resolve_settings, save_profile, config_path
from visual_prompt_guidance.runtime import build_engine, canonical_model, model_options, build_observers
from visual_prompt_guidance import Point, Box, Scribble, Request, FileObserver
from visual_prompt_guidance.training import EditingDataset
from visual_prompt_guidance.experiments import create_run_dir, save_run_config


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--save-config", action="store_true")
    p.add_argument("--backend", choices=["flux2", "pix2pix"], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--model-id", default=None)
    p.add_argument("--dataset", default=None, help="Dataset folder / specific manifest / ZIP")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--image", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--object", default=None, help="Exact target phrase occurring in text prompt")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"))
    group.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    group.add_argument("--scribble-mask", default=None)
    group.add_argument("--scribble-path", default=None, help="JSON array of [x,y] ordered points, or path to JSON file")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guided-steps", type=int, default=None)
    p.add_argument("--max-side", type=int, default=None)
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--layer", action="append", default=None)
    p.add_argument("--head", action="append", default=None)
    p.add_argument("--policy", choices=["Uniform", "Adaptive", "Manual", "Learned"], default=None)
    p.add_argument("--checkpoint", default=None, help="Trained learned_heads.json; sets policy to Learned")
    p.add_argument("--guidance-loss", action="append", default=None)
    p.add_argument("--text-cfg", type=float, default=None)
    p.add_argument("--image-cfg", type=float, default=None)
    p.add_argument("--flux-mode", choices=["base", "inpaint"], default=None)
    p.add_argument("--inpaint-strength", type=float, default=None)
    p.add_argument("--propagation", action="store_true", default=None)
    p.add_argument("--save-maps", action="store_true", default=None)
    p.add_argument("--output-dir", default=None)
    return p


def settings_from_args(args):
    fields = {"device": "device", "dtype": "dtype", "model_id": "model_id", "steps": "steps",
              "guided_steps": "guide_steps", "max_side": "max_side", "strength": "strength",
              "seed": "seed", "layer": "selected_layers", "head": "selected_heads",
              "policy": "head_mode", "guidance_loss": "selected_losses",
              "image_cfg": "pix_image_cfg", "flux_mode": "flux_mode",
              "inpaint_strength": "flux_inpaint_strength", "propagation": "use_propagation",
              "save_maps": "save_maps", "prompt": "prompt", "object": "object_name"}
    overrides = {dest: getattr(args, key) for key, dest in fields.items() if getattr(args, key) is not None}
    if args.backend is not None:
        overrides["model"] = canonical_model(args.backend)
    model = overrides.get("model") or resolve_settings("generate", path=args.config)["model"]
    if args.text_cfg is not None:
        overrides["flux_cfg" if canonical_model(model) == "FLUX.2 Klein" else "pix_text_cfg"] = args.text_cfg
    if args.checkpoint is not None:
        overrides.update(head_mode="Learned", learned_checkpoint=args.checkpoint)
        # Checkpoint specifies its own layer/head identities. Avoid stale UI selections.
        if args.layer is None:
            overrides["selected_layers"] = []
        if args.head is None:
            overrides["selected_heads"] = []
    return overrides


def choose_visual(args):
    if args.point is not None:
        return Point(*args.point)
    if args.box is not None:
        return Box(*args.box)
    if args.scribble_mask:
        with Image.open(args.scribble_mask) as mask:
            return Scribble(mask=mask.convert("L"))
    if args.scribble_path:
        raw = Path(args.scribble_path)
        coords = json.loads(raw.read_text() if raw.is_file() else args.scribble_path)
        return Scribble(path=[tuple(map(float, p)) for p in coords])
    raise ValueError("Specify a visual prompt: --point, --box, --scribble-mask, or --scribble-path")


def main(argv=None):
    args = make_parser().parse_args(argv)
    overrides = settings_from_args(args)
    settings = resolve_settings("generate", overrides, path=args.config)
    if args.save_config:
        save_profile("generate", overrides, path=args.config)
    if args.dataset:
        with EditingDataset(args.dataset, options=model_options(settings)) as dataset:
            example = dataset[args.sample_index]
        request, visual = example.request, example.visual
        source = request.data["image"]
    else:
        if not args.image:
            raise ValueError("Specify --dataset or --image")
        if not settings.get("prompt") or not settings.get("object_name"):
            raise ValueError("Set prompt and object phrase in saved config or --prompt/--object")
        with Image.open(args.image) as image:
            source = image.convert("RGB")
        visual = choose_visual(args)
        request = Request({"image": source, "prompt": settings["prompt"],
                           "object_phrase": settings["object_name"]}, options=model_options(settings))
    if args.output_dir:
        folder = Path(args.output_dir)
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Run directory must be empty: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
    else:
        folder = create_run_dir("generate")
    engine = build_engine(settings, observers=build_observers(settings, folder))
    save_run_config(folder, {"workflow": "generate", "source": "cli", "settings": settings,
                             "config_source": str(config_path(args.config)),
                             "dataset": args.dataset, "sample_index": args.sample_index,
                             "model_options": request.options})
    source.save(folder / "input.png")
    result = engine.run(request, visual)
    result.output.save(folder / "result.png")
    print(f"Saved {folder / 'result.png'}")
    return result


if __name__ == "__main__":
    main()
