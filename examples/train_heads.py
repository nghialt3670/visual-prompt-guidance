"""Train head weights using the same project configuration as Gradio.

Without CLI overrides, this reads configs/project.json -> train profile.
Overrides affect the current run only; --save-config writes them back.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from visual_prompt_guidance.config import resolve_settings, save_profile, config_path
from visual_prompt_guidance.devices import device_choices
from visual_prompt_guidance.runtime import build_engine, canonical_model, model_options, train_config, resolve_training_precision
from visual_prompt_guidance import HeadTrainer, losses
from visual_prompt_guidance.experiments import create_run_dir, save_run_config, save_training_reports


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None, help="Shared project configuration (default VPG_CONFIG or configs/project.json)")
    p.add_argument("--save-config", action="store_true", help="Persist supplied overrides to the train profile")
    p.add_argument("--backend", choices=["pix2pix", "flux2"], default=None)
    p.add_argument("--dataset", default=None, help="Dataset folder, manifest JSON, or ZIP")
    p.add_argument("--device", default=None, help=f"Device override, one of: {', '.join(device_choices())}")
    p.add_argument("--dtype", default=None)
    p.add_argument("--model-id", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guided-steps", type=int, default=None)
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--max-side", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--layer", action="append", default=None)
    p.add_argument("--head", action="append", default=None)
    p.add_argument("--all-heads", action="store_true",
                   help="Clear any saved head filter and train all heads in the selected layers")
    p.add_argument("--guidance-loss", action="append", choices=list(losses.BUILTIN_LOSSES), default=None,
                   help="Repeat for multiple guidance losses; weights default from shared config")
    p.add_argument("--guidance-loss-weight", type=float, default=None,
                   help="Apply this weight to CLI-specified guidance losses")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--no-transformer-checkpoint", action="store_false",
                   dest="train_checkpoint_transformer", default=None,
                   help="Disable FLUX recomputation of transformer activations")
    p.add_argument("--no-attention-checkpoint", action="store_false",
                   dest="train_checkpoint_attention", default=None,
                   help="Disable FLUX recomputation of attention-probability chunks")
    p.add_argument("--no-vae-checkpoint", action="store_false",
                   dest="train_checkpoint_vae", default=None,
                   help="Disable FLUX VAE decoder checkpointing")
    p.add_argument("--autograd-offload", action="store_true",
                   dest="train_offload_autograd", default=None,
                   help="Offload FLUX autograd saved tensors to CPU RAM (needs ample host RAM)")
    p.add_argument("--no-autograd-offload", action="store_false",
                   dest="train_offload_autograd", default=None,
                   help="Disable CPU offloading of FLUX autograd saved tensors")
    p.add_argument("--attention-query-chunk", type=int, default=None,
                   help="Number of output-image queries per differentiable attention chunk")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--edit-loss", choices=["l1", "mse", "charbonnier", "smooth_l1", "lpips"], default=None)
    p.add_argument("--edit-weight", type=float, default=None)
    p.add_argument("--background-loss", choices=["l1", "mse", "charbonnier", "smooth_l1", "lpips"], default=None)
    p.add_argument("--background-weight", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--text-cfg", type=float, default=None)
    p.add_argument("--image-cfg", type=float, default=None)
    p.add_argument("--flux-mode", choices=["base", "inpaint"], default=None)
    p.add_argument("--inpaint-strength", type=float, default=None)
    p.add_argument("--propagation", dest="use_propagation", action="store_true", default=None)
    p.add_argument("--no-propagation", dest="use_propagation", action="store_false")
    p.add_argument("--affinity-layer", action="append", default=None)
    for field, kind in [("propagation-start", int), ("propagation-stop", int),
                        ("propagation-limit", int), ("propagation-threshold", float),
                        ("propagation-max-fraction", float)]:
        p.add_argument("--" + field, type=kind, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--output", default=None, help="Legacy explicit checkpoint path")
    return p


def settings_from_args(args):
    if args.all_heads and args.head:
        raise ValueError("Choose either --all-heads or --head, not both")
    direct = {
        "dataset": "dataset_path", "device": "device", "dtype": "dtype", "model_id": "model_id",
        "steps": "steps", "guided_steps": "guide_steps", "strength": "strength",
        "max_side": "max_side", "seed": "seed", "layer": "selected_layers",
        "head": "selected_heads", "guidance_loss": "selected_losses",
        "epochs": "training_epochs", "lr": "learning_rate", "optimizer": "optimizer_name",
        "grad_clip": "grad_clip", "weight_decay": "weight_decay",
        "train_checkpoint_transformer": "train_checkpoint_transformer",
        "train_checkpoint_attention": "train_checkpoint_attention",
        "train_checkpoint_vae": "train_checkpoint_vae",
        "train_offload_autograd": "train_offload_autograd",
        "attention_query_chunk": "train_query_chunk",
        "edit_loss": "edit_loss_name", "edit_weight": "edit_loss_weight",
        "background_loss": "background_loss_name", "background_weight": "background_loss_weight",
        "temperature": "learned_temperature",
        "image_cfg": "pix_image_cfg", "flux_mode": "flux_mode",
        "inpaint_strength": "flux_inpaint_strength", "use_propagation": "use_propagation",
        "affinity_layer": "propagation_layers", "propagation_start": "propagation_start",
        "propagation_stop": "propagation_stop", "propagation_limit": "propagation_limit",
        "propagation_threshold": "propagation_threshold",
        "propagation_max_fraction": "propagation_max_fraction",
    }
    overrides = {dest: getattr(args, source) for source, dest in direct.items()
                 if getattr(args, source) is not None}
    if args.all_heads:
        overrides['selected_heads'] = []
    if args.backend:
        overrides["model"] = canonical_model(args.backend)
    model = overrides.get("model") or resolve_settings("train", path=args.config)["model"]
    if args.text_cfg is not None:
        overrides["flux_cfg" if canonical_model(model) == "FLUX.2 Klein" else "pix_text_cfg"] = args.text_cfg
    if args.guidance_loss_weight is not None:
        names = overrides.get("selected_losses") or resolve_settings("train", path=args.config)["selected_losses"]
        overrides.update({f"{name}_w": args.guidance_loss_weight for name in names})
    return overrides


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    overrides = settings_from_args(args)
    overrides["head_mode"] = "Learned"  # This CLI always trains Learned policies.
    settings = resolve_settings("train", overrides, path=args.config)
    requested_dtype = settings["dtype"]
    settings = resolve_training_precision(settings)
    if requested_dtype != settings["dtype"]:
        print("[VPG training] MPS automatic precision resolved to fp32 for second-order gradients", flush=True)
    dataset = settings["dataset_path"]
    if not dataset:
        parser.error("Dataset not configured: pass --dataset or set train.dataset_path in project config")
    if args.output_dir and args.output:
        parser.error("Choose either --output-dir or --output")
    if args.save_config:
        save_profile("train", overrides, path=args.config)
    if args.output:
        checkpoint = Path(args.output)
        folder = checkpoint.parent
        folder.mkdir(parents=True, exist_ok=True)
    elif args.output_dir:
        folder = Path(args.output_dir)
        if folder.exists() and any(folder.iterdir()):
            parser.error(f"Run folder must be empty: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "learned_heads.json"
    else:
        folder = create_run_dir("train")
        checkpoint = folder / "learned_heads.json"
    engine = build_engine(settings, training=True)
    train_cfg, image_cfg = train_config(settings, checkpoint, folder / "history.json")
    trainer = HeadTrainer(engine, config=train_cfg, image_loss=image_cfg)
    options = model_options(settings)
    save_run_config(folder, {"workflow": "train", "source": "cli", "config_source": str(config_path(args.config)),
                             "settings": settings, "dataset": dataset, "model_options": options,
                             "train_config": vars(train_cfg), "image_loss": vars(image_cfg)})
    result = trainer.fit(dataset, options=options, progress=lambda n, total, event: print(
        f"{n}/{total} image loss={event['loss']:.6f} grad={event['grad_norm']:.5g}" +
        (f" CUDA peak={event['cuda_peak_allocated_gib']:.2f} GiB"
         if 'cuda_peak_allocated_gib' in event else ''), flush=True))
    save_training_reports(folder, result.history, result.head_labels, result.weights)
    print(f"Saved training run to {folder} ({len(result.weights)} heads)")
    print(f"For inference: set generate.head_mode=Learned and generate.learned_checkpoint={checkpoint}")
    return result


if __name__ == "__main__":
    main()
