"""Standalone dataset-based Learned head training without running Gradio.

Example:
  uv run --extra pix2pix python examples/train_heads.py \
    --backend pix2pix --dataset ./data/my_edits --steps 4 \
    --guided-steps 1 --max-side 128 --epochs 1 --lr 0.001
"""
from __future__ import annotations
import argparse
from pathlib import Path
from visual_prompt_guidance.experiments import create_run_dir, save_run_config, save_training_reports

from visual_prompt_guidance import (
    Engine, Guidance, Learned, Loss, Objective, HeadTrainer, TrainConfig,
    ImageLossConfig, losses,
)
from visual_prompt_guidance.backends import Flux2Klein, InstructPix2Pix


def main():
    parser = argparse.ArgumentParser(description="Learn attention head weights from reference edited images")
    parser.add_argument("--backend", choices=["pix2pix", "flux2"], default="pix2pix")
    parser.add_argument("--dataset", required=True, help="Dataset directory, manifest.json or ZIP")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guided-steps", type=int, default=1)
    parser.add_argument("--strength", type=float, default=.01)
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", action="append", default=None, help="Repeat for each selected layer")
    parser.add_argument("--head", action="append", default=None, help="Repeat for each selected head")
    parser.add_argument("--guidance-loss", choices=list(losses.BUILTIN_LOSSES), default="coverage")
    parser.add_argument("--guidance-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=["l1", "mse", "charbonnier", "smooth_l1"], default="l1")
    parser.add_argument("--inside-weight", type=float, default=1.0)
    parser.add_argument("--outside-weight", type=float, default=1.0)
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None, help="Run folder; default outputs/train/<timestamp>")
    parser.add_argument("--output", default=None, help="Legacy checkpoint path; optional")
    parser.add_argument("--text-cfg", type=float, default=None)
    parser.add_argument("--image-cfg", type=float, default=1.5)
    parser.add_argument("--flux-mode", choices=["base", "inpaint"], default="base")
    parser.add_argument("--inpaint-strength", type=float, default=0.8)
    args = parser.parse_args()

    backend = (InstructPix2Pix(device=args.device, dtype=args.dtype)
               if args.backend == "pix2pix" else Flux2Klein(device=args.device, dtype=args.dtype))
    policy = Learned(selected=tuple(args.head) if args.head else None, temperature=args.temperature)
    runner = Engine(backend, Objective(Loss(args.guidance_loss,
                                           losses.BUILTIN_LOSSES[args.guidance_loss],
                                           weight=args.guidance_loss_weight)),
                    guidance=Guidance(strength=args.strength, max_steps=args.guided_steps,
                                      layers=tuple(args.layer) if args.layer else None),
                    policy=policy)
    if args.output_dir and args.output:
        parser.error("Choose either --output-dir or --output")
    if args.output:
        # Backward compatibility with existing scripts that specify a checkpoint.
        checkpoint_path = Path(args.output)
        run_dir = checkpoint_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
    elif args.output_dir:
        run_dir = Path(args.output_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            parser.error(f"Run folder must be empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "learned_heads.json"
    else:
        run_dir = create_run_dir("train")
        checkpoint_path = run_dir / "learned_heads.json"
    trainer = HeadTrainer(runner,
        config=TrainConfig(epochs=args.epochs, learning_rate=args.lr,
                           optimizer=args.optimizer, weight_decay=args.weight_decay,
                           grad_clip=args.grad_clip,
                           checkpoint=str(checkpoint_path),
                           history=str(run_dir / "history.json")),
        image_loss=ImageLossConfig(pixel=args.loss, inside_weight=args.inside_weight,
                                   outside_weight=args.outside_weight, lpips_weight=args.lpips_weight))
    options = {"steps":args.steps, "max_side":args.max_side, "seed":args.seed,
               "cfg": args.text_cfg if args.text_cfg is not None else (7.5 if args.backend == "pix2pix" else 4.0)}
    if args.backend == "pix2pix":
        options.update(image_cfg=args.image_cfg)
    else:
        options.update(mode=args.flux_mode, inpaint_strength=args.inpaint_strength)
    save_run_config(run_dir, {
        "workflow": "train", "source": "cli", "model": args.backend,
        "dataset": args.dataset, "guidance": vars(runner.guidance),
        "losses": [{"name": args.guidance_loss, "weight": args.guidance_loss_weight}],
        "selected_heads": args.head, "selected_layers": args.layer,
        "train_config": vars(trainer.config), "image_loss": vars(trainer.criterion.config),
        "model_options": options, "dtype": args.dtype, "device": args.device,
        "learned_temperature": args.temperature,
    })
    result = trainer.fit(args.dataset, options=options,
        progress=lambda n,total,event: print(f"{n}/{total} image loss={event['loss']:.6f} grad={event['grad_norm']:.5g}",flush=True))
    save_training_reports(run_dir, result.history, result.head_labels, result.weights)
    print(f"Saved training run to {run_dir} ({len(result.weights)} heads)")


if __name__ == "__main__":
    main()
