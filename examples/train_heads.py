"""Standalone dataset-based Learned head training without running Gradio.

Example:
  uv run --extra pix2pix python examples/train_heads.py \
    --backend pix2pix --dataset ./data/my_edits --steps 4 \
    --guided-steps 1 --max-side 128 --epochs 1 --lr 0.001
"""
from __future__ import annotations
import argparse
from pathlib import Path

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
    parser.add_argument("--output", default="outputs/learned_heads.json")
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
    output = Path(args.output)
    trainer = HeadTrainer(runner,
        config=TrainConfig(epochs=args.epochs, learning_rate=args.lr,
                           optimizer=args.optimizer, weight_decay=args.weight_decay,
                           grad_clip=args.grad_clip,
                           checkpoint=str(output),
                           history=str(output.with_suffix(".history.json"))),
        image_loss=ImageLossConfig(pixel=args.loss, inside_weight=args.inside_weight,
                                   outside_weight=args.outside_weight, lpips_weight=args.lpips_weight))
    options = {"steps":args.steps, "max_side":args.max_side, "seed":args.seed,
               "cfg": args.text_cfg if args.text_cfg is not None else (7.5 if args.backend == "pix2pix" else 4.0)}
    if args.backend == "pix2pix":
        options.update(image_cfg=args.image_cfg)
    else:
        options.update(mode=args.flux_mode, inpaint_strength=args.inpaint_strength)
    result = trainer.fit(args.dataset, options=options,
        progress=lambda n,total,event: print(f"{n}/{total} image loss={event['loss']:.6f} grad={event['grad_norm']:.5g}",flush=True))
    print(f"Saved {result.checkpoint} ({len(result.weights)} heads)")


if __name__ == "__main__":
    main()
