"""Run: uv run --extra pix2pix python examples/instruct_pix2pix.py input.png"""
import argparse
from pathlib import Path
from PIL import Image
from visual_prompt_guidance import (
    Engine, Guidance, Request, Point, Box, Scribble,
    Objective, Loss, Uniform, Adaptive, FileObserver, losses,
)
from visual_prompt_guidance.backends import InstructPix2Pix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Source image path")
    parser.add_argument("--prompt", default="make the bird blue", help="Editing instruction containing the object name")
    parser.add_argument("--object", default="bird")
    parser.add_argument("--point", default="128,128", help="Point x,y in SOURCE image pixels")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guide-steps", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--out", default="outputs/instruct_pix2pix")
    args = parser.parse_args()
    x, y = (float(v) for v in args.point.split(","))
    source = Image.open(args.image).convert("RGB")
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    model = InstructPix2Pix()  # lazy model load; also supports device='mps', dtype='fp16'
    engine = Engine(
        backend=model,
        objective=Objective(Loss("coverage", losses.coverage)),
        policy=Adaptive(temperature=.01) if args.adaptive else Uniform(),
        guidance=Guidance(strength=.01, max_steps=args.guide_steps),
        observers=[FileObserver(folder, maps=False)],
    )
    request = Request(
        {"image": source, "prompt": args.prompt, "object_phrase": args.object},
        options={"steps": args.steps, "max_side": args.max_side, "seed": 42,
                 "cfg": 7.5, "image_cfg": 1.5},
    )
    result = engine.run(request, Point(x, y, sigma=16))
    result.output.save(folder / "result.png")
    print(f"Saved: {folder / 'result.png'}")


if __name__ == "__main__":
    main()
