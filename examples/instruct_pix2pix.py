"""Legacy short command for Pix2Pix; delegates to the shared generate CLI/config."""
from __future__ import annotations
import argparse
from generate import main as generate_main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--prompt", default=None)
    p.add_argument("--object", default=None)
    p.add_argument("--point", default=None, help="x,y in image coordinates")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guide-steps", type=int, default=None)
    p.add_argument("--max-side", type=int, default=None)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args(argv)
    params = ["--backend", "pix2pix", "--image", args.image]
    if args.point is not None:
        x, y = args.point.split(",")
        params += ["--point", x, y]
    for arg, field in [("--prompt", "prompt"), ("--object", "object"),
                       ("--steps", "steps"), ("--guided-steps", "guide_steps"),
                       ("--max-side", "max_side"), ("--output-dir", "out"),
                       ("--config", "config"), ("--checkpoint", "checkpoint")]:
        value = getattr(args, field)
        if value is not None:
            params += [arg, str(value)]
    if args.adaptive:
        params += ["--policy", "Adaptive"]
    return generate_main(params)


if __name__ == "__main__":
    main()
