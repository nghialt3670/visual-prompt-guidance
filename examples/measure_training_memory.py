"""Measure per-phase CUDA memory for one training sample.

Accepts the same overrides as train_heads.py and stops after the first sample.
Each row's peak_gib is the peak since the previous row, so rows partition the
timeline. Requires CUDA; library code is not modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import train_heads

from visual_prompt_guidance import HeadTrainer
from visual_prompt_guidance.config import resolve_settings
from visual_prompt_guidance.experiments import create_run_dir, save_run_config
from visual_prompt_guidance.runtime import (
    build_engine,
    model_options,
    resolve_training_precision,
    train_config,
)

GIB = 1024 ** 3
REPORTED_SETTINGS = ("model", "dtype", "steps", "guide_steps", "max_side", "train_query_chunk",
                     "train_checkpoint_transformer", "train_checkpoint_attention", "train_checkpoint_vae")


class _Stop(Exception):
    """Raised after the first sample to end the measurement."""


class Probe:
    def __init__(self):
        self.rows: list[dict] = []
        self.forward_calls = 0
        self.diagnosed = False

    def mark(self, name: str) -> None:
        self.rows.append({
            "phase": name,
            "peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
            "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 3),
        })
        torch.cuda.reset_peak_memory_stats()

    def diagnose(self, output) -> None:
        self.diagnosed = True
        logits = self.engine.policy.logits
        print(f"[probe] current device: cuda:{torch.cuda.current_device()} of {torch.cuda.device_count()}", flush=True)
        print(f"[probe] logits device: {None if logits is None else logits.device}", flush=True)
        grad_fn = getattr(output, "grad_fn", None)
        print(f"[probe] output requires_grad={getattr(output, 'requires_grad', None)} "
              f"grad_fn={type(grad_fn).__name__ if grad_fn is not None else None}", flush=True)
        for index in range(torch.cuda.device_count()):
            print(f"[probe] cuda:{index} allocated={torch.cuda.memory_allocated(index) / GIB:.3f} GiB", flush=True)

    def attach(self, engine) -> None:
        self.engine = engine
        backend = engine.backend

        original_prepare = backend.prepare

        def prepare(*args, **kwargs):
            out = original_prepare(*args, **kwargs)
            self.mark("prepare (text encoding, load)")
            return out

        backend.prepare = prepare

        original_forward = backend.forward

        def forward(*args, **kwargs):
            guided = kwargs.get("attention") is not None
            out = original_forward(*args, **kwargs)
            self.forward_calls += 1
            kind = "guided pre-forward" if guided else "post-forward"
            self.mark(f"{kind} #{self.forward_calls}")
            return out

        backend.forward = forward

        original_updater = engine.updater

        def updater(*args, **kwargs):
            # updater runs immediately after torch.autograd.grad(..., create_graph=True)
            self.mark(f"guided inner backward #{self.forward_calls}")
            return original_updater(*args, **kwargs)

        engine.updater = updater

        if hasattr(backend, "decode_tensor"):
            original_decode = backend.decode_tensor

            def decode_tensor(*args, **kwargs):
                out = original_decode(*args, **kwargs)
                self.mark("VAE decode")
                return out

            backend.decode_tensor = decode_tensor


class _CriterionProbe:
    def __init__(self, inner, probe: Probe):
        self.inner, self.probe = inner, probe

    def __call__(self, output, *args, **kwargs):
        if not self.probe.diagnosed:
            self.probe.diagnose(output)
        out = self.inner(output, *args, **kwargs)
        self.probe.mark("image loss")
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main(argv=None):
    parser = train_heads.make_parser()
    parser.add_argument("--report", default=None, help="JSON output path (default: run folder probe.json)")
    parser.add_argument("--snapshot", default=None, help="Write a CUDA memory snapshot pickle for the sample")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("CUDA is required for memory measurement")

    overrides = train_heads.settings_from_args(args)
    overrides["head_mode"] = "Learned"
    settings = resolve_settings("train", overrides, path=args.config)
    settings = resolve_training_precision(settings)
    device = str(settings.get("device", "auto"))
    if device != "auto" and not device.startswith("cuda"):
        parser.error(f"Measurement needs a CUDA device, got {device}")
    if not settings["dataset_path"]:
        parser.error("Dataset not configured: pass --dataset or set train.dataset_path in project config")

    folder = create_run_dir("memprobe")
    checkpoint = folder / "learned_heads.json"
    engine = build_engine(settings, training=True)
    train_cfg, image_cfg = train_config(settings, checkpoint, folder / "history.json")
    trainer = HeadTrainer(engine, config=train_cfg, image_loss=image_cfg)
    options = model_options(settings)
    save_run_config(folder, {"workflow": "memprobe", "settings": settings, "dataset": settings["dataset_path"],
                             "model_options": options})

    probe = Probe()
    probe.attach(engine)
    trainer.criterion = _CriterionProbe(trainer.criterion, probe)
    probe.mark("after build_engine (resident weights)")

    def on_progress(n, total, event):
        probe.mark("backward and optimizer step")
        raise _Stop

    if args.snapshot:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
    try:
        trainer.fit(settings["dataset_path"], options=options, progress=on_progress)
    except _Stop:
        pass
    finally:
        if args.snapshot:
            torch.cuda.memory._dump_snapshot(args.snapshot)
            torch.cuda.memory._record_memory_history(enabled=None)

    report = {
        "settings": {key: settings.get(key) for key in REPORTED_SETTINGS},
        "max_peak_gib": max(row["peak_gib"] for row in probe.rows),
        "phases": probe.rows,
    }
    path = Path(args.report) if args.report else folder / "probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    for row in probe.rows:
        print(f"{row['phase']:<52} peak={row['peak_gib']:8.3f} GiB  allocated={row['allocated_gib']:8.3f} GiB")
    print(f"max peak={report['max_peak_gib']:.3f} GiB; report: {path}")
    return report


if __name__ == "__main__":
    main()
