"""Paired learned-vs-uniform inference comparison on a ground-truth manifest.

Both policies use identical input images, visual prompts, text prompts,
head sets, sampling settings and random seeds. The trained checkpoint sets
which heads are eligible; only the relative head weights differ.

Example:
  uv run --extra flux2 --extra perceptual python examples/compare_heads.py \
    --checkpoint outputs/train/RUN/learned_heads.json \
    --dataset data/compbench-vpg/add/manifest_scribble.json \
    --max-samples 5

By default the script loads settings saved beside the checkpoint (config.json),
so evaluation uses the actual training backend/settings rather than a possibly
out-of-sync inference profile. You may override steps and guided steps.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from visual_prompt_guidance import Request, Learned
from visual_prompt_guidance.config import profile_settings
from visual_prompt_guidance.experiments import create_run_dir, save_run_config
from visual_prompt_guidance.runtime import build_backend, build_engine, model_options
from visual_prompt_guidance.training import EditingDataset, ImageLossConfig, ImageObjective, _rgb_tensor


def build_comparison_settings(settings: dict, checkpoint: str) -> tuple[dict, dict]:
    """Use the identical head subset for both policies; only logits differ."""
    policy = Learned.load(checkpoint)
    selected = list(policy.selected or policy.labels or ())
    if not selected:
        raise ValueError('Checkpoint contains no trained attention heads')
    layers = list(dict.fromkeys(label.rsplit(':H', 1)[0] for label in policy.labels))
    common = dict(settings)
    # Keep configured propagation behavior and all guidance settings unchanged.
    common.update(selected_layers=layers, selected_heads=selected)
    uniform = dict(common, head_mode='Uniform', learned_checkpoint='')
    learned = dict(common, head_mode='Learned', learned_checkpoint=str(checkpoint))
    return uniform, learned


def load_snapshot(checkpoint: Path, config_path: str | None) -> dict:
    snapshot = checkpoint.parent / 'config.json'
    if snapshot.exists():
        settings = json.loads(snapshot.read_text(encoding='utf-8')).get('settings')
        if isinstance(settings, dict):
            return settings
    return profile_settings('train', path=config_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--guided-steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dtype', default=None)
    parser.add_argument('--max-side', type=int, default=None)
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    settings = load_snapshot(checkpoint, args.config)
    for cli_key, name in [('steps', 'steps'), ('guided_steps', 'guide_steps'),
                          ('seed', 'seed'), ('device', 'device'), ('dtype', 'dtype'),
                          ('max_side', 'max_side')]:
        val = getattr(args, cli_key)
        if val is not None:
            settings[name] = val
    uniform_settings, learned_settings = build_comparison_settings(settings, str(checkpoint))
    criterion = ImageObjective(ImageLossConfig(
        edit_loss=settings['edit_loss_name'], edit_weight=float(settings['edit_loss_weight']),
        background_loss=settings['background_loss_name'],
        background_weight=float(settings['background_loss_weight'])))

    shared_backend = build_backend(settings)  # Model is loaded once, not twice.
    engines = {'uniform': build_engine(uniform_settings, backend=shared_backend),
               'learned': build_engine(learned_settings, backend=shared_backend)}
    options = model_options(settings)
    folder = create_run_dir('compare')
    save_run_config(folder, {'checkpoint': str(checkpoint), 'dataset': args.dataset,
                             'settings': settings, 'comparison': 'uniform vs learned',
                             'same_seeds': True})
    records = []
    with EditingDataset(args.dataset, options=options) as dataset:
        limit = min(len(dataset), args.max_samples) if args.max_samples is not None else len(dataset)
        if limit < 1:
            parser.error('--max-samples must be positive')
        for index in range(limit):
            example = dataset[index]
            # Matching seeds within each pair, and deterministic per-example variation.
            paired_seed = int(options['seed']) + index
            request = Request(data=example.request.data,
                              options={**example.request.options, 'seed': paired_seed})
            scores = {}
            for mode in ('uniform', 'learned'):
                result = engines[mode].run(request, example.visual)
                image = result.output
                image.save(folder / f'{index:03d}_{mode}.png')
                predicted = _rgb_tensor(image, device=torch.device('cpu'))
                _, parts = criterion.evaluate(predicted, example, result.target.edit_mask)
                total = sum((settings['edit_loss_weight'] if key.startswith('edit/') else
                             settings['background_loss_weight']) * val
                            for key, val in parts.items())
                scores[mode] = {'loss': float(total), **parts}
                del result, predicted
            records.append({'sample': index, 'seed': paired_seed,
                            'uniform_loss': scores['uniform']['loss'],
                            'learned_loss': scores['learned']['loss'],
                            'delta_learned_minus_uniform': scores['learned']['loss'] - scores['uniform']['loss']})
            print(f"{index+1}/{limit}: uniform={scores['uniform']['loss']:.6f} "
                  f"learned={scores['learned']['loss']:.6f}", flush=True)

    with (folder / 'paired_losses.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        'samples': len(records),
        'mean_uniform_loss': sum(r['uniform_loss'] for r in records) / len(records),
        'mean_learned_loss': sum(r['learned_loss'] for r in records) / len(records),
        'learned_better_samples': sum(r['learned_loss'] < r['uniform_loss'] for r in records),
        'mean_delta_learned_minus_uniform':
            sum(r['delta_learned_minus_uniform'] for r in records) / len(records),
    }
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Saved {folder} · mean delta (learned - uniform) = "
          f"{summary['mean_delta_learned_minus_uniform']:+.6f} (negative is better)")
    return summary


if __name__ == '__main__':
    main()
