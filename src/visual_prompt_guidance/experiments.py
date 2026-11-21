"""Run directories and dependency-light training reports.

These utilities are optional: the core Engine still accepts arbitrary observers
and paths. Applications may use them without importing a UI or a model backend.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw


def create_run_dir(kind: str, root: str | Path = "outputs") -> Path:
    """Create a non-overwriting `outputs/{kind}/{timestamp}/` folder."""
    if not kind or kind in {".", ".."} or not kind.replace("_", "").replace("-", "").isalnum():
        raise ValueError("kind must be a simple run category")
    parent = Path(root) / kind
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for suffix in range(1000):
        candidate = parent / (stamp if suffix == 0 else f"{stamp}_{suffix}")
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Cannot allocate a unique run directory")


def save_run_config(directory: str | Path, config: Mapping[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(dict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_training_reports(directory: str | Path, history: Sequence[Mapping[str, Any]],
                          labels: Sequence[str], weights: Sequence[float]) -> tuple[Path, Path]:
    """Save a readable weight table and loss curve without a plotting dependency."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "head_weights.csv"
    if len(labels) != len(weights):
        raise ValueError("Head labels and weights must have the same length")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["head", "weight"])
        for name, value in sorted(zip(labels, weights), key=lambda item: item[1], reverse=True):
            writer.writerow([name, float(value)])

    chart_path = directory / "loss_curve.png"
    width, height = 1000, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 95, 45, width - 36, height - 72
    draw.line([(left, top), (left, bottom), (right, bottom)], fill="#334155", width=3)
    draw.text((left, 14), "Training image loss", fill="#0f172a")
    draw.text((right - 140, bottom + 28), "Training iteration", fill="#334155")
    values = [float(entry["loss"]) for entry in history]
    if values:
        lo, hi = min(values), max(values)
        spread = max(hi - lo, abs(hi) * 0.02, 1e-7)
        lo -= spread * 0.06
        hi += spread * 0.06
        for index in range(5):
            val = lo + (hi - lo) * index / 4
            y = round(bottom - index * (bottom - top) / 4)
            draw.line((left, y, right, y), fill="#e2e8f0", width=1)
            draw.text((7, y - 7), f"{val:.5g}", fill="#475569")
        coords = [
            (round(left + i * (right - left) / max(len(values) - 1, 1)),
             round(bottom - (value - lo) / (hi - lo) * (bottom - top)))
            for i, value in enumerate(values)
        ]
        if len(coords) > 1:
            draw.line(coords, fill="#2563eb", width=3)
        for x, y in (coords if len(coords) <= 40 else [coords[0], coords[-1]]):
            draw.ellipse((x-3, y-3, x+3, y+3), fill="#2563eb")
        draw.text((left, bottom + 26), "1", fill="#475569")
        draw.text((right - 30, bottom + 26), str(len(values)), fill="#475569")
    else:
        draw.text((left + 24, top + 20), "No training history", fill="#475569")
    image.save(chart_path)
    return csv_path, chart_path
