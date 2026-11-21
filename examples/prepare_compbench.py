"""Sample CompBench local edits and convert to Visual Prompt Guidance datasets.

Downloads only requested rows and images via Hugging Face's dataset-viewer
/filter API. Each selected original pair is exported with point, box, and
skeleton-centerline scribble annotations derived from its supplied edit mask.

Run from project root:
    uv run --extra dataset python examples/prepare_compbench.py \
        --per-task 5 --out data/compbench-vpg --seed 42

The source dataset has *edit region masks*, which may not be a precise object
silhouette (especially for replacement/removal). Inspect the previews before
using the synthesized scribbles as shape constraints.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import heapq
import json
from pathlib import Path
import random
import re
import sys
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw
import requests
from requests.adapters import HTTPAdapter
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from urllib3.util.retry import Retry

REPO = "BohanJia/CompBench"
DATASET_URL = "https://datasets-server.huggingface.co/filter"
TASKS = ("add", "remove", "replace")
VISUALS = ("point", "box", "scribble")


def session_with_retries() -> requests.Session:
    sess = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.headers["User-Agent"] = "visual-prompt-guidance/compbench-dataset-preparer"
    return sess


def fetch_filtered(session: requests.Session, task: str, offset: int, length: int = 1) -> dict:
    params = {"dataset": REPO, "config": "default", "split": "train",
              "where": f'"task"=\'{task}\'', "offset": offset, "length": length}
    response = session.get(DATASET_URL, params=params, timeout=75)
    response.raise_for_status()
    payload = response.json()
    if "rows" not in payload or "num_rows_total" not in payload:
        raise RuntimeError(f"Hugging Face dataset viewer returned an unexpected response for {task!r}: {str(payload)[:250]}")
    if payload.get("partial"):
        raise RuntimeError("Dataset viewer returned partial results; cannot sample fairly by task")
    return payload


def download_image(session: requests.Session, data: Any, name: str) -> Image.Image:
    if not isinstance(data, dict) or not isinstance(data.get("src"), str):
        raise ValueError(f"Missing image URL for {name}")
    source = data["src"]
    url = urlsplit(source)
    if url.scheme != "https" or url.hostname not in {
        "datasets-server.huggingface.co", "huggingface.co", "cdn-lfs.huggingface.co",
    }:
        raise ValueError(f"Unexpected {name} image host: {url.hostname}")
    response = session.get(source, timeout=90)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as image:
        return image.copy()


def clean_mask(image: Image.Image, *, min_area: int, max_area_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return original edit mask and largest cleaned connected component.

    We retain the source edit mask for image-level supervision; select a single
    connected region for deriving one-object geometry.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    # CompBench supplies a binary mask; threshold ignores PNG antialias values.
    full = gray >= 128
    frac = full.mean()
    if full.sum() < min_area or frac > max_area_fraction:
        raise ValueError(f"Mask area is unsuitable: {int(full.sum())} px ({frac:.1%} of image)")
    labeled, count = ndi.label(full, structure=np.ones((3, 3), dtype=np.uint8))
    if not count:
        raise ValueError("Empty edit mask")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = labeled == int(np.argmax(sizes))
    # Fill internal holes before skeletonizing. Do not invent new disconnected areas.
    region = ndi.binary_fill_holes(largest)
    if region.sum() < min_area:
        raise ValueError("Largest mask component is too small")
    return full, region


def mask_point(region: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(region)
    cx, cy = float(xx.mean()), float(yy.mean())
    # Centroid of a curved or hollow object may lie outside the object.
    # Snap to the closest foreground pixel to keep the point valid.
    i = int(np.argmin((xx - cx) ** 2 + (yy - cy) ** 2))
    return float(xx[i]), float(yy[i])


def mask_box(region: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(region)
    return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)


def _skeleton_longest_path(skeleton: np.ndarray) -> list[tuple[float, float]]:
    """Approximate longest geodesic centerline along the skeleton's largest component.

    Uses 8-connectivity, two Dijkstra sweeps, and Euclidean edge costs. This is
    a practical main-axis approximation, not the complete medial-axis graph.
    """
    labeled, count = ndi.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    if not count:
        raise ValueError("Skeleton is empty")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    coords = np.argwhere(labeled == int(np.argmax(counts)))  # [y,x]
    if len(coords) == 1:
        return [(float(coords[0, 1]), float(coords[0, 0]))]
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(coords)}
    links: list[list[tuple[int, float]]] = [[] for _ in range(len(coords))]
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nxt = index.get((int(y + dy), int(x + dx)))
                if nxt is not None:
                    links[i].append((nxt, 2 ** 0.5 if dy and dx else 1.0))

    def traverse(start: int) -> tuple[int, list[int]]:
        distances = [float("inf")] * len(coords)
        parents = [-1] * len(coords)
        distances[start] = 0.0
        heap = [(0.0, start)]
        while heap:
            d, i = heapq.heappop(heap)
            if d > distances[i]:
                continue
            for j, cost in links[i]:
                nd = d + cost
                if nd < distances[j] - 1e-9:
                    distances[j] = nd
                    parents[j] = i
                    heapq.heappush(heap, (nd, j))
        endpoints = [i for i, adjacent in enumerate(links) if len(adjacent) == 1]
        candidates = endpoints or list(range(len(coords)))
        farthest = max(candidates, key=lambda i: (distances[i], -i))
        return farthest, parents

    endpoint = next((i for i, neighbors in enumerate(links) if len(neighbors) == 1), 0)
    source, _ = traverse(endpoint)
    finish, parents = traverse(source)
    route = [finish]
    while route[-1] != source and parents[route[-1]] != -1:
        route.append(parents[route[-1]])
    route.reverse()
    return [(float(coords[i, 1]), float(coords[i, 0])) for i in route]


def centerline_scribble(region: np.ndarray, max_points: int = 36) -> list[list[float]]:
    skel = skeletonize(region)
    path = np.asarray(_skeleton_longest_path(skel), dtype=np.float64)
    if len(path) < 2:
        return [[round(float(path[0, 0]), 1), round(float(path[0, 1]), 1)]]
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arclength = np.r_[0.0, np.cumsum(lengths)]
    n = min(max_points, max(2, int(arclength[-1] / 2) + 1))
    distances = np.linspace(0, arclength[-1], n)
    xs = np.interp(distances, arclength, path[:, 0])
    ys = np.interp(distances, arclength, path[:, 1])
    return [[round(float(x), 1), round(float(y), 1)] for x, y in zip(xs, ys)]


def _extract_after(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    segment = match.group(1).strip()
    # Stop at a spatial relation or a second instruction clause.
    segment = re.split(r"\s+(?:in|into|on|at|from|to|near|next\s+to|beside|behind|under|above|between|while|and|so\s+that|which|where|wearing|holding|carrying)\b|[,.;!?]", segment, maxsplit=1, flags=re.IGNORECASE)[0]
    segment = re.sub(r"^(?:a|an|the|another|one|some)\s+", "", segment, flags=re.IGNORECASE)
    segment = segment.strip(" \"'.,;:!?-")
    if not segment or segment.lower() in {"one", "ones", "it", "them", "something", "object", "thing"} or segment.lower().endswith((" one", " ones")):
        return None
    words = segment.split()
    # Reject ambiguous long spans rather than supervising arbitrary words.
    if len(words) > 6:
        return None
    return segment


def extract_object_phrase(instruction: str, task: str) -> tuple[str | None, str]:
    """Conservative non-LLM candidate extraction; users should verify the output.

    `object_phrase` MUST be a literal substring of instruction for both model
    tokenizers. For replacement prefer the new object after 'with' / 'into'.
    """
    text = instruction.strip()
    if task == "add":
        patterns = [r"\b(?:add|insert|place|put|draw|create|include|introduce)\s+(.+)$"]
    elif task == "remove":
        patterns = [r"\b(?:remove|delete|erase|eliminate|take\s+away|get\s+rid\s+of)\s+(.+)$"]
    else:
        patterns = [r"\b(?:with|into)\s+(.+)$", r"\b(?:replace|swap|change)\s+(.+)$"]
    for pattern in patterns:
        phrase = _extract_after(text, pattern)
        if phrase is not None and re.search(re.escape(phrase), text, re.I):
            return phrase, "heuristic"
    return None, "unresolved"


def make_preview(source: Image.Image, region: np.ndarray, center: tuple[float, float],
                 box: tuple[int, int, int, int], path: list[list[float]]) -> Image.Image:
    output = source.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle(box, outline=(255, 190, 20), width=3)
    px, py = center
    draw.ellipse((px-5, py-5, px+5, py+5), fill=(0, 235, 70), outline="black", width=1)
    if len(path) > 1:
        draw.line([tuple(p) for p in path], fill=(255, 35, 55), width=5, joint="curve")
    elif path:
        x, y = path[0]
        draw.ellipse((x-4, y-4, x+4, y+4), fill=(255, 35, 55))
    return output


def export_sample(session: requests.Session, row: dict, task: str, row_idx: int,
                  root: Path, visuals: tuple[str, ...], args: argparse.Namespace) -> list[dict]:
    source = download_image(session, row.get("input_image"), "input_image").convert("RGB")
    target = download_image(session, row.get("edited_image"), "edited_image").convert("RGB")
    mask_img = download_image(session, row.get("mask"), "mask")
    if source.size != target.size or mask_img.size != source.size:
        raise ValueError("Input, edited image and mask dimensions disagree")
    full, region = clean_mask(mask_img, min_area=args.min_mask_pixels,
                              max_area_fraction=args.max_mask_fraction)
    instruction = str(row.get("instruction") or "").strip()
    phrase, phrase_status = extract_object_phrase(instruction, task)
    if not phrase:
        raise ValueError(f"Could not safely identify target phrase in instruction {instruction!r}")
    if args.prompt_override:
        # Deliberate opt-in override: still require the phrase to appear in the prompt.
        phrase = args.prompt_override
        if phrase.lower() not in instruction.lower():
            raise ValueError(f"Override phrase {phrase!r} absent from instruction")
        phrase_status = "override"

    center = mask_point(region)
    box = mask_box(region)
    scribble_path = centerline_scribble(region, args.scribble_points)
    ident = f"{task}_{row_idx:05d}"
    task_dir = root / task
    (task_dir / "images").mkdir(parents=True, exist_ok=True)
    (task_dir / "masks").mkdir(parents=True, exist_ok=True)
    (task_dir / "previews").mkdir(parents=True, exist_ok=True)
    source.save(task_dir / "images" / f"{ident}_input.png")
    target.save(task_dir / "images" / f"{ident}_target.png")
    Image.fromarray((full * 255).astype(np.uint8), "L").save(task_dir / "masks" / f"{ident}_edit.png")
    Image.fromarray((region * 255).astype(np.uint8), "L").save(task_dir / "masks" / f"{ident}_region.png")
    make_preview(source, region, center, box, scribble_path).save(task_dir / "previews" / f"{ident}.png")

    annotations = {
        "point": {"type": "point", "x": center[0], "y": center[1]},
        "box": {"type": "box", "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]},
        "scribble": {"type": "scribble", "path": scribble_path, "width": args.scribble_width},
    }
    data = []
    for visual_type in visuals:
        data.append({
            "id": f"{ident}_{visual_type}",
            "input_image": f"images/{ident}_input.png",
            "target_image": f"images/{ident}_target.png",
            "edit_mask": f"masks/{ident}_edit.png",
            "prompt": instruction,
            "object_phrase": phrase,
            "visual": annotations[visual_type],
            # Additional metadata is ignored by the existing EditingDataset.
            "metadata": {
                "dataset": REPO, "source_split": "train", "row_idx": row_idx,
                "task": task, "visual_type": visual_type,
                "object_phrase_source": phrase_status,
                "caption": row.get("caption", ""),
                "image_path": row.get("image_path", ""),
                "region_mask": f"masks/{ident}_region.png",
            },
        })
    return data


def prepare(args: argparse.Namespace, session: requests.Session | None = None) -> dict[str, Any]:
    session = session or session_with_retries()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    combined: list[dict[str, Any]] = []
    report: dict[str, Any] = {"source": REPO, "seed": args.seed, "tasks": {}, "skipped": []}

    for task in args.tasks:
        page = fetch_filtered(session, task, offset=0)
        count = int(page["num_rows_total"])
        if count <= 0:
            raise RuntimeError(f"No records for {task!r}; check dataset viewer accessibility")
        offsets = list(range(count))
        rng.shuffle(offsets)
        samples: list[dict] = []
        max_attempts = min(count, max(args.per_task * args.max_attempt_factor, args.per_task))
        for offset in offsets[:max_attempts]:
            if len(samples) >= args.per_task * len(args.visuals):
                break
            try:
                if offset == 0:
                    fetched = page
                else:
                    fetched = fetch_filtered(session, task, offset=offset)
                entries = fetched.get("rows") or []
                if not entries:
                    raise ValueError("Empty result row")
                row = entries[0]["row"]
                if row.get("task") != task:
                    raise ValueError("Dataset filter returned wrong task")
                raw_index = int(entries[0].get("row_idx", offset))
                new_samples = export_sample(session, row, task, raw_index, root, args.visuals, args)
                samples.extend(new_samples)
                print(f"[{task}] {len(samples)//len(args.visuals)}/{args.per_task} (row {raw_index})", flush=True)
            except (ValueError, requests.RequestException, OSError) as exc:
                report["skipped"].append({"task": task, "filter_offset": offset, "reason": str(exc)})
                print(f"[{task}] skipping offset {offset}: {exc}", file=sys.stderr)
        task_dir = root / task
        task_dir.mkdir(exist_ok=True)
        (task_dir / "manifest.json").write_text(json.dumps({"samples": samples}, indent=2) + "\n")
        report["tasks"][task] = {"total_available": count,
                                  "source_pairs_exported": len(samples) // len(args.visuals),
                                  "training_entries": len(samples)}
        for row in samples:
            copy = dict(row)
            for field in ("input_image", "target_image", "edit_mask"):
                copy[field] = f"{task}/{copy[field]}"
            copy["metadata"] = dict(copy["metadata"])
            copy["metadata"]["region_mask"] = f"{task}/{copy['metadata']['region_mask']}"
            combined.append(copy)
        if len(samples) < args.per_task * len(args.visuals):
            print(f"Warning: {task} exported only {len(samples)//len(args.visuals)} / {args.per_task} requested pairs", file=sys.stderr)
    (root / "manifest.json").write_text(json.dumps({"samples": combined}, indent=2) + "\n")
    (root / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Done: {len(combined)} training entries → {root}")
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/compbench-vpg"))
    parser.add_argument("--per-task", type=int, default=5, help="Unique source/GT image pairs per task")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--visuals", nargs="+", choices=VISUALS, default=list(VISUALS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scribble-points", type=int, default=36)
    parser.add_argument("--scribble-width", type=float, default=6.0)
    parser.add_argument("--min-mask-pixels", type=int, default=48)
    parser.add_argument("--max-mask-fraction", type=float, default=0.85)
    parser.add_argument("--max-attempt-factor", type=int, default=12,
                        help="Maximum sampled candidates per requested example to handle invalid annotations")
    parser.add_argument("--prompt-override", type=str, default=None,
                        help="Optional object phrase override, only if it occurs verbatim in every instruction")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.per_task < 1 or args.scribble_points < 1 or args.scribble_width <= 0:
        parser.error("per-task, scribble-points and scribble-width must be positive")
    if args.max_attempt_factor < 1 or args.min_mask_pixels < 1 or not 0 < args.max_mask_fraction <= 1:
        parser.error("Invalid sampling/mask settings")
    prepare(args)


if __name__ == "__main__":
    main()
