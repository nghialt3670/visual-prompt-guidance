"""Offline tests for selective CompBench download and annotation conversion."""
import importlib.util
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "prepare_compbench.py"
spec = importlib.util.spec_from_file_location("prepare_compbench", SCRIPT)
comp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comp)


def test_geometry_rectangle():
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[12:50, 29:65] = 255
    full, region = comp.clean_mask(Image.fromarray(mask), min_area=12, max_area_fraction=.85)
    assert np.array_equal(full, region)
    assert comp.mask_box(region) == (29, 12, 65, 50)
    x, y = comp.mask_point(region)
    assert region[int(y), int(x)]
    pts = comp.centerline_scribble(region, max_points=20)
    assert 1 <= len(pts) <= 20
    assert all(region[int(round(y)), int(round(x))] for x, y in pts)


def test_geometry_curved_mask():
    arr = np.zeros((110, 120), dtype=np.uint8)
    from PIL import ImageDraw
    canvas = Image.fromarray(arr)
    ImageDraw.Draw(canvas).line([(10, 10), (10, 80), (50, 80), (100, 45)], width=17, fill=255)
    _, region = comp.clean_mask(canvas, min_area=12, max_area_fraction=.85)
    pts = comp.centerline_scribble(region, max_points=32)
    assert len(pts) >= 10
    assert all(region[int(round(y)), int(round(x))] for x, y in pts)
    assert (max(x for x, y in pts) - min(x for x, y in pts)) > 55
    assert (max(y for x, y in pts) - min(y for x, y in pts)) > 40


@pytest.mark.parametrize("text,task,expected", [
    ("Add a red dog to the left", "add", "red dog"),
    ("Remove the brown cat from the sofa", "remove", "brown cat"),
    ("Replace the dog with a black cat", "replace", "black cat"),
    ("The scene is sunny", "add", None),
])
def test_object_phrase(text, task, expected):
    phrase, _ = comp.extract_object_phrase(text, task)
    assert phrase == expected


def _png(image):
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class FakeResponse:
    def __init__(self, *, body=None, image=None):
        self.body, self.content = body, image
    def json(self):
        return self.body
    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        bg = Image.new("RGB", (96, 96), (230, 230, 230))
        gt = bg.copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(gt)
        draw.ellipse((20, 10, 70, 86), fill=(10, 200, 80))
        mask = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(mask).ellipse((20, 10, 70, 86), fill=255)
        self.images = {"input": _png(bg), "edited": _png(gt), "mask": _png(mask)}
    def get(self, url, params=None, timeout=None):
        if url == comp.DATASET_URL:
            task = params["where"].split("'")[1]
            offset = int(params["offset"])
            row = {
                "task": task,
                "image_path": f"example_{offset}",
                "input_image": {"src": "https://datasets-server.huggingface.co/input"},
                "edited_image": {"src": "https://datasets-server.huggingface.co/edited"},
                "mask": {"src": "https://datasets-server.huggingface.co/mask"},
                "instruction": {
                    "add": "Add a green ball to the table",
                    "remove": "Remove a green ball from the table",
                    "replace": "Replace a green ball with a red ball",
                }[task],
                "caption": "A ball on a table",
            }
            return FakeResponse(body={"num_rows_total": 5, "rows": [{"row_idx": offset, "row": row}], "partial": False})
        return FakeResponse(image=self.images[url.rsplit("/", 1)[-1]])


def test_end_to_end_mock_compbench_conversion(tmp_path):
    args = SimpleNamespace(
        out=tmp_path / "data", tasks=["add", "remove", "replace"],
        per_task=2, visuals=("point", "box", "scribble"), seed=55,
        scribble_points=22, scribble_width=6,
        min_mask_pixels=16, max_mask_fraction=.85,
        max_attempt_factor=5, prompt_override=None,
    )
    report = comp.prepare(args, session=FakeSession())
    assert sum(x["source_pairs_exported"] for x in report["tasks"].values()) == 6
    from visual_prompt_guidance.training import EditingDataset
    with EditingDataset(args.out) as ds:
        assert len(ds) == 18
        for idx in range(len(ds)):
            ex = ds[idx]
            assert ex.request.data["image"].size == ex.target_image.size
            assert ex.edit_mask is not None
            projected = ex.visual.project(__import__("visual_prompt_guidance").Space((96, 96), (6, 6)))
            assert projected.density.sum() > 0
    for task in args.tasks:
        with EditingDataset(args.out / task) as ds:
            assert len(ds) == 6
        assert len(list((args.out / task / "previews").glob("*.png"))) == 2
