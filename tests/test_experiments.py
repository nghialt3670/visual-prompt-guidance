import csv
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

from visual_prompt_guidance.experiments import create_run_dir, save_run_config, save_training_reports

# The Gradio app and its settings are deliberately outside the installable core.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from ui_settings import read_store, save_profile, normalize_component_value, profile_settings


def test_run_dirs_unique_and_separated(tmp_path):
    a = create_run_dir("train", tmp_path)
    b = create_run_dir("train", tmp_path)
    c = create_run_dir("generate", tmp_path)
    assert a != b and a.parent == b.parent == tmp_path / "train"
    assert c.parent == tmp_path / "generate"
    config = save_run_config(a, {"steps": 50, "layers": ("D0", "S1")})
    assert json.loads(config.read_text())["layers"] == ["D0", "S1"]
    assert (a / "config.json").is_file()


def test_training_reports(tmp_path):
    history = [{"epoch": 1, "loss": 0.9}, {"epoch": 1, "loss": 0.4}, {"epoch": 2, "loss": 0.2}]
    weights, plot = save_training_reports(tmp_path, history, ["D0:H1", "D0:H0"], [0.1, 0.9])
    assert weights.name == "head_weights.csv" and plot.name == "loss_curve.png"
    with weights.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["head"] == "D0:H0"
    assert float(rows[0]["weight"]) == 0.9
    with Image.open(plot) as view:
        assert view.size == (1000, 520)


def test_saved_generate_and_train_settings_survive_restart(tmp_path):
    settings_path = tmp_path / "ui_settings.json"
    assert read_store(settings_path)["active"] == "generate"
    save_profile("generate", {"steps": 50, "model": "FLUX.2 Klein", "selected_heads": ["D0:H01"]}, path=settings_path)
    save_profile("train", {"steps": 4, "learning_rate": 0.003, "training_epochs": 7}, path=settings_path)
    assert read_store(settings_path)["active"] == "train"
    assert profile_settings("generate", path=settings_path)["steps"] == 50
    assert profile_settings("train", path=settings_path)["steps"] == 4
    save_profile("generate", {"steps": 60}, path=settings_path)
    assert profile_settings("train", path=settings_path)["training_epochs"] == 7
    assert profile_settings("generate", path=settings_path)["steps"] == 60
    assert read_store(settings_path)["active"] == "generate"


def test_preferences_do_not_overwrite_corrupt_json(tmp_path):
    file = tmp_path / "ui_settings.json"
    file.write_text("not json")
    with pytest.raises(ValueError, match="Cannot load"):
        save_profile("train", {"steps": 12}, path=file)
    assert file.read_text() == "not json"


def test_ui_initializes_from_last_saved_profile(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    path = tmp_path / "app.json"
    save_profile("train", {
        "model": "FLUX.2 Klein", "head_mode": "Learned", "steps": 4,
        "training_epochs": 9, "learning_rate": 0.004,
        "selected_layers": ["D2"], "selected_heads": ["D2:H03"],
        "propagation_layers": ["D2"],
    }, path=path)
    monkeypatch.setenv("VPG_UI_CONFIG", str(path))
    from gradio_app import build_demo
    demo = build_demo()
    settings = {component.get("props", {}).get("label"): component["props"]
                for component in demo.config["components"] if "props" in component}
    assert settings["Denoising steps"]["value"] == 4
    assert settings["Epochs"]["value"] == 9
    assert settings["Learning rate"]["value"] == pytest.approx(0.004)
    assert settings["Guidance attention layers"]["value"] == ["D2"]
    assert settings["Include specific heads (optional)"]["value"] == ["D2:H03"]


def test_train_and_generate_run_handlers_create_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import numpy as np
    from gradio_app import train_heads, run
    import gradio_app
    from test_learning import DifferentiableTinyAdapter

    class MockBackend(DifferentiableTinyAdapter):
        def decode(self, state):
            data = (self.decode_tensor(state)[0].detach().clamp(0, 1) * 255).byte()
            return Image.fromarray(data.permute(1, 2, 0).cpu().numpy())

    monkeypatch.setattr(gradio_app, "backend", lambda _: MockBackend())
    monkeypatch.chdir(tmp_path)
    Image.fromarray(np.full((16, 16, 3), 110, dtype=np.uint8)).save("input.png")
    Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)).save("target.png")
    (tmp_path / "manifest.json").write_text(json.dumps([{
        "input_image": "input.png", "target_image": "target.png", "prompt": "Add a dog",
        "object_phrase": "dog", "visual": {"type": "point", "x": 4, "y": 4},
    }]))
    checkpoint, message = train_heads(
        None, str(tmp_path), "InstructPix2Pix", ["coverage"],
        1, 1, 1, 1, 1, 1, 2, 1, 0.02, 42, 128,
        "base", 0.8, 4.0, 7.5, 1.5,
        None, None, 1.0,
        False, None, 0, 1, 2, 0.2, 0.3,
        1, .02, "adamw", 0, "l1", 1, 1, 0, 1, "unused",
        progress=lambda *args, **kwargs: None,
    )
    folder = Path(checkpoint).parent
    assert folder.parent == Path("outputs/train")
    for filename in ("config.json", "learned_heads.json", "history.json",
                     "loss_curve.png", "head_weights.csv"):
        assert (folder / filename).is_file(), filename
    assert json.loads((folder / "config.json").read_text())["train_config"]["epochs"] == 1
    assert read_store()["train"]["learned_checkpoint"] == checkpoint

    bg = Image.open("input.png")
    output, info = run(
        {"background": bg, "layers": []}, [(4, 4)], "Point", "Add a dog", "dog",
        ["coverage"], 1, 1, 1, 1, 1, 1,
        "InstructPix2Pix", 2, 1, .02, 42, 128,
        "base", .8, 4.0, 7.5, 1.5,
        "Uniform", None, None, None, .01, .5, .1,
        "", False, None, 0, 1, 2, .2, .3, False,
    )
    assert isinstance(output, Image.Image)
    gen_folders = list((tmp_path / "outputs/generate").iterdir())
    assert len(gen_folders) == 1
    for filename in ("input.png", "config.json", "result.png", "steps.json", "heads.csv"):
        assert (gen_folders[0] / filename).exists(), filename
