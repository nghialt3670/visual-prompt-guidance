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


