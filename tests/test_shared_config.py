"""Regressions for CLI/UI sharing, overrides, checkpoints, and head identity."""
from pathlib import Path
import json
import sys
import torch
import pytest
from visual_prompt_guidance.config import (config_path, read_store, save_profile,
                                           resolve_settings, profile_settings)
from visual_prompt_guidance.runtime import build_engine, build_policy, model_options
from visual_prompt_guidance.heads import Learned

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import train_heads as train_cli
import generate as generate_cli


def test_cli_training_loads_same_shared_config_and_only_explicit_overrides(tmp_path):
    path = tmp_path / "project.json"
    save_profile("train", {"model": "FLUX.2 Klein", "dataset_path": "my_dataset",
                           "steps": 30, "guide_steps": 5, "max_side": 512,
                           "selected_layers": ["S0"], "selected_losses": ["coverage", "moment"],
                           "moment_w": 2.0, "use_propagation": True}, path=path)
    args = train_cli.make_parser().parse_args(["--config", str(path), "--epochs", "7"])
    settings = resolve_settings("train", train_cli.settings_from_args(args), path=args.config)
    assert settings["steps"] == 30
    assert settings["guide_steps"] == 5
    assert settings["training_epochs"] == 7
    assert settings["selected_layers"] == ["S0"]
    assert settings["use_propagation"]
    assert model_options(settings)["cfg"] == pytest.approx(4.0)
    assert profile_settings("train", path=path)["training_epochs"] == 1  # overrides not persisted
    # Saving from any host reappears in both hosts.
    save_profile("train", train_cli.settings_from_args(args), path=path)
    assert profile_settings("train", path=path)["training_epochs"] == 7


def test_generate_checkpoint_auto_restores_exact_layer_order(tmp_path):
    labels = ("D1:H00", "D1:H01", "S2:H00")
    learned = Learned(temperature=1.0)
    learned.weights(labels, device=torch.device("cpu"))
    with torch.no_grad():
        learned.logits.copy_(torch.tensor([0.4, -0.3, 1.2]))
    checkpoint = tmp_path / "learned_heads.json"
    learned.save(checkpoint)
    project = tmp_path / "project.json"
    save_profile("generate", {"model": "FLUX.2 Klein", "head_mode": "Uniform",
                              "selected_layers": ["D0"], "selected_heads": ["D0:H00"]}, path=project)
    args = generate_cli.make_parser().parse_args(["--config", str(project), "--checkpoint", str(checkpoint)])
    overrides = generate_cli.settings_from_args(args)
    resolved = resolve_settings("generate", overrides, path=args.config)
    engine = build_engine(resolved, backend=object())
    assert engine.guidance.layers == ("D1", "S2")
    actual = engine.policy.weights(labels, device=torch.device("cpu"))
    assert torch.allclose(actual, torch.softmax(torch.tensor([0.4,-0.3,1.2]), dim=0))
    assert profile_settings("generate", path=project)["head_mode"] == "Uniform"  # non-persistent


def test_generate_fails_fast_on_mismatched_checkpoint_layers(tmp_path, monkeypatch):
    # Isolate from the workspace's configs/project.json, whose saved selected_heads
    # would otherwise trigger the head-ID check before the layer check under test.
    monkeypatch.setenv("VPG_CONFIG", str(tmp_path / "project.json"))
    policy = Learned()
    policy.weights(("D2:H00",), device=torch.device("cpu"))
    checkpoint = tmp_path / "w.json"
    policy.save(checkpoint)
    config = resolve_settings("generate", {"head_mode": "Learned",
                                             "learned_checkpoint": str(checkpoint),
                                             "selected_layers": ["S2"]})
    with pytest.raises(ValueError, match="Checkpoint requires"):
        build_engine(config, backend=object())


def test_ui_and_cli_share_canonical_location(tmp_path, monkeypatch):
    path = tmp_path / "shared.json"
    monkeypatch.setenv("VPG_CONFIG", str(path))
    save_profile("generate", {"steps": 41, "head_mode": "Adaptive"})
    assert read_store()["generate"]["steps"] == 41
    args = generate_cli.make_parser().parse_args([])
    cli_settings = resolve_settings("generate", generate_cli.settings_from_args(args))
    assert cli_settings["steps"] == 41
    from ui_settings import profile_settings as ui_profile
    assert ui_profile("generate")["steps"] == 41


def test_shared_config_migrates_old_ui_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = Path("configs/ui_settings.json")
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({"schema_version": 1, "active": "train",
                                  "generate": {}, "train": {"steps": 19}}))
    assert resolve_settings("train")["steps"] == 19
    save_profile("train", {"max_side": 384})
    assert Path("configs/project.json").is_file()
    assert resolve_settings("train")["steps"] == 19
