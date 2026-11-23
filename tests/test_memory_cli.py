"""Explicit all-head selection must override a saved narrow filter."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import train_heads


def test_all_heads_cli_overrides_selected_heads():
    args = train_heads.make_parser().parse_args(['--backend', 'flux2', '--all-heads', '--layer', 'D2'])
    overrides = train_heads.settings_from_args(args)
    assert overrides['selected_heads'] == []
    assert overrides['selected_layers'] == ['D2']


def test_all_heads_and_individual_head_conflict():
    args = train_heads.make_parser().parse_args(['--all-heads', '--head', 'D2:H00'])
    with pytest.raises(ValueError, match='either --all-heads or --head'):
        train_heads.settings_from_args(args)
