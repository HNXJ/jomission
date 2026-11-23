"""OPEN-69: the interaction compare refuses to overwrite its tracked result unless --overwrite is given.

Fixtures are tmp_path copies of the script's ROOT; no simulation and no repo file is written.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/whole_system_ei_interaction_compare.py"
_spec = importlib.util.spec_from_file_location("whole_system_ei_interaction_compare", _SCRIPT)
cmp_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp_)


def _root_with_result(tmp_path: Path, exists: bool) -> Path:
    (tmp_path / "results").mkdir()
    target = tmp_path / "results" / "whole_system_ei_interaction.json"
    if exists:
        target.write_text("{}\n", encoding="utf-8")
    assert target.exists() is exists, "fixture must build the case it is named after"
    return tmp_path


def test_refuses_to_overwrite_existing_result_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp_, "ROOT", _root_with_result(tmp_path, exists=True))
    with pytest.raises(SystemExit, match="--overwrite"):
        cmp_.main([])
    assert (tmp_path / "results" / "whole_system_ei_interaction.json").read_text(encoding="utf-8") == "{}\n"


def test_flag_passes_the_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp_, "ROOT", _root_with_result(tmp_path, exists=True))
    # No cell receipts exist in the fixture, so the run stops at input loading, past the guard.
    with pytest.raises(ValueError, match="expected one baseline cell receipt"):
        cmp_.main(["--overwrite"])


def test_absent_result_is_not_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp_, "ROOT", _root_with_result(tmp_path, exists=False))
    with pytest.raises(ValueError, match="expected one baseline cell receipt"):
        cmp_.main([])
