"""Tier 0: the PV state-predecessor record equals its read-only derivation. No simulation."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "v21_pv_state_predecessor.json"


def _mod():
    return runpy.run_path(str(ROOT / "scripts" / "v21_pv_state_predecessor.py"), run_name="pv_pred")


def test_record_equals_derivation():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    assert rec["derived"] == _mod()["derive"]()
    assert rec["verdict"] == rec["derived"]["label"]


def test_first_divergence_known_answers():
    fd = _mod()["first_divergence"]
    assert fd([14.0, 14.0, 14.0], [14.5, 12.0, 0.0]) == 2
    assert fd([0.5, 0.5], [1.4, 0.0]) is None
