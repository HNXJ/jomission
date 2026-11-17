"""OPEN-38 stage-1 checker on synthetic runs: planted pass; a flat seed with d <= 0, a ref seed out of band and a
missing run each fail their own gate."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
SPEC = importlib.util.spec_from_file_location("stage1", ROOT / "scripts" / "open38_stage1_check.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
import test_open38_smoke_check as fx  # noqa: E402

EXPORT_FILE = "D:/repos/_work/_temp_jaxfne_351f6048/jaxfne/__init__.py"
REF = {1: 0.84, 2: 0.85, 3: 0.83, 4: 0.86, 5: 0.82}
ARM = {s: -0.72 + 0.01 * s for s in range(1, 6)}


def _plant(d: Path, ref=REF, flat=ARM, rev=ARM, skip=()):
    for arm, vals in (("ref", ref), ("flat", flat), ("rev", rev)):
        for s, r in vals.items():
            t = mod.tag(arm, s)
            if t in skip:
                continue
            fx._write(d, t, r_ab=r, sha=None)
            rec = json.loads((d / f"{t}.json").read_text())
            rec["jaxfne_file"] = EXPORT_FILE
            (d / f"{t}.json").write_text(json.dumps(rec))


def _gates(d: Path):
    return json.loads((d / "s1.json").read_text())["gates"]


def test_planted_pass(tmp_path):
    _plant(tmp_path)
    assert mod.main(str(tmp_path / "s1.json"), tmp_path) == 0
    g = _gates(tmp_path)
    assert g["flat"]["mean"] == pytest.approx(0.84 + 0.72 - 0.03, abs=0.01), "fixture must plant a large flat effect"
    assert g["ref_band"]["all_five_in_band"] is True


def test_each_defect_fails_its_gate(tmp_path):
    _plant(tmp_path, ref={**REF, 2: 0.95}, flat={**ARM, 4: 0.90}, skip=(mod.tag("rev", 5),))
    assert mod.main(str(tmp_path / "s1.json"), tmp_path) == 1
    g = _gates(tmp_path)
    assert not g["ref_band"]["ok"] and g["ref_band"]["in_band"]["2"] is False
    assert not g["flat"]["ok"] and min(g["flat"]["d_s"]) <= 0
    assert g["rev"]["status"].startswith("not evaluable") and not g["runs"]["ok"]
