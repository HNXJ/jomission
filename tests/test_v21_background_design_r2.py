"""Tier 0: analytic BKG regeneration record equals its derivation; bifurcation algebra known answers.

No simulation. A scientific UNRESOLVED keeps pytest green.
"""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "v21_background_design_r2.json"


def _mod():
    return runpy.run_path(str(ROOT / "scripts" / "v21_background_design_r2.py"), run_name="bkg_r2")


def test_record_equals_derivation():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    assert rec["derived"] == _mod()["derive"]()


def test_bifurcation_known_answers():
    m = _mod()
    ons = m["onsets"]()
    # saddle-node of 0.04 v^2 + (5 - b) v + 140 + I: I = (5 - b)^2 / 0.16 - 140
    assert abs(ons["PV"]["saddle_node_I"] - 4.0) < 1e-9 and abs(ons["VIP"]["saddle_node_I"] - 22.5625) < 1e-9
    # continuous Hopf: trace 0 at v = (a - 5) / 0.08; I from the rest equation
    for c, (a, b, _, _) in m["CELLS"].items():
        v = (a - 5.0) / 0.08
        i_h = -(0.04 * v * v + (5.0 - b) * v + 140.0)
        if ons[c]["instability_kind"] == "Hopf":
            assert abs(ons[c]["continuous_instability_I"] - i_h) < 1e-4, c


def test_verdict_label_follows_criteria():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    all_supported = all(v.startswith("SUPPORTED") for v in rec["criteria"].values())
    assert (rec["verdict"] == "BKG_DESIGN_" + "PASS") == all_supported
