"""Tier 0: SCI-V21-BACKGROUND design record equals its derivation from sealed results.

No simulation. The verdict label must follow the criteria: the design passes only when every
criterion is SUPPORTED. A scientific UNRESOLVED keeps pytest green.
"""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "v21_background_design.json"


def _derive():
    return runpy.run_path(str(ROOT / "scripts" / "v21_background_design.py"), run_name="bkg_design")["derive"]()


def test_record_equals_derivation():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    assert rec["derived"] == _derive()


def test_verdict_label_follows_criteria():
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    all_supported = all(v.startswith("SUPPORTED") for v in rec["criteria"].values())
    assert rec["verdict"] in ("BKG_DESIGN_PASS", "BKG_DESIGN_UNRESOLVED")
    assert (rec["verdict"] == "BKG_DESIGN_" + "PASS") == all_supported


def test_key_derived_facts():
    d = _derive()
    mid = d["native_noise_v21"]["mid"]
    assert mid["rate_max"]["PV"] == 0.0 and d["fi_battery_b1"]["PV"]["onset_bin"]
    assert all(9.5 <= mid["rate_min"][c] and mid["rate_max"][c] <= 11.0 for c in ("E", "SST", "VIP"))
    b = d["normalization"]["B_gain_normalized"]
    assert b["1.5"]["E"]["rate_max"] < d["bands_hz"]["E"][1] < d["shot_v21b"]["mid_sm2p0"]["rate_max"]["E"]
    assert b["1.5"]["PV"]["evidence"] == "UNSUPPORTED"
    assert d["normalization"]["A_tonic_relative"]["1.5"]["VIP"]["rate_max"] > d["bands_hz"]["VIP"][1]
