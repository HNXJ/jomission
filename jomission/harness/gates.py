"""Canonical gate evaluation from manifests/gates/*.json (single source of thresholds).

A window summary is a dict:
    rates:              {class: Hz}
    isi_fraction_pooled: float            (historical profile)
    isi_fraction:        {class: float}   (prospective profile)
    rate_cv:             {class: float}
    sync:                float
drift is {class: float} over the analysis windows.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE_DIR = ROOT / "manifests" / "gates"


def load_gate(gate_id: str = "v2_local_operation") -> dict:
    return json.load(open(GATE_DIR / f"{gate_id}.json", encoding="utf-8"))


def _check(spec: dict, windows: list[dict], drift: dict) -> bool:
    k = spec["kind"]
    if k == "class_rate_band":
        lo, hi = spec["rate_hz"]
        return all(lo <= w["rates"][c] <= hi for w in windows for c in spec["classes"])
    if k == "class_rate_band_by_class":
        return all(b[0] <= w["rates"][c] <= b[1] for w in windows for c, b in spec["rate_hz"].items())
    if k == "class_rate_min":
        return all(w["rates"][c] >= spec["rate_hz_min"] for w in windows for c in spec["classes"])
    if k == "isi_fraction_pooled":
        return all(w["isi_fraction_pooled"] >= spec["min_fraction"] for w in windows)
    if k == "isi_fraction_by_class":
        return all(w["isi_fraction"][c] is not None and w["isi_fraction"][c] >= spec["min_fraction"]
                   for w in windows for c in spec["classes"])
    if k == "rate_cv_min":
        return all(w["rate_cv"][c] >= spec["rate_cv_min"] for w in windows for c in spec["classes"])
    if k == "sync_max_exclusive":
        return max(w["sync"] for w in windows) < spec["max"]
    if k == "drift_max":
        return all(drift[c] <= spec["max"] for c in spec["classes"])
    raise ValueError(f"unknown check kind {k!r}")


def evaluate(profile: str, windows: list[dict], drift: dict, gate_id: str = "v2_local_operation") -> dict:
    """Return {check_name: bool} in definition order, plus nothing else."""
    checks = load_gate(gate_id)["profiles"][profile]["checks"]
    return {name: bool(_check(spec, windows, drift)) for name, spec in checks.items()}


def windows_from_v21_result(res: dict) -> list[dict]:
    """Adapter for sealed V2.1/V2.1b cell JSONs (tests/test_v21_operation.py layout)."""
    out = []
    for r, cv, s in zip(res["rates"], res["cv"], res["sync"]):
        out.append({"rates": r, "isi_fraction_pooled": cv["frac_in"],
                    "rate_cv": {"E": cv["E_rate_CV"]}, "sync": s})
    return out
