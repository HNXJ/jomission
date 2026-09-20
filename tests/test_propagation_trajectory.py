"""SCI-PROP-TRAJECTORY: matched ON/OFF trajectory divergence with the g=0 null.

Read-only analysis of the six sealed WS-PROP-1 cells. No construction, no
simulation. The spec (results/propagation_trajectory_spec.json) predeclares
the divergence, the null, the threshold, the onset rule and the four verdicts;
this test executes them exactly once and writes results/propagation_trajectory.json.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.load(open(ROOT / "results" / "propagation_trajectory_spec.json"))
BASE = "results/whole_system_causal_g"
OUT = ROOT / "results" / "propagation_trajectory.json"

AREAS = ("V1_1", "V1_2", "V4_1", "V4_2", "FEF", "PFC")
DOWNSTREAM = ("V1_2", "V4_1", "V4_2", "FEF", "PFC")
FLOOR = 1e-12
SUSTAIN = 5


def load(name):
    z = np.load(ROOT / f"{BASE}{name}_y.npz")
    groups = [str(g) for g in z["series_groups"]]
    return {g: np.asarray(z["series_rate_hz"][i], dtype=np.float64) for i, g in enumerate(groups)}


def divergence(area, run_a, run_b):
    gs = [g for g in run_a if g.split(".")[0] == area]
    return np.mean([np.abs(run_a[g] - run_b[g]) for g in gs], axis=0)


def first_onset(d, threshold):
    over = d > threshold
    if over.size < SUSTAIN or not over.any():
        return None
    run = np.convolve(over.astype(int), np.ones(SUSTAIN, dtype=int), mode="valid")
    hit = np.flatnonzero(run == SUSTAIN)
    return None if not hit.size else float(hit[0] * 10.0)


def test_propagation_trajectory():
    runs = {
        n: load(n)
        for n in (
            "g500_on_s0",
            "g500_off_s0",
            "g000_on_s0",
            "g000_off_s0",
            "g000_on_s1",
            "g000_off_s1",
        )
    }
    d_test = {a: divergence(a, runs["g500_on_s0"], runs["g500_off_s0"]) for a in AREAS}
    d_null0 = {a: divergence(a, runs["g000_on_s0"], runs["g000_off_s0"]) for a in AREAS}
    d_null1 = {a: divergence(a, runs["g000_on_s1"], runs["g000_off_s1"]) for a in AREAS}

    threshold = float(max(d_null1[a].max() for a in DOWNSTREAM)) + FLOOR
    null_ok = all(
        bool((d_null0[a] <= threshold).all()) and bool((d_null1[a] <= threshold).all())
        for a in DOWNSTREAM
    )
    v1_detects = bool((d_test["V1_1"] > threshold).any())
    downstream = {}
    for a in DOWNSTREAM:
        peak = float(d_test[a].max())
        downstream[a] = {
            "peak": peak,
            "detects": bool(peak > threshold),
            "onset_ms": first_onset(d_test[a], threshold),
        }
    if not v1_detects:
        verdict, boundary = "PROPAGATION_V1_UNDETECTED", "V1_1 shows no divergence in the test pair"
    elif not null_ok:
        verdict, boundary = (
            "PROPAGATION_NULL_FAILED",
            "a downstream g=0 null exceeds the held-out threshold",
        )
    elif not any(v["detects"] for v in downstream.values()):
        verdict, boundary = (
            "PROPAGATION_ABSENT_AT_G05",
            "controls pass; nothing downstream detects at g=0.5",
        )
    else:
        ordered = sorted(
            ((v["onset_ms"], a) for a, v in downstream.items() if v["onset_ms"] is not None)
        )
        verdict = "PROPAGATION_DETECTED_ORDERED"
        boundary = "onset order (ms, area): " + ", ".join(f"{a} {t:.0f}" for t, a in ordered)

    rec = {
        "spec": "results/propagation_trajectory_spec.json",
        "lineage": "SCI-PROP-TRAJECTORY",
        "estimator": SPEC["estimator"],
        "threshold": threshold,
        "v1_1_peak": float(d_test["V1_1"].max()),
        "null_ok": null_ok,
        "downstream": downstream,
        "verdict": verdict,
        "fail_boundary": boundary,
    }
    OUT.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    assert verdict in SPEC["verdicts"], verdict
    assert OUT.is_file()
