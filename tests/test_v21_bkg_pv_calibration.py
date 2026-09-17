"""SCI-V21-BKG-PV-CALIBRATION: isolated PV F-I on I in [2.8, 3.6] under native noise.

Spec (committed before the procedure runs): results/v21_bkg_pv_calibration_spec.json.
Tier 3: writes results/v21_bkg_pv_calibration.json. The driver passes when the procedure ran;
the calibration label is data.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from jomission.qualification import intrinsic as ih
from jomission.qualification.cmin import build_cmin

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21_bkg_pv_calibration_spec.json"
OUT = ROOT / "results" / "v21_bkg_pv_calibration.json"
LO, HI, TARGET = 2.8, 3.6, 10.0
WIDTH, SEED, STEPS, N_CHUNK, WINDOW_S = 0.0125, 32100, 50000, 4, 5.0


def _pv_params():
    model = build_cmin()
    cls = np.array([str(r["cell_type"]) for r in model.neuron_table()])
    em = model.params["emitter"]
    return {k: np.asarray(getattr(em, k))[cls == "PV"].astype(np.float64) for k in "abcd"}


def _rate(params, current):
    p = dict(params, I0=float(current))
    m = p["a"].shape[0]
    zeros = [np.zeros((STEPS, m))] * N_CHUNK
    wins = ih.simulate_population(p, zeros, {}, N_CHUNK, STEPS, SEED)
    assert wins is not None, "non-finite state"
    per_window = [float(w.sum() / (m * WINDOW_S)) for w in wins]
    return float(np.mean(per_window)), per_window


def _blob(rel):
    return subprocess.run(["git", "hash-object", str(ROOT / rel)], capture_output=True, text=True).stdout.strip()


def test_spec_pins_match():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for rel, pin in spec["frozen"]["code"].items():
        assert _blob(rel) == pin.split()[0], rel


def test_v21_bkg_pv_calibration():
    params = _pv_params()
    homogeneous = {k: [float(params[k].min()), float(params[k].max())] for k in "abcd"}
    evals = []

    def ev(current, role):
        r, pw = _rate(params, current)
        evals.append({"I": round(float(current), 6), "rate_hz": r, "per_window_hz": pw, "role": role})
        return r

    r_lo, r_hi = ev(LO, "endpoint"), ev(HI, "endpoint")
    res = {"spec": "results/v21_bkg_pv_calibration_spec.json", "n_cells": int(params["a"].shape[0]),
           "cell_params_min_max": homogeneous, "evaluations": evals}
    if not (r_lo < TARGET <= r_hi):
        res["label"] = "PV_CALIBRATION_FAIL"
        res["reason"] = f"10 Hz not bracketed in [{LO}, {HI}]: r({LO}) = {r_lo}, r({HI}) = {r_hi}"
    else:
        lo, hi, rl, rh = LO, HI, r_lo, r_hi
        monotone = True
        while hi - lo > WIDTH:
            mid = 0.5 * (lo + hi)
            rm = ev(mid, "bisection")
            if rm < rl - 1.0 or rm > rh + 1.0:
                monotone = False
            if rm < TARGET:
                lo, rl = mid, rm
            else:
                hi, rh = mid, rm
        i10 = lo + (TARGET - rl) * (hi - lo) / (rh - rl)
        slopes = {}
        for off in (0.05, 0.1):
            a, b = max(LO, i10 - off), min(HI, i10 + off)
            slopes[off] = (ev(b, f"slope+{off}") - ev(a, f"slope-{off}")) / (b - a)
        g05, g10 = slopes[0.05], slopes[0.1]
        usable = g05 > 0 and g10 > 0 and 0.5 <= g05 / g10 <= 2.0
        res.update({"bracket": [lo, hi], "bracket_rates_hz": [rl, rh], "I_PV10": i10,
                    "g_05": g05, "g_10": g10, "g_PV": g05, "slope_usable": usable, "monotone": monotone})
        if monotone and (rh - rl) <= 5.0 and usable:
            res["label"] = "PV_CALIBRATION_PASS"
        else:
            res["label"] = "PV_CALIBRATION_UNRESOLVED"
    v_mid = json.loads((ROOT / "results" / "v21_mid.json").read_text())
    v_high = json.loads((ROOT / "results" / "v21_high.json").read_text())
    r28 = next(e["rate_hz"] for e in evals if e["I"] == LO)
    r36 = next(e["rate_hz"] for e in evals if e["I"] == HI)
    res["reported_not_gating"] = {
        "network_V_MID_PV_hz": [w["PV"] for w in v_mid["rates"]], "isolated_r_2p8_hz": r28,
        "diagnostic_reading": ("network-mediated PV suppression at V_MID" if r28 >= 5.0 else
                               "battery_b1 F-I inversion fails near threshold" if r28 < 1.0 else "intermediate"),
        "network_V_HIGH_PV_hz": [w["PV"] for w in v_high["rates"]], "isolated_r_3p6_hz": r36,
        "analytic_saddle_node_I": (5.0 - float(params["b"][0])) ** 2 / 0.16 - 140.0,
    }
    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(res["label"], res.get("reason", ""))
