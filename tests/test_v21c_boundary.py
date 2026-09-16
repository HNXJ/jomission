"""SCI-V21C-BOUNDARY: order of w_spontaneous and w_CV=0.3 for E b-dispersion.

Spec (committed before any output): results/v21c_boundary_spec.json.
Frozen library: jomission/qualification/intrinsic.py and the V2.1c driver
helpers in tests/test_v21c_intrinsic_design.py (imported, not copied).

Fast: spec pins match the working tree.
Slow driver (tier 3): test_v21c_boundary -> results/v21c_boundary.json.
The driver passes when the procedure ran; the scientific verdict is data.
"""

import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from jomission.qualification import intrinsic as ih

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "results" / "v21c_boundary_spec.json"
OUT = ROOT / "results" / "v21c_boundary.json"
A, B0, C, D = 0.02, 0.2, -65.0, 8.0
M, PERM_SEED = 300, 31001
CV_TARGET = 0.3
VERDICTS = {"SPONTANEOUS_BEFORE_CV": "INTRINSIC_HETEROGENEITY_INFEASIBLE",
            "CV_BEFORE_SPONTANEOUS": "INTRINSIC_HETEROGENEITY_FEASIBLE",
            "COINCIDENT": "INTRINSIC_HETEROGENEITY_UNRESOLVED"}


def _design():
    spec = importlib.util.spec_from_file_location("_v21c_design", ROOT / "tests" / "test_v21c_intrinsic_design.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _blob(rel):
    return subprocess.run(["git", "hash-object", str(ROOT / rel)], capture_output=True, text=True).stdout.strip()


def test_spec_pins_match_tree():
    s = json.load(open(SPEC))
    for group in ("code", "data"):
        for rel, pin in s["frozen"][group].items():
            assert _blob(rel) == pin.split()[0], rel
    dm = _design()
    assert (dm.M, dm.PERM_SEED, dm.NOISE_SEED, dm.N_CHUNK, dm.STEPS) == (M, PERM_SEED, 31000, 4, 50000)
    assert dm.P0 == {"a": A, "b": B0, "c": C, "d": D}
    assert ih.DT_MS == 0.1


# ---------------- boundary numerics ----------------

def spontaneous(b, dur_s=10.0):
    """(n,) bool: >=1 spike in the last half of dur_s at I=0, noise-free (intrinsic.py protocol)."""
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    n = b.shape[0]
    r, _ = ih.deterministic_rate_cv(np.full(n, A), b, np.full(n, C), np.full(n, D), 0.0, dur_s=dur_s)
    return r > 0.0


def spontaneous_from(b, v0, dur_s=10.0):
    """Same update as intrinsic.deterministic_rate_cv but initial v = v0, u = b*v0."""
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    v = np.full(b.shape, float(v0))
    u = b * v
    n = int(round(dur_s * 1000.0 / ih.DT_MS))
    cnt = np.zeros(b.shape)
    for t in range(n):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u
        du = A * (b * v - u)
        v = v + ih.DT_MS * dv
        u = u + ih.DT_MS * du
        spk = v >= ih.V_PEAK
        v = np.where(spk, C, v)
        u = np.where(spk, u + D, u)
        if t >= n // 2:
            cnt += spk
    return cnt > 0


def population_b(w):
    return ih.lognormal_dispersion(B0, w, M, PERM_SEED)


def bisect(pred, lo, hi, tol, log):
    """pred(lo) False, pred(hi) True assumed checked by caller; returns (lo, hi) with hi - lo <= tol."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        val = pred(mid)
        log.append({"x": mid, "pred": bool(val)})
        if val:
            hi = mid
        else:
            lo = mid
    return lo, hi


def rest_point_analytics():
    b_sn = 5.0 - math.sqrt(22.4)
    v_h = (A - 5.0) / 0.08
    b_hopf = 5.0 + (0.04 * v_h * v_h + 140.0) / v_h

    def rho(b, dt=0.1):
        disc = (5.0 - b) ** 2 - 22.4
        if disc < 0:
            return math.inf
        v = (-(5.0 - b) - math.sqrt(disc)) / 0.08  # lower root (rest)
        j = np.array([[1 + dt * (0.08 * v + 5.0), -dt], [dt * A * b, 1 - dt * A]])
        return float(max(abs(np.linalg.eigvals(j))))

    lo, hi = 0.2, b_sn - 1e-9
    assert rho(lo) < 1 < rho(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rho(mid) < 1 else (lo, mid)
    return {"saddle_node_b": b_sn, "hopf_b_continuous": b_hopf, "euler_map_unit_radius_b_dt0p1": 0.5 * (lo + hi),
            "rest_root": "lower root of 0.04 v^2 + (5 - b) v + 140 = 0", "evidence_class": "DERIVED"}


def test_v21c_boundary():
    spec = json.load(open(SPEC))
    dm = _design()
    gates, failures = {}, []

    def gate(name, ok):
        gates[name] = bool(ok)
        if not ok:
            failures.append(name)

    # A: b_crit
    s_lo, s_hi = spontaneous([0.260, 0.265])
    gate("A_endpoint_reproduction", (not s_lo) and s_hi)
    a_log = []
    bc_lo, bc_hi = bisect(lambda x: bool(spontaneous(x)[0]), 0.260, 0.265, 1e-6, a_log)
    grid = np.round(np.arange(0.200, 0.4600001, 0.0005), 6)
    fires = spontaneous(grid)
    gate("V1_predicate_monotone_in_b", bool(np.all(fires[grid >= bc_hi]) and not np.any(fires[grid <= bc_lo])))

    # B: w_spontaneous
    S = lambda w: bool(spontaneous(population_b(w)).any())  # noqa: E731
    gate("B_endpoint_reproduction", (not S(0.05)) and S(0.10))
    wgrid = np.round(np.arange(0.05, 0.1000001, 0.0001), 6)
    bmax = np.array([population_b(w).max() for w in wgrid])
    gate("V2_max_b_monotone_in_w", bool(np.all(np.diff(bmax) >= 0.0)))
    b_log = []
    ws_lo, ws_hi = bisect(S, 0.05, 0.10, 1e-4, b_log)
    gate("C1_consistency", population_b(ws_lo).max() < bc_hi and population_b(ws_hi).max() >= bc_lo)
    b_hi = population_b(ws_hi)
    first = int(np.argmax(b_hi))
    first_rate, _ = ih.deterministic_rate_cv([A], [b_hi[first]], [C], [D], 0.0, dur_s=10.0)

    # C: w_CV
    sealed = json.load(open(ROOT / "results" / "v21c_width_b.json"))["widths"]
    sealed_cv = {w: min(x["rate_cv"] for x in sealed[w]["windows"]) for w in sealed}
    evals = {}

    def cv(w):
        arr = dm._homog()
        arr["b"] = population_b(w)
        ws = dm._run(arr)
        val = None if ws is None else min(x["rate_cv"] for x in ws)
        evals[repr(float(w))] = {"w": float(w), "min_window_rate_cv": val, "windows": ws,
                                 "pacemakers": int(spontaneous(arr["b"]).sum())}
        print("cv", w, val, flush=True)
        return val

    cv020, cv030 = cv(0.20), cv(0.30)
    gate("R_reproduction", cv020 is not None and cv030 is not None
         and abs(cv020 - sealed_cv["0.2"]) <= 1e-12 and abs(cv030 - sealed_cv["0.3"]) <= 1e-12)
    c_log = []
    wc_lo, wc_hi = bisect(lambda w: (cv(w) or -1.0) >= CV_TARGET, 0.20, 0.30, 0.005, c_log)
    cv_at_wsp = cv(ws_hi)
    gate("V3_below", cv_at_wsp is not None and cv_at_wsp < CV_TARGET
         and all(sealed_cv[k] < CV_TARGET for k in ("0.0", "0.05", "0.1", "0.15")))
    pts = sorted((e["w"], e["min_window_rate_cv"]) for e in evals.values())
    nonmonotone = any(b[1] < a[1] for a, b in zip(pts, pts[1:]))
    gate("nonfinite_free", all(e["min_window_rate_cv"] is not None for e in evals.values()))

    if failures:
        ordering, verdict = None, "INTRINSIC_HETEROGENEITY_UNRESOLVED"
    elif ws_hi < wc_lo:
        ordering = "SPONTANEOUS_BEFORE_CV"
    elif wc_hi < ws_lo:
        ordering = "CV_BEFORE_SPONTANEOUS"
    else:
        ordering = "COINCIDENT"
    if not failures:
        verdict = VERDICTS[ordering]

    # reported, not gating
    sens = {}
    for name, fn in (("dur_40s", lambda b: spontaneous(b, dur_s=40.0)),
                     ("v0_-70", lambda b: spontaneous_from(b, -70.0)),
                     ("v0_-60", lambda b: spontaneous_from(b, -60.0))):
        lg = []
        lo_ok, hi_ok = fn([0.25, 0.28])
        sens[name] = (bisect(lambda x: bool(fn([x])[0]), 0.25, 0.28, 1e-6, lg)
                      if (not lo_ok and hi_ok) else {"bracket_failed": [bool(lo_ok), bool(hi_ok)]})
    ih.DT_MS = 0.05
    try:
        lo_ok, hi_ok = spontaneous([0.25, 0.28])
        sens["dt_0p05"] = (bisect(lambda x: bool(spontaneous(x)[0]), 0.25, 0.28, 1e-6, [])
                           if (not lo_ok and hi_ok) else {"bracket_failed": [bool(lo_ok), bool(hi_ok)]})
    finally:
        ih.DT_MS = 0.1
    pacemakers = {k: int(spontaneous(population_b(w)).sum())
                  for k, w in (("w_sp_hi", ws_hi), ("w_CV_lo", wc_lo), ("w_CV_hi", wc_hi), ("0.2", 0.2), ("0.3", 0.3))}

    out = {
        "lineage": "V2.1c-boundary", "spec": "results/v21c_boundary_spec.json",
        "gates": gates, "failed_gates": failures,
        "b_crit": {"silent": bc_lo, "fires": bc_hi, "bisection": a_log},
        "w_spontaneous": {"S_false": ws_lo, "S_true": ws_hi, "bisection": b_log,
                          "first_cell": {"index": first, "b": float(b_hi[first]), "rate_I0_Hz": float(first_rate[0]),
                                         "rank_from_top": 1}},
        "w_CV": {"below": wc_lo, "at_or_above": wc_hi, "target": CV_TARGET, "bisection": c_log,
                 "cv_at_w_sp_hi": cv_at_wsp, "sealed_cv": sealed_cv, "evaluations": sorted(evals.values(), key=lambda e: e["w"]),
                 "evaluated_nonmonotone": nonmonotone,
                 "estimator": {"bin_width_ms": 5000.0, "aggregation": "rate-CV across 300 cells per 5 s window; min over windows 1-3",
                               "baseline": "none", "sign": "not applicable (dispersion statistic)"}},
        "ordering": ordering,
        "reported_not_gating": {"b_crit_sensitivity": sens, "rest_point": rest_point_analytics(), "pacemaker_counts": pacemakers},
        "residual_assumption": spec["classification"]["residual_assumption"],
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print("verdict", verdict, ordering, gates, flush=True)
    assert verdict in set(VERDICTS.values())
