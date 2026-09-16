"""V2.1c intrinsic-heterogeneity design checkpoint (single-cell numerics only).

Protocol and criteria: results/v21c_design_protocol.json (committed before any
output below was computed). Library: jomission/qualification/intrinsic.py.

Fast (pre-output): numerics agree with the sealed H1 assay code; dispersion
is exactly zero-mean with the requested relative sd; shot background columns
are the V2.1b low_sm2p0 generator output.
Slow drivers, one output file each:
  validation   -> results/v21c_validation.json
  sensitivity  -> results/v21c_sensitivity_<p>.json
  width scan   -> results/v21c_width_<p>.json
  verdict      -> results/v21c_design.json
"""

import functools
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jomission.qualification import intrinsic as ih

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "results" / "v21c_design_protocol.json"
P0 = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0}
M = 300
N_CHUNK, STEPS = 4, 50000
NOISE_SEED, PERM_SEED = 31000, 31001
PARAMS = ("a", "b", "c", "d")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _proto():
    return json.load(open(PROTO))


@functools.lru_cache(maxsize=1)
def _inputs():
    """Shot E columns (low_sm2p0), recurrent spec, network targets."""
    from jomission.qualification.cmin import build_cmin
    b = _load("_v21b_battery", ROOT / "tests" / "test_v21b_operation.py")
    seal = json.load(open(ROOT / "results" / "v21b_vectors.json"))
    cfg = seal["bracket"]["cells"]["low_sm2p0"]
    vec = json.load(open(ROOT / "results" / "v21_vectors.json"))["vectors"][cfg["tonic_vector"]]
    model = build_cmin()
    cls = np.array([str(r["cell_type"]) for r in model.neuron_table()])
    mu = np.array([vec[c] for c in cls], dtype=np.float64)
    chunks, _ = b.shot_chunks(mu, cfg["lambda_hz"], seal["shot"]["tau_ms"], 0.1,
                              cfg["shot_seed"], N_CHUNK, STEPS)
    e = cls == "E"
    assert int(e.sum()) == M
    shot_e = [np.ascontiguousarray(ch[:, e]) for ch in chunks]
    cell = json.load(open(ROOT / "results" / "v21b_low_sm2p0.json"))
    rates = {c: float(np.mean([w[c] for w in cell["rates"]])) for c in ih.CLASSES}
    rec = ih.recurrent_spec(model, cls, rates)
    return {"shot_e": shot_e, "rec": rec, "I0": float(vec["E"]), "rates": rates}


def _run(params_arrays, I0=None):
    inp = _inputs()
    p = dict(params_arrays)
    p["I0"] = inp["I0"] if I0 is None else I0
    rasters = ih.simulate_population(p, inp["shot_e"], inp["rec"], N_CHUNK, STEPS, NOISE_SEED)
    if rasters is None:
        return None
    return [ih.window_metrics(r) for r in rasters]


def _homog(**over):
    return {k: np.full(M, over.get(k, v), dtype=np.float64) for k, v in P0.items()}


def _mean(ws, key):
    return float(np.mean([w[key] for w in ws]))


# ---------------- fast, pre-output checks ----------------

def test_deterministic_numerics_match_h1_assay():
    from jomission.qualification import symmetry
    cases = [dict(P0), {**P0, "d": 6.4}, {**P0, "b": 0.24}, {**P0, "c": -58.5}, {**P0, "a": 0.016}]
    for p in cases:
        r_ref, cv_ref = symmetry.run_cell(p, 9.2)
        r, cv = ih.deterministic_rate_cv([p["a"]], [p["b"]], [p["c"]], [p["d"]], 9.2)
        # Running-moment variance cancels to ~1e-8 when all ISIs are equal.
        assert r[0] == r_ref and abs(cv[0] - cv_ref) < 1e-6, (p, r, r_ref, cv, cv_ref)
        rh_grid = symmetry.rheobase(p)
        rh = ih.rheobase([p["a"]], [p["b"]], [p["c"]], [p["d"]])[0]
        assert rh_grid - 1.0 < rh <= rh_grid + 1e-9, (p, rh, rh_grid)


def test_dispersion_zero_mean_exact_width():
    for p0 in (0.02, -65.0, 8.0):
        for w in (0.05, 0.3, 1.0):
            x = ih.lognormal_dispersion(p0, w, M, PERM_SEED)
            assert abs(x.mean() / p0 - 1.0) < 1e-12
            assert abs(x.std() / abs(p0) - w) < 1e-9
            assert np.all(np.sign(x) == np.sign(p0))


def test_protocol_matches_code():
    pr = _proto()
    rm = pr["reduced_model"]
    assert rm["seeds"] == {"recurrent_and_native_noise": NOISE_SEED, "dispersion_permutation": PERM_SEED}
    assert pr["H1_sensitivity"]["parameters"] == list(PARAMS)
    assert pr["H2_H3_width_scan"]["widths"][0] > 0


# ---------------- slow drivers ----------------

def test_v21c_validation():
    base = _run(_homog())
    assert base is not None
    inp = _inputs()
    tgt_isi = json.load(open(ROOT / "results" / "v21b_isi_classes.json"))
    net = {"rate": 19.206, "cv_median": 0.525, "frac_in": 0.722}
    red = {"rate": _mean(base, "rate_mean"), "cv_median": _mean(base, "cv_median"),
           "frac_in": _mean(base, "frac_in"), "rate_cv": _mean(base, "rate_cv")}
    checks = {
        "rate": abs(red["rate"] - net["rate"]) / net["rate"] <= 0.15,
        "cv_median": abs(red["cv_median"] - net["cv_median"]) <= 0.05,
        "frac_in": abs(red["frac_in"] - net["frac_in"]) <= 0.10,
        "rate_cv": 0.02 <= red["rate_cv"] <= 0.07,
    }
    plus = _run(_homog(), I0=inp["I0"] + 0.05)
    minus = _run(_homog(), I0=inp["I0"] - 0.05)
    out = {"network_targets": net,
           "network_E_frac_in_windows": tgt_isi["E_frac_in"],
           "recurrent_spec": inp["rec"],
           "reduced_windows": base, "reduced_means": red, "checks": checks,
           "pass": bool(all(checks.values())),
           "closed_loop_bound": {"dI": 0.05,
                                 "rate_plus": _mean(plus, "rate_mean"),
                                 "rate_minus": _mean(minus, "rate_mean")}}
    json.dump(out, open(ROOT / "results" / "v21c_validation.json", "w"), indent=2)
    print("validation", out["pass"], red, flush=True)


@pytest.mark.parametrize("par", PARAMS)
def test_v21c_sensitivity(par):
    levels = _proto()["H1_sensitivity"]["levels"]
    rows = {}
    for lv in levels:
        val = P0[par] * lv
        ws = _run(_homog(**{par: val}))
        p = {**P0, par: val}
        rh = float(ih.rheobase([p["a"]], [p["b"]], [p["c"]], [p["d"]])[0])
        _, dcv = ih.deterministic_rate_cv([p["a"]], [p["b"]], [p["c"]], [p["d"]], 9.2)
        rows[str(lv)] = {"value": val, "windows": ws,
                         "rate_mean": None if ws is None else _mean(ws, "rate_mean"),
                         "cv_median": None if ws is None else _mean(ws, "cv_median"),
                         "frac_in": None if ws is None else _mean(ws, "frac_in"),
                         "rheobase": rh, "deterministic_cv_I9p2": float(dcv[0])}
    lo, hi = rows["0.9"], rows["1.1"]
    dl = np.log(1.1) - np.log(0.9)
    sens = {
        "S_r": (np.log(hi["rate_mean"]) - np.log(lo["rate_mean"])) / dl
        if lo["rate_mean"] and hi["rate_mean"] else None,
        "S_CV": (hi["cv_median"] - lo["cv_median"]) / dl
        if lo["cv_median"] is not None and hi["cv_median"] is not None else None,
        "S_rh": (hi["rheobase"] - lo["rheobase"]) / dl,
    }
    out = {"parameter": par, "p0": P0[par], "levels": rows, "sensitivity": sens}
    json.dump(out, open(ROOT / "results" / f"v21c_sensitivity_{par}.json", "w"), indent=2)
    print("sensitivity", par, sens, flush=True)


@pytest.mark.parametrize("par", PARAMS)
def test_v21c_width(par):
    widths = [0.0] + _proto()["H2_H3_width_scan"]["widths"]
    rh0 = float(ih.rheobase([P0["a"]], [P0["b"]], [P0["c"]], [P0["d"]])[0])
    rows = {}
    for w in widths:
        arr = _homog()
        arr[par] = ih.lognormal_dispersion(P0[par], w, M, PERM_SEED)
        ws = _run(arr)
        rh = ih.rheobase(arr["a"], arr["b"], arr["c"], arr["d"])
        _, dcv = ih.deterministic_rate_cv(arr["a"], arr["b"], arr["c"], arr["d"], 9.2)
        rows[str(w)] = {
            "windows": ws,
            "nonfinite": ws is None,
            "param_min": float(arr[par].min()), "param_max": float(arr[par].max()),
            "rheobase_mean_shift": (float(np.nanmean(rh) - rh0) if np.isfinite(rh).any() else None),
            "rheobase_unreachable_frac": float(np.isnan(rh).mean()),
            "rheobase_q10_50_90": ([float(q) for q in np.nanquantile(rh, [0.1, 0.5, 0.9])]
                                   if np.isfinite(rh).any() else None),
            "burster_frac_I9p2": float((dcv >= 0.5).mean()),
        }
    out = {"parameter": par, "p0": P0[par], "rheobase_p0": rh0, "widths": rows}
    json.dump(out, open(ROOT / "results" / f"v21c_width_{par}.json", "w"), indent=2)
    print("width", par, {w: (None if r["windows"] is None else round(min(x["rate_cv"] for x in r["windows"]), 3))
                         for w, r in rows.items()}, flush=True)


def evaluate(width_file, validation):
    """Apply protocol F1-F8 to one parameter's width scan. Returns dict."""
    rows = width_file["widths"]
    base = rows["0.0"]["windows"]
    r0 = _mean(base, "rate_mean")
    f0 = min(x["frac_in"] for x in base)
    grid = [float(w) for w in rows if float(w) > 0]
    w_star = None
    for w in sorted(grid):
        ws = rows[str(w)]["windows"]
        if ws is not None and min(x["rate_cv"] for x in ws) >= 0.30:
            w_star = w
            break
    res = {"w_star": w_star, "r0": r0, "frac_in_min_w0": f0}
    if w_star is None:
        res.update({"criteria": None, "meets_all": False,
                    "reason": "E rate-CV >= 0.30 not reached on the grid"})
        return res
    row = rows[str(w_star)]
    ws = row["windows"]
    r = _mean(ws, "rate_mean")
    fmin = min(x["frac_in"] for x in ws)
    rca = [x["rate_cv_active"] for x in ws]
    crit = {
        "F1_plausibility": w_star <= 0.5,
        "F2_temporal": fmin >= f0 - 0.03,
        "F3_silence": max(x["silent_frac"] for x in ws) <= 0.05,
        "F4_runaway": max(x["above_60Hz_frac"] for x in ws) <= 0.05,
        "F5_mean_rate": 2.0 <= r <= 30.0 and abs(r - r0) / r0 <= 0.25,
        "F6_bursting": row["burster_frac_I9p2"] <= 0.05,
        "F7_rheobase": (row["rheobase_mean_shift"] is not None
                        and row["rheobase_unreachable_frac"] == 0.0
                        and abs(row["rheobase_mean_shift"]) <= 0.25),
        "F8_not_silence_driven": all(v is not None for v in rca) and min(rca) >= 0.25,
    }
    res.update({"criteria": crit, "meets_all": bool(all(crit.values())),
                "at_w_star": {"rate_mean": r, "frac_in_min": fmin,
                              "rate_cv_min": min(x["rate_cv"] for x in ws),
                              "rate_cv_active_min": min(rca) if all(v is not None for v in rca) else None,
                              "silent_max": max(x["silent_frac"] for x in ws),
                              "cv_median_mean": _mean(ws, "cv_median"),
                              "rheobase_mean_shift": row["rheobase_mean_shift"],
                              "burster_frac": row["burster_frac_I9p2"],
                              "predicted_E_ISI_gate_0p8": fmin >= 0.8}})
    return res


def test_v21c_verdict():
    val = json.load(open(ROOT / "results" / "v21c_validation.json"))
    per = {}
    for par in PARAMS:
        wf = json.load(open(ROOT / "results" / f"v21c_width_{par}.json"))
        assert wf["widths"]["0.0"]["windows"] == val["reduced_windows"], "w=0 must equal validation base"
        per[par] = evaluate(wf, val)
        per[par]["sensitivity"] = json.load(open(ROOT / "results" / f"v21c_sensitivity_{par}.json"))["sensitivity"]
    if not val["pass"]:
        verdict, selected = "INTRINSIC_HETEROGENEITY_UNRESOLVED", None
    else:
        ok = [p for p in PARAMS if per[p]["meets_all"]]
        if ok:
            selected = sorted(ok, key=lambda p: (per[p]["w_star"], -per[p]["at_w_star"]["frac_in_min"],
                                                 abs(per[p]["at_w_star"]["rate_mean"] - per[p]["r0"])))[0]
            verdict = "INTRINSIC_HETEROGENEITY_FEASIBLE"
        else:
            verdict, selected = "INTRINSIC_HETEROGENEITY_INFEASIBLE", None
    bracket = None
    if selected:
        ws = per[selected]["w_star"]
        bracket = {"parameter": selected, "widths": [0.5 * ws, ws, 1.5 * ws]}
    out = {"protocol": "results/v21c_design_protocol.json", "validation_pass": val["pass"],
           "per_parameter": per, "selected": selected, "bracket_prediction": bracket,
           "verdict": verdict,
           "class_viability_blocker": "low_sm2p0 VIP ~130 Hz exceeds spec band [1, 80]; unaffected by E-only dispersion except via E->VIP recurrence",
           "next": "STOP for review; only FEASIBLE can authorize another V2.1 execution"}
    json.dump(out, open(ROOT / "results" / "v21c_design.json", "w"), indent=2)
    print("V2.1c design verdict:", verdict, selected, flush=True)
