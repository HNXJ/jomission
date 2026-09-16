"""V2.1b class-resolved ISI-CV: instrumentation rerun of one frozen cell.

Question: is E itself temporally irregular under shot drive, or is the pooled
V2.1 ISI-CV fraction carried by inhibitory classes?

Cell choice (results/v21b_isi_rule.json, sealed before this rerun): among
V2.1b cells with E in [2, 30] Hz in all analysis windows, the maximum over
cells of the minimum pooled ISI-CV fraction across windows.

Method: tests/test_v21b_operation.py run_cell called unchanged, with two
module-level patches: its result write goes to a temp file (the sealed cell
JSON is never touched), and the frozen window_isis is wrapped to also record
per-class CVs from the same per-neuron values. The rerun must reproduce the
sealed cell (rates, cv, sync, drift, checks, realized) exactly; otherwise the
reading is UNRESOLVED.

Reading: E_IRREGULAR if the E fraction of CV-evaluable E cells with CV in
[0.5, 1.5] is >= 0.8 in every window (the V2.1 gate applied to E alone);
E_NOT_IRREGULAR otherwise.
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "results" / "v21b_isi_rule.json"
OUT = ROOT / "results" / "v21b_isi_classes.json"
CLASSES = ("E", "PV", "SST", "VIP")
EXACT_KEYS = ("rates", "cv", "sync", "drift", "checks", "realized", "pass", "cell_pass")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_rule(cells):
    """cells: {key: sealed cell JSON}. Returns (chosen key, candidate table)."""
    table = {}
    for k, r in cells.items():
        in_band = all(2.0 <= w["E"] <= 30.0 for w in r["rates"])
        table[k] = {"E_in_band_all_windows": in_band,
                    "min_pooled_frac_in": min(w["frac_in"] for w in r["cv"])}
    cand = {k: v for k, v in table.items() if v["E_in_band_all_windows"]}
    chosen = max(cand, key=lambda k: cand[k]["min_pooled_frac_in"])
    return chosen, table


def _sealed_cells():
    seal = json.load(open(ROOT / "results" / "v21b_vectors.json"))
    return {k: json.load(open(ROOT / "results" / f"v21b_{k}.json")) for k in seal["bracket"]["cells"]}


def test_isi_rule_sealed():
    rule = json.load(open(RULE))
    chosen, table = apply_rule(_sealed_cells())
    assert chosen == rule["chosen"]
    assert table == rule["candidates"]


def _class_stats(nsp, cv_all, cls):
    out = {}
    for c in CLASSES:
        sel = cls == c
        v = cv_all[sel]
        v = v[~np.isnan(v)]
        out[c] = {"n_cells": int(sel.sum()), "cv_evaluable": int(len(v)),
                  "mean_spikes": float(nsp[sel].mean()),
                  "frac_in": (float(((v >= 0.5) & (v <= 1.5)).mean()) if len(v) else None),
                  "cv_quantiles_10_50_90": ([float(q) for q in np.quantile(v, [0.1, 0.5, 0.9])]
                                            if len(v) else None)}
    return out


def test_v21b_isi_classes():
    rule = json.load(open(RULE))
    regime, cell = rule["chosen"].split("_", 1)
    b = _load("_v21b_battery", ROOT / "tests" / "test_v21b_operation.py")
    sealed = json.load(open(ROOT / "results" / f"v21b_{rule['chosen']}.json"))

    records = []
    cls_holder = {}
    orig_load = b._load_v21

    def load_and_wrap():
        v21 = orig_load()
        frozen_isis = v21.window_isis
        frozen_build = v21.build_plant

        def build_plant(vec):
            model, step_fn, cls = frozen_build(vec)
            cls_holder["cls"] = cls
            return model, step_fn, cls

        def window_isis(spikes):
            nsp, cvs = frozen_isis(spikes)
            # Same per-neuron CV as the frozen estimator, kept at neuron index.
            cv_all = np.full(spikes.shape[1], np.nan)
            for i in range(spikes.shape[1]):
                idx = np.flatnonzero(spikes[:, i] > 0.5)
                if len(idx) >= 4:
                    isi = np.diff(idx).astype(float) * v21.DT_MS
                    cv_all[i] = float(isi.std() / max(isi.mean(), 1e-9))
            assert np.array_equal(np.sort(cv_all[~np.isnan(cv_all)]), np.sort(cvs))
            records.append((nsp, cv_all))
            return nsp, cvs

        v21.window_isis = window_isis
        v21.build_plant = build_plant
        return v21

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dst = Path(tmp) / "rerun.json"
        orig_redirect = b._redirect_open

        def redirect(name):
            op, _ = orig_redirect(name)
            src = f"results/v21_{name}.json"

            def _open(path, mode="r", *a, **k):
                if str(path) == src:
                    return open(tmp_dst, mode, *a, **k)
                return op(path, mode, *a, **k)
            return _open, tmp_dst

        b._load_v21 = load_and_wrap
        b._redirect_open = redirect
        res = b.run_cell(regime, cell)
        rerun = json.load(open(tmp_dst))

    reproduced = {k: rerun[k] == sealed[k] for k in EXACT_KEYS}
    cur_rel = []
    for wr, ws in zip(rerun["currents"], sealed["currents"]):
        for key, d in ws.items():
            for f, v in d.items():
                cur_rel.append(abs(wr[key][f] - v) / max(abs(v), 1e-12))
    assert len(records) == 3, len(records)
    cls = cls_holder["cls"]
    per_window = [_class_stats(nsp, cv_all, cls) for nsp, cv_all in records]
    ok = all(reproduced.values())
    e_fracs = [w["E"]["frac_in"] for w in per_window]
    if not ok or any(f is None for f in e_fracs):
        reading = "UNRESOLVED"
    elif all(f >= 0.8 for f in e_fracs):
        reading = "E_IRREGULAR"
    else:
        reading = "E_NOT_IRREGULAR"
    out = {"cell": rule["chosen"], "rule": "results/v21b_isi_rule.json",
           "reproduction_exact": reproduced,
           "currents_max_rel_diff": float(max(cur_rel)) if cur_rel else None,
           "pooled_frac_in_sealed": [w["frac_in"] for w in sealed["cv"]],
           "per_window_class": per_window,
           "E_frac_in": e_fracs,
           "reading": reading,
           "scope": "one frozen V2.1b cell; class-resolved instrumentation only"}
    json.dump(out, open(OUT, "w"), indent=2)
    print("ISI classes:", reading, "E frac", e_fracs, "reproduced", ok, flush=True)
    assert res["cell"] == cell
