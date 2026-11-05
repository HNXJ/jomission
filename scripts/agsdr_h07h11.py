"""AGSDR hard gates H07 (locality) and H11 (CL1-A audit) on the 24 stage-0 candidates (spec: docs/designs/SCI-AGSDR-H07H11_spec.md).

Usage:
  python scripts/agsdr_h07h11.py control      # controls 1 to 3 (canonical passes, globalized fails, altered metadata fails)
  python scripts/agsdr_h07h11.py run          # H07, H11 for 24 candidates; saturation report; summary with final feasible set
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402

SPEC = REPO / "docs/designs/SCI-AGSDR-H07H11_spec.md"
OUT = REPO / "docs/receipts/agsdr_h07h11"
CORRECTION = A.RECEIPT_DIR / "stage0_correction_v1.json"
PLAN = A.RECEIPT_DIR / "stage0_plan.json"
VERTICAL = {("L4", "L2/3"), ("L2/3", "L5")}
NONVERT_MAX, VERT_MAX = 25, 40
WITHIN_BOX, BETWEEN_BOX = (9000, 11000), (531, 649)
Q_MIN, LOC_MIN, DENSIFY = 0.2, 3.0, 1.10
H11_FIXED = {"delay_ff_ms": 8.0, "delay_fb_ms": 12.0, "delay_within_ms": 2.0, "weight_lognormal_cv": 1.5,
             "weight_lognormal_sigma": 1.085658784490618, "weight_lognormal_mu": -0.5893274981708232}
H11_PRESENT = ("motif_gain", "delay_ff_ms", "delay_fb_ms", "delay_within_ms", "weight_lognormal_cv",
               "weight_lognormal_sigma", "weight_lognormal_mu", "background_poisson_rate_hz",
               "background_poisson_amplitude", "heterogeneity_jitter")


def graph(model) -> dict:
    el = model.params["edge_list"]
    t = model.neuron_table()
    return {"pre": np.asarray(el.pre, dtype=np.int64), "post": np.asarray(el.post, dtype=np.int64),
            "area": np.array([r["area"] for r in t]), "layer": np.array([r["layer"] for r in t]),
            "xyz": np.array([[r["x"], r["y"], r["z"]] for r in t], dtype=np.float64)}


def modularity(pre, post, area) -> float:
    """Newman Q of the undirected multigraph (each directed edge one undirected edge), areas as partition."""
    m = len(pre)
    q = 0.0
    for a in np.unique(area):
        ina = area == a
        e_in = np.sum(ina[pre] & ina[post]) / m
        deg = (np.sum(ina[pre]) + np.sum(ina[post])) / (2.0 * m)
        q += e_in - deg ** 2
    return float(q)


def locality_index(pre, post, area, xyz) -> float:
    """Mean distance of all within-area ordered pairs / mean distance of within-area connected pairs."""
    w = area[pre] == area[post]
    d_conn = np.linalg.norm(xyz[pre[w]] - xyz[post[w]], axis=1).mean()
    tot, cnt = 0.0, 0
    for a in np.unique(area):
        P = xyz[area == a]
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        n = len(P)
        tot += D.sum()
        cnt += n * (n - 1)
    return float((tot / cnt) / d_conn)


def h07_terms(g: dict, ref_within: int, ref_p: float, meta: dict) -> dict:
    pre, post, area, layer = g["pre"], g["post"], g["area"], g["layer"]
    w = area[pre] == area[post]
    pair = np.array([(layer[i], layer[j]) in VERTICAL for i, j in zip(pre, post)])
    nv_max = v_max = 0
    for sel, name in ((w & ~pair, "nv"), (w & pair, "v")):
        if sel.any():
            mx = int(np.bincount(post[sel]).max())  # per-post in-degree, pooled (builder cap is per post)
            nv_max, v_max = (mx, v_max) if name == "nv" else (nv_max, mx)
    n_within, n_between = int(w.sum()), int((~w).sum())
    n_pairs = sum(int(np.sum(area == a)) * (int(np.sum(area == a)) - 1) for a in np.unique(area))
    p_within = n_within / n_pairs
    q = modularity(pre, post, area)
    loc = locality_index(pre, post, area, g["xyz"])
    wl = meta.get("within_locality") or {}
    terms = {
        "t1_declared_sigma_and_cap": bool(wl.get("spatial_sigma") == 0.08 and wl.get("max_in_degree") == 25),
        "t1_nonvertical_max_in_degree": nv_max, "t1_vertical_max_in_degree": v_max,
        "t1_pass": bool(wl.get("spatial_sigma") == 0.08 and wl.get("max_in_degree") == 25
                        and nv_max <= NONVERT_MAX and v_max <= VERT_MAX),
        "t2_within": n_within, "t2_between": n_between,
        "t2_pass": bool(WITHIN_BOX[0] <= n_within <= WITHIN_BOX[1] and BETWEEN_BOX[0] <= n_between <= BETWEEN_BOX[1]),
        "t3_Q": q, "t3_pass": bool(q > Q_MIN),
        "t4_locality_index": loc, "t4_pass": bool(loc > LOC_MIN),
        "t5_within_over_ref": n_within / ref_within, "t5_p_within_over_ref": p_within / ref_p,
        "t5_pass": bool(n_within <= DENSIFY * ref_within and p_within <= DENSIFY * ref_p),
        "p_within": p_within,
    }
    terms["H07_pass"] = all(terms[f"t{i}_pass"] for i in range(1, 6))
    return terms


def poisson_box() -> dict:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    b = plan["log10_bounds"]
    return {k: [10 ** b[k][0], 10 ** b[k][1]] for k in ("poisson_rate_hz", "poisson_amplitude")}


def h11_terms(meta: dict, box: dict) -> dict:
    present = {k: k in meta and meta[k] is not None for k in H11_PRESENT}
    fixed = {k: (meta.get(k) == v) for k, v in H11_FIXED.items()}
    rate, amp = meta.get("background_poisson_rate_hz"), meta.get("background_poisson_amplitude")
    pin = bool(rate is not None and amp is not None
               and box["poisson_rate_hz"][0] <= rate <= box["poisson_rate_hz"][1]
               and box["poisson_amplitude"][0] <= amp <= box["poisson_amplitude"][1])
    t = {"t1_e_mixture_M2": meta.get("e_mixture_version") == "M2", "t1_present": present,
         "t2_fixed_equal_canonical": fixed, "t2_poisson_in_box": pin}
    t["H11_pass"] = bool(t["t1_e_mixture_M2"] and all(present.values()) and all(fixed.values()) and pin)
    return t


def canonical():
    from jomission.network.builder import build_jomission_model
    return build_jomission_model(n_per_area=100, seed=0, dt_ms=A.DT_MS)


def control() -> int:
    m = canonical()
    g = graph(m)
    w = g["area"][g["pre"]] == g["area"][g["post"]]
    n_pairs = sum(int(np.sum(g["area"] == a)) * (int(np.sum(g["area"] == a)) - 1) for a in np.unique(g["area"]))
    ref_within, ref_p = int(w.sum()), int(w.sum()) / n_pairs
    c1 = h07_terms(g, ref_within, ref_p, m.cfg.metadata)
    rng = np.random.default_rng(20261105)
    gg = dict(g)
    pre, post = g["pre"].copy(), g["post"].copy()
    n = len(g["area"])
    k = int(w.sum())
    rp = rng.integers(0, n, k)
    rq = rng.integers(0, n - 1, k)
    rq = rq + (rq >= rp)
    pre[w], post[w] = rp, rq
    gg["pre"], gg["post"] = pre, post
    c2 = h07_terms(gg, ref_within, ref_p, m.cfg.metadata)
    meta = dict(m.cfg.metadata)
    box = {"poisson_rate_hz": [0.0, 1e9], "poisson_amplitude": [0.0, 1e9]}
    c3_ref = h11_terms(meta, box)
    bad = copy.deepcopy(meta)
    bad["delay_fb_ms"] = 6.0
    c3 = h11_terms(bad, box)
    res = {"C1_canonical_H07": c1, "C1_pass": c1["H07_pass"],
           "C2_globalized": {k: c2[k] for k in ("t3_Q", "t4_locality_index", "t3_pass", "t4_pass")},
           "C2_pass": bool(not c2["t3_pass"] and not c2["t4_pass"]),
           "C3_canonical_H11_pass": c3_ref["H11_pass"], "C3_altered_fails": bool(not c3["H11_pass"])}
    res["control_pass"] = bool(res["C1_pass"] and res["C2_pass"] and res["C3_canonical_H11_pass"] and res["C3_altered_fails"])
    print(json.dumps(res, indent=1, default=float))
    return 0 if res["control_pass"] else 1


def saturation(cand: dict) -> dict:
    """Report only: |bar_r(2 x within_gain) - bar_r(within_gain)| / bar_r(within_gain), 2000 ms, seed 0."""
    from jomission.network import builder as B

    out = {}
    for label, scale in (("x1", 1.0), ("x2", 2.0)):
        c = copy.deepcopy(cand)
        c["builder_args"] = dict(c["builder_args"], within_gain=B.WITHIN_GAIN_DEFAULT * scale)
        _, spikes, _ = A.simulate_candidate(c, 0)
        out[label] = float(np.asarray(spikes).sum() * 1000.0 / A.DURATION_MS / spikes.shape[1])
    out["delta_rel"] = abs(out["x2"] - out["x1"]) / out["x1"] if out["x1"] > 0 else math.nan
    out["saturating_flag"] = bool(out["delta_rel"] <= 0.05)
    return out


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m = canonical()
    g0 = graph(m)
    w = g0["area"][g0["pre"]] == g0["area"][g0["post"]]
    n_pairs = sum(int(np.sum(g0["area"] == a)) * (int(np.sum(g0["area"] == a)) - 1) for a in np.unique(g0["area"]))
    ref_within, ref_p = int(w.sum()), int(w.sum()) / n_pairs
    box = poisson_box()
    corr = json.loads(CORRECTION.read_text(encoding="utf-8"))
    rows = {}
    for f in sorted(A.RECEIPT_DIR.glob("candidate_AGS_S0_*_seed0.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        cand = {"candidate_id": r["candidate_id"], "values": r["values"], "builder_args": r["builder_args"],
                "motif_gains_effective": r["motif_gains_effective"], "mu_total": A.mu_of(r["values"])}
        model = A.build_candidate_model(cand, 0)
        h07 = h07_terms(graph(model), ref_within, ref_p, model.cfg.metadata)
        h11 = h11_terms(model.cfg.metadata, box)
        sat = saturation(cand)
        rec = {"candidate_id": cand["candidate_id"], "H07": h07, "H11": h11, "saturation_report_only": sat}
        (OUT / f"{cand['candidate_id']}.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + "\n",
                                                          encoding="utf-8")
        rows[cand["candidate_id"]] = rec
        print(cand["candidate_id"], "H07", h07["H07_pass"], "H11", h11["H11_pass"], "sat", round(sat["delta_rel"], 3), flush=True)
    prior = feasible_prior(corr)
    ref_terms = h07_terms(g0, ref_within, ref_p, m.cfg.metadata)
    h07_determined = bool(ref_terms["H07_pass"])  # stop rule: a gate that fails its reference is not interpreted
    after_h11 = sorted(c for c in prior if rows[c]["H11"]["H11_pass"])
    final = sorted(c for c in after_h11 if rows[c]["H07"]["H07_pass"]) if h07_determined else None
    failing_ref_terms = [k for k in ref_terms if k.endswith("_pass") and k != "H07_pass" and not ref_terms[k]]
    summary = {"spec": "docs/designs/SCI-AGSDR-H07H11_spec.md", "spec_sha256": A.sha256_of(SPEC),
               "script_sha256": A.sha256_of(Path(__file__).resolve()), "reference": {"within": ref_within, "p_within": ref_p},
               "poisson_box": box, "H07_pass": {c: r["H07"]["H07_pass"] for c, r in rows.items()},
               "H11_pass": {c: r["H11"]["H11_pass"] for c, r in rows.items()},
               "prior_feasible_H01_H10d_H05allE": prior, "after_H11": after_h11,
               "H07_reference_terms": ref_terms, "H07_determined": h07_determined,
               "H07_reference_failing_terms": failing_ref_terms,
               "H07_status": "DETERMINED" if h07_determined else "NOT_DETERMINED: gate fails its canonical reference (stop rule); owner ruling needed",
               "final_hard_feasible": final, "final_count": None if final is None else len(final),
               "upper_bound_feasible": after_h11, "stop_empty_set": bool(final is not None and len(final) == 0)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("prior_feasible_H01_H10d_H05allE", "after_H11", "H07_status",
                                              "H07_reference_failing_terms", "final_hard_feasible")}, indent=1))
    return 0


def feasible_prior(corr: dict) -> list[str]:
    """Feasible before H07/H11: tau-corrected hard pass and H05 under the all-E ruling (stage0_correction_v1.json)."""
    return sorted(v["candidate_id"] for v in corr["per_candidate"]
                  if v["hard_pass_tau"] and v["H05_pass_all_E_denominator"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run"])
    a = ap.parse_args(argv)
    return control() if a.cmd == "control" else run_all()


if __name__ == "__main__":
    raise SystemExit(main())
