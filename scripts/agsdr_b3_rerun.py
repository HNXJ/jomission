"""SCI-AGSDR-CALIBRATION B3 rerun (sealed spec: docs/designs/SCI-AGSDR-B3-RERUN_spec.md).

Reruns the 24 stage-0 candidates with the runner's own build, simulate and score functions,
checks reproduction against the stored receipts, and accumulates per-population E/I currents
(tau-free and tau-weighted) to recompute B3. Writes new receipts only.

Usage:
  python scripts/agsdr_b3_rerun.py smoke AGS_S0_06   # one candidate, reproduction check
  python scripts/agsdr_b3_rerun.py run               # all 24, writes receipts + summary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402

SPEC = A.REPO / "docs/designs/SCI-AGSDR-B3-RERUN_spec.md"
OUT = A.REPO / "docs/receipts/agsdr_b3_rerun"
RECEIPTS = A.RECEIPT_DIR
B3_BAND = (0.15, 0.60)
B3_TARGET = 0.8


def pop_ids(model):
    """Population index per neuron, as in the runner's score_run (area, layer, cell_type)."""
    rows = model.neuron_table()
    keys = [(r["area"], r["layer"], r["cell_type"]) for r in rows]
    index = {k: i for i, k in enumerate(sorted(set(keys)))}
    pid = np.array([index[k] for k in keys], dtype=np.int64)
    cell = np.array([r["cell_type"] for r in rows])
    return pid, len(index), cell


def per_pop_sums(model, spikes) -> dict:
    """Per-population E and I currents: tau-free (runner proxy) and tau-weighted (frozen J_EI)."""
    S = np.asarray(spikes, dtype=np.float64)
    rates = S.sum(axis=0) / (A.DURATION_MS / 1000.0)
    pid, n_pop, cell = pop_ids(model)
    el = model.params["edge_list"]
    pre = np.asarray(el.pre).astype(np.int64)
    post = np.asarray(el.post).astype(np.int64)
    w = np.abs(np.asarray(el.weight, dtype=np.float64))
    tau = np.asarray(el.tau_ms, dtype=np.float64)
    pre_e = (cell == "E")[pre]
    # tau must be single-valued per presynaptic class (tau-check receipt)
    tau_e = sorted({float(t) for t in tau[pre_e]})
    tau_i = sorted({float(t) for t in tau[~pre_e]})
    if tau_e != [2.0] or tau_i != [5.0]:
        raise RuntimeError(f"tau not as verified: E-pre {tau_e}, I-pre {tau_i}")
    contrib = w * rates[pre]
    tgt = pid[post]

    def acc(mask, weights):
        return np.bincount(tgt[mask], weights=weights[mask], minlength=n_pop)

    e_free = acc(pre_e, contrib)
    i_free = acc(~pre_e, contrib)
    e_tau = acc(pre_e, contrib * tau)
    i_tau = acc(~pre_e, contrib * tau)
    return {"E_free": e_free, "I_free": i_free, "E_tau": e_tau, "I_tau": i_tau, "n_pop": n_pop}


def b3_from_sums(e: np.ndarray, i: np.ndarray) -> dict:
    """B3 as the runner defines it: populations with nonzero current; in-band fraction; soft term."""
    has = (e + i) > 0
    ratio = e[has] / (e[has] + i[has])
    n = int(ratio.size)
    frac = float(np.mean((ratio >= B3_BAND[0]) & (ratio <= B3_BAND[1]))) if n else math.nan
    term = max(0.0, B3_TARGET - frac) / B3_TARGET if n else math.nan
    return {"n_populations_with_current": n, "frac_in_band": frac, "B3_term": term}


def _receipt(cid: str) -> dict:
    return json.loads((RECEIPTS / f"candidate_{cid}_seed0.json").read_text(encoding="utf-8"))


def run_candidate(cid: str) -> dict:
    rec = _receipt(cid)
    cand = {"candidate_id": cid, "values": rec["values"], "builder_args": rec["builder_args"],
            "mu_total": rec["metrics"]["mu_total_proxy"]}
    model, spikes, v_m = A.simulate_candidate(cand, 0)
    res = A.score_run(model, spikes, v_m, cand, 0)
    m, st = res["metrics"], rec["metrics"]
    sums = per_pop_sums(model, spikes)
    free = b3_from_sums(sums["E_free"], sums["I_free"])
    tau = b3_from_sums(sums["E_tau"], sums["I_tau"])

    repro = {
        "total_spikes_exact": m["total_spikes"] == st["total_spikes"],
        "bar_r_abs_diff": abs(m["bar_r_Hz"] - st["bar_r_Hz"]),
        "mean_rho_abs_diff": abs(m["mean_rho"] - st["mean_rho"]),
        "I_E_rel_diff": abs(m["I_E_proxy"] - st["I_E_proxy"]) / max(1e-300, abs(st["I_E_proxy"])),
        "I_I_rel_diff": abs(m["I_I_proxy"] - st["I_I_proxy"]) / max(1e-300, abs(st["I_I_proxy"])),
    }
    repro["pass"] = bool(repro["total_spikes_exact"] and repro["bar_r_abs_diff"] <= 1e-9
                         and repro["mean_rho_abs_diff"] <= 1e-9
                         and repro["I_E_rel_diff"] <= 1e-9 and repro["I_I_rel_diff"] <= 1e-9)
    control_tau_free = abs(free["frac_in_band"] - st["population_Efrac_frac_in_0p15_0p60"])
    out = {
        "candidate_id": cid,
        "reproduction": repro,
        "control_tau_free_frac_matches_stored": bool(control_tau_free <= 1e-12),
        "control_tau_free_abs_diff": control_tau_free,
        "tau_free": free,
        "tau_weighted": tau,
        "stored_population_frac": st["population_Efrac_frac_in_0p15_0p60"],
        "n_pop_total": int(sums["n_pop"]),
    }
    out["pass"] = bool(repro["pass"] and out["control_tau_free_frac_matches_stored"])
    return out


def smoke(cid: str) -> int:
    r = run_candidate(cid)
    print(json.dumps({k: r[k] for k in ("candidate_id", "reproduction", "pass",
                                        "control_tau_free_frac_matches_stored")}, indent=1))
    return 0 if r["pass"] else 1


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cids = [f"AGS_S0_{i:02d}" for i in range(24)]
    results = []
    for cid in cids:
        r = run_candidate(cid)
        (OUT / f"candidate_{cid}_b3.json").write_text(
            json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        results.append(r)
        print(cid, "pass" if r["pass"] else "FAIL", flush=True)
        if not r["pass"]:
            print("STOP: control failed; no further candidates", flush=True)
            break
    all_pass = len(results) == 24 and all(r["pass"] for r in results)
    summary = {
        "schema": "agsdr_b3_rerun_summary_v1",
        "spec": "docs/designs/SCI-AGSDR-B3-RERUN_spec.md",
        "spec_sha256": A.sha256_of(SPEC),
        "runner_sha256": A.sha256_of(A.THIS_FILE),
        "rerun_script_sha256": A.sha256_of(Path(__file__).resolve()),
        "controls_all_pass": all_pass,
        "candidates_run": len(results),
        "candidates": {r["candidate_id"]: {
            "pass": r["pass"],
            "B3_tau_weighted": r["tau_weighted"]["frac_in_band"],
            "B3_term_tau_weighted": r["tau_weighted"]["B3_term"],
            "B3_frac_tau_free": r["tau_free"]["frac_in_band"],
            "B3_term_tau_free": r["tau_free"]["B3_term"],
        } for r in results},
        "selection": "NOT RUN: no selection in this spec (OPEN-26 shortfall; OPEN-25 stage-1 condition)",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print("controls_all_pass", all_pass, "wrote", OUT / "summary.json")
    return 0 if all_pass else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("candidate")
    sub.add_parser("run")
    a = ap.parse_args(argv)
    return smoke(a.candidate) if a.cmd == "smoke" else run_all()


if __name__ == "__main__":
    raise SystemExit(main())
