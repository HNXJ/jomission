"""SCI-AGSDR-CALIBRATION J_rates rerun (sealed spec: docs/designs/SCI-AGSDR-JRATES_spec.md).

Reruns the 24 stage-0 candidates with the runner's build, simulate and score functions, saves
per-population mean rates mu_p, checks reproduction against the stored receipts, and computes
J_rates under rulings R1 to R5 (2026-11-04). Writes new receipts only.

Usage:
  python scripts/agsdr_jrates_rerun.py smoke AGS_S0_06
  python scripts/agsdr_jrates_rerun.py run
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

SPEC = A.REPO / "docs/designs/SCI-AGSDR-JRATES_spec.md"
OUT = A.REPO / "docs/receipts/agsdr_jrates_rerun"
RECEIPTS = A.RECEIPT_DIR

# Class-global E/PV/SST/VIP targets (frozen) and per-layer E targets (frozen, partial).
CLASS_TARGET = {"E": (5.0, 8.0), "PV": (12.0, 25.0), "SST": (6.0, 12.0), "VIP": (8.0, 15.0)}
E_LAYER_TARGET = {("V1", "L4"): (6.0, 7.0), ("V1", "L2/3"): (4.0, 6.0), ("V1", "L5"): (7.0, 10.0)}
E_FEF_PFC_TARGET = (3.0, 6.0)  # R2: every layer of FEF and PFC
SOFT_BAND = 0.5  # +/-50 percent (frozen)
COVERAGE_TARGET = 0.60
W_COVERAGE = 0.5  # R4
W_ORDERING = 0.5  # R4


def target_for(area: str, layer: str, cell: str) -> tuple[float, float]:
    """Band [0.5 lo, 1.5 hi] for the population's target (R1 to R2)."""
    if cell == "E":
        if (area, layer) in E_LAYER_TARGET:
            lo, hi = E_LAYER_TARGET[(area, layer)]
        elif area in ("FEF", "PFC"):
            lo, hi = E_FEF_PFC_TARGET
        else:
            lo, hi = CLASS_TARGET["E"]  # R1
    else:
        lo, hi = CLASS_TARGET[cell]
    return (1.0 - SOFT_BAND) * lo, (1.0 + SOFT_BAND) * hi


def population_means(model, spikes) -> dict:
    """Per-population mean rate mu_p = mean over neurons of r_i (spikes / 2 s), runner labels."""
    S = np.asarray(spikes, dtype=np.float64)
    rates = S.sum(axis=0) / (A.DURATION_MS / 1000.0)
    rows = model.neuron_table()
    keys = [(r["area"], r["layer"], r["cell_type"]) for r in rows]
    labels = sorted(set(keys))
    index = {k: i for i, k in enumerate(labels)}
    pid = np.array([index[k] for k in keys], dtype=np.int64)
    counts = np.bincount(pid, minlength=len(labels)).astype(np.float64)
    sums = np.bincount(pid, weights=rates, minlength=len(labels))
    cell = np.array([r["cell_type"] for r in rows])
    class_mean = {c: float(rates[cell == c].mean()) for c in sorted(set(cell.tolist()))}
    return {"labels": labels, "mu": sums / counts, "n": counts.astype(int),
            "class_mean": class_mean, "rates_mean": float(rates.mean())}


def j_rates(labels, mu) -> dict:
    """Coverage, ordering and J_rates per the R1 to R4 rulings. Inclusive band edges."""
    inband = []
    for (area, layer, cell), m in zip(labels, mu):
        lo, hi = target_for(area, layer, cell)
        inband.append(lo <= m <= hi)
    coverage = float(np.mean(inband))
    by_al = {}
    for (area, layer, cell), m in zip(labels, mu):
        by_al.setdefault((area, layer), {})[cell] = float(m)
    pairs = [(v["E"], v["PV"]) for v in by_al.values() if "E" in v and "PV" in v]
    violations = sum(1 for e, p in pairs if not (e < p))
    ordering_penalty = violations / len(pairs) if pairs else math.nan
    jr = W_COVERAGE * (1.0 - coverage) + W_ORDERING * ordering_penalty
    return {"coverage": coverage, "n_populations": len(labels), "n_inband": int(sum(inband)),
            "ordering_pairs": len(pairs), "ordering_violations": int(violations),
            "ordering_penalty": ordering_penalty, "J_rates": jr,
            "coverage_soft_target_met": bool(coverage >= COVERAGE_TARGET)}


def _receipt(cid: str) -> dict:
    return json.loads((RECEIPTS / f"candidate_{cid}_seed0.json").read_text(encoding="utf-8"))


def run_candidate(cid: str) -> dict:
    rec = _receipt(cid)
    cand = {"candidate_id": cid, "values": rec["values"], "builder_args": rec["builder_args"],
            "mu_total": rec["metrics"]["mu_total_proxy"]}
    model, spikes, v_m = A.simulate_candidate(cand, 0)
    res = A.score_run(model, spikes, v_m, cand, 0)
    m, st = res["metrics"], rec["metrics"]
    pm = population_means(model, spikes)
    jr = j_rates(pm["labels"], pm["mu"])

    repro = {
        "total_spikes_exact": m["total_spikes"] == st["total_spikes"],
        "bar_r_abs_diff": abs(m["bar_r_Hz"] - st["bar_r_Hz"]),
        "mean_rho_abs_diff": abs(m["mean_rho"] - st["mean_rho"]),
        "class_mean_max_abs_diff": max(abs(pm["class_mean"][c] - st["class_mean_Hz"][c])
                                       for c in st["class_mean_Hz"]),
    }
    repro["pass"] = bool(repro["total_spikes_exact"] and repro["bar_r_abs_diff"] <= 1e-9
                         and repro["mean_rho_abs_diff"] <= 1e-9
                         and repro["class_mean_max_abs_diff"] <= 1e-9
                         and abs(pm["rates_mean"] - st["bar_r_Hz"]) <= 1e-9)
    out = {
        "candidate_id": cid,
        "reproduction": repro,
        "populations": [{"area": a, "layer": l, "cell_type": c, "n": int(n), "mu_Hz": float(mu)}
                        for (a, l, c), n, mu in zip(pm["labels"], pm["n"], pm["mu"])],
        "J_rates": jr,
    }
    out["pass"] = repro["pass"]
    return out


def smoke(cid: str) -> int:
    r = run_candidate(cid)
    print(json.dumps({"candidate_id": cid, "reproduction": r["reproduction"], "pass": r["pass"],
                      "J_rates": r["J_rates"]}, indent=1))
    return 0 if r["pass"] else 1


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cids = [f"AGS_S0_{i:02d}" for i in range(24)]
    results = []
    for cid in cids:
        r = run_candidate(cid)
        (OUT / f"candidate_{cid}_jrates.json").write_text(
            json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        results.append(r)
        print(cid, "pass" if r["pass"] else "FAIL", flush=True)
        if not r["pass"]:
            print("STOP: reproduction control failed; no further candidates", flush=True)
            break
    all_pass = len(results) == 24 and all(r["pass"] for r in results)
    summary = {
        "schema": "agsdr_jrates_rerun_summary_v1",
        "spec": "docs/designs/SCI-AGSDR-JRATES_spec.md",
        "spec_sha256": A.sha256_of(SPEC),
        "rerun_script_sha256": A.sha256_of(Path(__file__).resolve()),
        "runner_sha256": A.sha256_of(A.THIS_FILE),
        "controls_all_pass": all_pass,
        "candidates_run": len(results),
        "candidates": {r["candidate_id"]: {"pass": r["pass"], "J_rates": r["J_rates"]["J_rates"],
                                           "coverage": r["J_rates"]["coverage"],
                                           "ordering_penalty": r["J_rates"]["ordering_penalty"]}
                       for r in results},
        "selection": "NOT RUN: J_rates alone does not select; the six-term rule and the shortfall "
                     "(OPEN-26) still apply",
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
