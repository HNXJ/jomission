"""Stage-0 correction packet for SCI-AGSDR-CALIBRATION (no simulation).

Subcommands
  tau-check  Build each stage-0 candidate with the runner's builder and record the realized
             tau_ms per presynaptic class, the E cell count and the edge counts. Builds only.
             Writes docs/receipts/agsdr_local/stage0_tau_check.json.
  correct    Read the sealed stage-0 receipts, the tau-check receipt and the LHS plan. Write the
             tau-weighted Efrac, the re-evaluated hard gates, the sampling-departure record and
             the CV-denominator record. Writes docs/receipts/agsdr_local/stage0_correction_v1.json.
             The original stage-0 receipts and stage0_summary.json are read, never written.

Frozen definition (manifests/agsdr_local_objectives.json, J_EI): Efrac = I_E / (I_E + I_I) with
currents W * r * tau. The runner omitted tau. The correction multiplies each class sum by its
class tau; thresholds, weights and candidate parameters are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402  (non-simulating import: runner builds lazily)

RECEIPT_DIR = A.RECEIPT_DIR
TAU_CHECK = RECEIPT_DIR / "stage0_tau_check.json"
CORRECTION = RECEIPT_DIR / "stage0_correction_v1.json"
PLAN = RECEIPT_DIR / "stage0_plan.json"
SUMMARY = RECEIPT_DIR / "stage0_summary.json"

# Hard gates: id -> predicate on the stored value. Mirrors A.score_run exactly; the test suite
# checks it against all 24 stored verdicts, so any drift fails there.
_FIN = math.isfinite


def _open(x: float, lo: float, hi: float) -> bool:
    return _FIN(x) and lo < x < hi


GATE_PREDICATES = {
    "H01_finite": lambda x: bool(x),
    "H02_bar_r_open_1_30": lambda x: _open(x, 1.0, 30.0),
    "H03_mean_rho_below_0p60": lambda x: _FIN(x) and x < 0.60,
    "H04_CV_rate_above_0p15": lambda x: _FIN(x) and x > 0.15,
    "H05_P_CV_E_gt_0p5_at_least_0p15": lambda x: _FIN(x) and x >= 0.15,
    "H06_max_CV_E_at_least_1p5": lambda x: _FIN(x) and x >= 1.5,
    "H09_Efrac_open_0p05_0p95": lambda x: _open(x, 0.05, 0.95),
    "H10a_class_mean_not_above_80": lambda x: x <= 80.0,
    "H10b_E_class_mean_above_0p5": lambda x: x > 0.5,
    "H10c_population_mean_not_above_80": lambda x: x <= 80.0,
    "H10d_E_population_mean_above_0p5": lambda x: x > 0.5,
}


def evaluate_gates(values: dict) -> dict[str, bool]:
    """Pass/fail per hard gate id. A missing id raises (no silent pass).

    The receipts write NaN as null, so None is read as NaN and fails the gate. H01 is a bool;
    None there means the finite check did not record, so it fails too.
    """
    out = {}
    for gid, pred in GATE_PREDICATES.items():
        v = values[gid]
        if v is None:
            v = False if gid == "H01_finite" else math.nan
        out[gid] = bool(pred(v))
    return out


def tau_weighted_efrac(i_e: float, i_i: float, tau_e: float, tau_i: float) -> float:
    """Frozen J_EI: tau_E*I_E / (tau_E*I_E + tau_I*I_I) with I_* = sum |w| r_pre per class."""
    num = tau_e * i_e
    den = num + tau_i * i_i
    return num / den if den > 0 else math.nan


def proxy_efrac(i_e: float, i_i: float) -> float:
    """Runner's recorded proxy (tau omitted). Kept only as the comparison in tests."""
    return i_e / (i_e + i_i) if i_e + i_i > 0 else math.nan


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_receipts() -> list[dict]:
    paths = sorted(RECEIPT_DIR.glob("candidate_AGS_S0_*_seed0.json"))
    if len(paths) != 24:
        raise RuntimeError(f"expected 24 stage-0 candidate receipts, found {len(paths)}")
    return [_load(p) for p in paths]


# ---------------------------------------------------------------- tau-check (builds only)

def tau_check() -> int:
    import numpy as np

    rows = {}
    for r in _candidate_receipts():
        cand = {"candidate_id": r["candidate_id"], "values": r["values"],
                "builder_args": r["builder_args"]}
        model = A.build_candidate_model(cand, 0)
        el = model.params["edge_list"]
        pre = np.asarray(el.pre).astype(np.int64)
        tau = np.asarray(el.tau_ms, dtype=np.float64)
        cells = model.neuron_table()
        is_e = np.array([c["cell_type"] == "E" for c in cells])
        pre_e = is_e[pre]
        classes = {"E_pre": pre_e, "I_pre": ~pre_e}
        rec = {"n_neurons": len(cells), "n_E": int(is_e.sum()), "n_edges": int(pre.size)}
        for name, mask in classes.items():
            vals = sorted({float(t) for t in tau[mask]})
            rec[name] = {"edges": int(mask.sum()), "tau_ms_values": vals}
        rows[r["candidate_id"]] = rec
        print(r["candidate_id"], rec["n_E"], rec["E_pre"]["tau_ms_values"],
              rec["I_pre"]["tau_ms_values"], flush=True)
    out = {
        "schema": "agsdr_stage0_tau_check_v1",
        "purpose": "realized tau_ms per presynaptic class from the runner's EdgeList (builds only)",
        "runner_sha256": A.sha256_of(A.THIS_FILE),
        "correction_script_sha256": A.sha256_of(Path(__file__).resolve()),
        "candidates": rows,
    }
    TAU_CHECK.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", TAU_CHECK)
    return 0


def _single_tau(rec: dict, cls: str) -> float:
    vals = rec[cls]["tau_ms_values"]
    if len(vals) != 1:
        raise RuntimeError(f"{cls} presynaptic tau not single-valued: {vals}")
    return vals[0]


# ---------------------------------------------------------------- sampling departure

def reconstruct_first_draw() -> dict:
    """Replay the LHS columns of A.latin_hypercube_plan and the first (pure LHS) mu per row.

    The LHS stream is generated before any redraw, so the first-draw values are exactly the
    values the plan produced before the mu-window redraw. Rows whose first draw falls outside
    MU_WINDOW are the redrawn rows; the count is checked against the recorded count.
    """
    rng = random.Random(A.LHS_SEED)
    specs = A.coord_specs()
    frozen = {s["name"]: s["baseline"] for s in specs if s["name"] in A.FROZEN_COORDINATES}
    sampled = [s for s in specs if s["name"] not in A.FROZEN_COORDINATES]
    log_lo = {s["name"]: math.log10(s["lo"]) for s in sampled}
    log_hi = {s["name"]: math.log10(s["hi"]) for s in sampled}
    n = A.STAGES[0]["candidates"]
    cols = []
    for _ in sampled:
        strata = list(range(n))
        rng.shuffle(strata)
        cols.append([(strata[i] + rng.random()) / n for i in range(n)])
    first = {}
    for i in range(n):
        vals = dict(frozen)
        for d, s in enumerate(sampled):
            name = s["name"]
            vals[name] = 10 ** (log_lo[name] + cols[d][i] * (log_hi[name] - log_lo[name]))
        first[f"AGS_S0_{i:02d}"] = vals
    return first


def sampling_departure(receipts: list[dict]) -> dict:
    first = reconstruct_first_draw()
    plan = _load(PLAN)
    predicted = sorted(cid for cid, v in first.items() if not (
        A.MU_WINDOW[0] <= A.mu_of(v) <= A.MU_WINDOW[1]))
    bg_names = {n for n, _, _ in A.BACKGROUND_COORDS}
    by_id = {r["candidate_id"]: r for r in receipts}
    mismatches = []
    for cid, r in by_id.items():
        for name, v in r["values"].items():
            if name in bg_names and cid in predicted:
                continue  # background redraw allowed on redrawn rows only
            if first[cid][name] != v:
                mismatches.append([cid, name])
    return {
        "method_executed": "log-space LHS (seed LHS_SEED); rows with mu_total outside MU_WINDOW "
                           "had their unfrozen background coordinates redrawn log-uniformly "
                           "(seeded). Redraws do not preserve one sample per stratum.",
        "method_declared_in_amendment_1": False,
        "mu_window": list(A.MU_WINDOW),
        "rows_redrawn_recorded": plan["rows_redrawn_for_mu"],
        "rows_redrawn_reconstructed": len(predicted),
        "redrawn_candidate_ids": predicted,
        "non_background_or_unredrawn_mismatches": mismatches,
        "consistent": len(predicted) == plan["rows_redrawn_for_mu"] and not mismatches,
    }


# ---------------------------------------------------------------- bound analysis (informational)

FIVE_EFRAC_FREE = ["B1_mu_band_3_15", "B2_CV_rate_above_0p3", "B2_mean_CV_E_band_0p5_1p5",
                   "B2_median_Fano_band_0p7_2", "B2_mean_rho_band_m0p05_0p2"]


def bound_analysis() -> dict:
    """What the five Efrac-free terms alone settle, with B3 in [0, 1] unknown.

    Dominance on the six terms needs dominance on the five. So a candidate dominated on the five
    terms is undetermined, and one dominated on none of them is certainly nondominated. No
    candidate can be certainly dominated, since B3 is unknown for both sides. Strict comparison,
    epsilon 0, as in the runner. Informational only: it does not select.
    """
    vec = {}
    for r in _candidate_receipts():
        vec[r["candidate_id"]] = [r["soft_terms"][k] for k in FIVE_EFRAC_FREE]

    def dom(y, x):
        return all(a <= b for a, b in zip(y, x)) and any(a < b for a, b in zip(y, x))

    certain = sorted(x for x in vec if not any(dom(vec[y], vec[x]) for y in vec if y != x))
    undetermined = sorted(x for x in vec if x not in certain)
    return {
        "scope": "STRICT comparison (epsilon 0), five Efrac-free terms, no B3. NOT valid under the "
                 "frozen seven-dimension epsilon-0.05 rule. Recompute after the dimension set and "
                 "the exact predicate are ruled.",
        "terms_used": FIVE_EFRAC_FREE,
        "certainly_nondominated": certain,
        "undetermined": undetermined,
        "certainly_dominated": [],
        "cut_note": "not settled by bounds: ordering inside an overflowing front uses the full "
                    "soft distance, which includes B3",
    }


# ---------------------------------------------------------------- correct

def correct() -> int:
    tc = _load(TAU_CHECK)
    receipts = _candidate_receipts()
    summary = _load(SUMMARY)
    rows = []
    for r in receipts:
        cid = r["candidate_id"]
        m = r["metrics"]
        t = tc["candidates"][cid]
        tau_e = _single_tau(t, "E_pre")
        tau_i = _single_tau(t, "I_pre")
        i_e, i_i = m["I_E_proxy"], m["I_I_proxy"]
        proxy_rec = m["Efrac_proxy_global"]
        proxy_chk = proxy_efrac(i_e, i_i)
        efrac_tau = tau_weighted_efrac(i_e, i_i, tau_e, tau_i)

        stored = {g["id"]: g for g in r["hard_gates"]}
        values = {gid: g["value"] for gid, g in stored.items()}
        stored_pass = {gid: g["pass"] for gid, g in stored.items()}
        recomputed_stored = evaluate_gates({**values, "H09_Efrac_open_0p05_0p95": proxy_rec})
        values_corr = {**values, "H09_Efrac_open_0p05_0p95": efrac_tau}
        corr_pass = evaluate_gates(values_corr)

        n_e_def = m["E_defined_CV_count"]
        rows.append({
            "candidate_id": cid,
            "I_E_proxy": i_e,
            "I_I_proxy": i_i,
            "tau_E_pre_ms": tau_e,
            "tau_I_pre_ms": tau_i,
            "Efrac_proxy_recorded": proxy_rec,
            "Efrac_proxy_recomputed": proxy_chk,
            "Efrac_tau_weighted": efrac_tau,
            "stored_gate_verdicts_reproduced": recomputed_stored == stored_pass,
            "H09_pass_proxy": stored_pass["H09_Efrac_open_0p05_0p95"],
            "H09_pass_tau": corr_pass["H09_Efrac_open_0p05_0p95"],
            "hard_pass_proxy": all(stored_pass.values()),
            "hard_pass_tau": all(corr_pass.values()),
            "failed_gates_tau": sorted(g for g, ok in corr_pass.items() if not ok),
            "E_cells_total": t["n_E"],
            "E_cells_CV_defined": n_e_def,
            "P_CV_E_gt_0p5": m["E_P_CV_gt_0p5"],
            "B3_population_Efrac_frac_in_band_proxy": m["population_Efrac_frac_in_0p15_0p60"],
            "B3_population_tau_correction": "NOT COMPUTABLE: per-population E/I sums not retained",
        })

    feasible = [x["candidate_id"] for x in rows if x["hard_pass_tau"]]
    proxy_feasible = [x["candidate_id"] for x in rows if x["hard_pass_proxy"]]
    changed = sorted(x["candidate_id"] for x in rows if x["hard_pass_tau"] != x["hard_pass_proxy"])
    out = {
        "schema": "agsdr_stage0_correction_v1",
        "task": "SCI-AGSDR-CALIBRATION",
        "source_receipts": "docs/receipts/agsdr_local/candidate_AGS_S0_*_seed0.json (unchanged)",
        "runner_sha256_in_stage0_summary": summary["runner_sha256"],
        "runner_sha256_current": A.sha256_of(A.THIS_FILE),
        "stage0_summary_file_sha256": A.sha256_of(SUMMARY),
        "tau_check": "docs/receipts/agsdr_local/stage0_tau_check.json",
        "tau_check_sha256": A.sha256_of(TAU_CHECK),
        "correction_script_sha256": A.sha256_of(Path(__file__).resolve()),
        "provenance_note": (
            "Commit c43443e carries the label 'Goal: G-AGSDR-CALIBRATION'. That ID is not a "
            "verified goal in this repo. The verified task identifier is SCI-AGSDR-CALIBRATION."),
        "changed_not": ["gate thresholds and predicates", "objective weights", "candidate "
                        "parameters", "sampling", "CV eligibility", "runner code and receipts"],
        "changed": ["H09 input: runner proxy replaced by tau-weighted Efrac in the record only; "
                    "no receipt or runner file was edited"],
        "frozen_departures": [
            "dimensions: runner soft set has 6 terms (B1, B2 x4, B3); frozen J set has 7 "
            "(J_rates, J_irregularity, J_synchrony, J_Fano, J_hetero, J_EI, J_CL1C). "
            "manifests/agsdr_local_objectives.json algorithm_spec; design note line 167",
            "dominance tolerance: frozen epsilon 0.05 RETAINED (owner ruling 1, 2026-11-04). "
            "The runner and the bound analysis use strict comparison, epsilon 0: not valid for "
            "selection. The exact predicate (absolute or relative, boundary handling) is not "
            "defined in the frozen source (OPEN-23)",
            "overflow: frozen rule keeps N1=8 nondominated survivors with w=1/6 tie-breaker "
            "(agsdr_local_harness.json selection). It does not define overflow beyond the front. "
            "The front-order fill from dominated fronts is WITHDRAWN as automatic (owner ruling "
            "1 note). Overflow is open (OPEN-18)",
            "time basis: frozen H09 and J_EI are per step (edge_current_trace[t]); runner uses a "
            "time-integrated ratio over the 2000 ms run",
            "sign: frozen J_EI takes I_e positive and I_i negative; runner and record take abs(w)",
            "hard list: design line 18 lists nine gates; manifest lists H01-H12. H07 and H11 "
            "are not computed",
        ],
        "superseded_not_edited": {
            "docs/receipts/agsdr_local/stage0_summary.json": [
                "best_feasible (AGS_S0_06 by scalar soft distance): superseded, not a selection",
                "feasible_ranked order: scalar ordering, superseded by the proposal rule",
                "stage1_selection: 'not run' stands",
            ],
        },
        "per_candidate": rows,
        "hard_feasibility": {
            "proxy_feasible_count": len(proxy_feasible),
            "tau_feasible_count": len(feasible),
            "changed_candidate_ids": changed,
        },
        "sampling_departure": sampling_departure(receipts),
        "selection": {
            "status": "NOT DETERMINED",
            "rule_status": "NOT APPLIED. Frozen epsilon 0.05 retained (ruling 1). Dimension set, "
                            "exact predicate and overflow open. No automatic fill from dominated "
                            "fronts.",
            "rule": "Frozen (agsdr_local_harness.json selection; objectives algorithm_spec): among "
                    "tau-feasible candidates, keep the nondominated set on the seven J_k with "
                    "dominance tolerance epsilon 0.05; tie-break by w=1/6 scalar among the "
                    "nondominated only; eliminate the remainder regardless of scalar.",
            "blocked_by": [
                "dimension set: runner computes 6 terms; frozen names 7 (J_hetero, J_CL1C not "
                "computed). Owner ruling 2",
                "exact dominance predicate for epsilon 0.05 (OPEN-23)",
                "overflow beyond the front (OPEN-18)",
                "B3 needs per-population tau-weighted ratios p' = 2p/(5-3p); the receipts keep "
                "only the in-band fraction, so none is retained (OPEN-19)",
            ],
            "eight_selected": None,
            "shortfall_rule": "stop and report if fewer than eight tau-feasible candidates",
            "bound_analysis_STRICT_5TERM_INFORMATIONAL": bound_analysis(),
        },
    }
    CORRECTION.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"feasible proxy {len(proxy_feasible)} tau {len(feasible)} changed {changed}")
    print("sampling consistent", out["sampling_departure"]["consistent"],
          out["sampling_departure"]["rows_redrawn_reconstructed"])
    print("wrote", CORRECTION)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["tau-check", "correct"])
    args = ap.parse_args(argv)
    return tau_check() if args.command == "tau-check" else correct()


if __name__ == "__main__":
    raise SystemExit(main())
