"""AGSDR stages 1 and 2 under amendment 4 on the stage-0 feasible set (spec: docs/designs/SCI-AGSDR-STAGE12-A4_spec.md).

Usage:
  python scripts/agsdr_stage12_a4.py control          # C1: seed 0 reproduces docs/receipts/agsdr_stage0_a4 (AGS_S0_06)
  python scripts/agsdr_stage12_a4.py run CAND SEED    # one (candidate, seed) receipt
  python scripts/agsdr_stage12_a4.py rank STAGE       # stage 1 or 2 ranking from the receipts
"""
from __future__ import annotations

import argparse
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
import agsdr_stage0_a4 as S0  # noqa: E402
import agsdr_stage0_correction as C  # noqa: E402
import vip_drive_probe as P  # noqa: E402

SPEC = REPO / "docs/designs/SCI-AGSDR-STAGE12-A4_spec.md"
OUT = REPO / "docs/receipts/agsdr_stage12_a4"
A4 = REPO / "docs/receipts/agsdr_stage0_a4"
EPS = 0.05
SEEDS = {1: (0, 1), 2: (0, 1, 2, 3)}
TOP_STAGE2 = 3


def class_tau(model) -> dict:
    el = model.params["edge_list"]
    cells = np.array([r["cell_type"] for r in model.neuron_table()])
    pre = np.asarray(el.pre)
    tau = np.asarray(el.tau_ms, dtype=np.float64)
    e = cells[pre] == "E"
    te, ti = np.unique(tau[e]), np.unique(tau[~e])
    if te.size != 1 or ti.size != 1:
        raise RuntimeError(f"tau not single-valued per presynaptic class: E {te}, I {ti}")
    return {"E": float(te[0]), "I": float(ti[0]), "n_E": int(np.sum(cells == "E"))}


def evaluate(cand: dict, seed: int) -> dict:
    base = A.build_candidate_model(cand, seed)
    model, _, d, cells = P.scale_vip_drive(base, S0.VIP_SCALE)
    spikes, v_m = P.simulate(model, cand, seed)
    res = A.score_run(model, spikes, v_m, cand, seed)
    m = res["metrics"]
    t = class_tau(model)
    values = {g["id"]: g["value"] for g in res["gates"]}
    efrac_tau = C.tau_weighted_efrac(m["I_E_proxy"], m["I_I_proxy"], t["E"], t["I"])
    gates_tau = C.evaluate_gates({**values, "H09_Efrac_open_0p05_0p95": efrac_tau})
    tail = round(m["E_P_CV_gt_0p5"] * m["E_defined_CV_count"])
    h05 = tail / t["n_E"]
    return {"candidate_id": cand["candidate_id"], "seed": seed, "vip_drive_scale": S0.VIP_SCALE,
            "class_mean_Hz": m["class_mean_Hz"], "viable_bands": P.viable(m["class_mean_Hz"]),
            "gates_tau": gates_tau, "H05_fraction_all_E": h05,
            "hard_pass": bool(all(gates_tau.values()) and h05 >= 0.15),
            "soft_terms": res["soft_terms"], "soft_vector": res["soft_vector"], "soft_distance": res["soft_distance"],
            "Efrac_tau_weighted": efrac_tau, "tau": t, "metrics": m}


def feasible_set() -> list[str]:
    s = json.loads((REPO / "docs/receipts/agsdr_h07h11/h07_q18_summary.json").read_text(encoding="utf-8"))
    return s["final_hard_feasible_amendment_4"]


def cand_by_id(cid: str) -> dict:
    return next(c for c in S0.candidates() if c["candidate_id"] == cid)


def run_one(cid: str, seed: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r = evaluate(cand_by_id(cid), seed)
    (OUT / f"{cid}_seed{seed}.json").write_text(json.dumps(A._jsonable(r), indent=1, sort_keys=True,
                                                          allow_nan=False) + "\n", encoding="utf-8")
    print(cid, seed, {k: round(v, 2) for k, v in r["class_mean_Hz"].items()}, "hard", r["hard_pass"],
          "soft", None if r["soft_distance"] is None else round(r["soft_distance"], 4), flush=True)
    return 0


def eps_dominates(a: list[float], b: list[float]) -> bool:
    """Q6-AGSDR: A dominates B iff J_k(A) <= J_k(B) + eps for all k and J_k(A) < J_k(B) - eps for some k."""
    return all(x <= y + EPS for x, y in zip(a, b)) and any(x < y - EPS for x, y in zip(a, b))


def rank(stage: int) -> int:
    seeds = SEEDS[stage]
    pool = feasible_set() if stage == 1 else json.loads((OUT / "stage1_ranking.json").read_text(encoding="utf-8"))["advance_to_stage2"]
    rows = {}
    for cid in pool:
        recs = [json.loads((OUT / f"{cid}_seed{s}.json").read_text(encoding="utf-8")) for s in seeds]
        hard_all = all(r["hard_pass"] for r in recs)
        vec = np.mean([r["soft_vector"] for r in recs], axis=0).tolist() if all(r["soft_vector"] for r in recs) else None
        sd = float(np.mean([r["soft_distance"] for r in recs])) if vec is not None else math.nan
        terms = list(recs[0]["soft_terms"])
        per = {k: [r["soft_terms"][k] for r in recs] for k in terms}
        ci = {}
        if stage == 2:
            from scipy import stats
            for k, v in per.items():
                v = np.asarray(v, dtype=float)
                h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
                ci[k] = {"mean": float(v.mean()), "ci95": [float(v.mean() - h), float(v.mean() + h)],
                         "spread": [float(v.min()), float(v.max())]}
        rows[cid] = {"hard_pass_all_seeds": hard_all, "failed_seeds": [r["seed"] for r in recs if not r["hard_pass"]],
                     "cross_seed_soft_vector": vec, "soft_terms_order": terms, "cross_seed_soft_distance": sd,
                     "per_seed_soft_terms": per, "class_mean_Hz_per_seed": {r["seed"]: r["class_mean_Hz"] for r in recs},
                     "viable_bands_all_seeds": all(r["viable_bands"]["all"] for r in recs), "stage2_stats": ci}
    feas = [c for c, v in rows.items() if v["hard_pass_all_seeds"] and v["cross_seed_soft_vector"] is not None]
    for c in feas:
        rows[c]["dominated_by"] = sorted(o for o in feas if o != c and
                                         eps_dominates(rows[o]["cross_seed_soft_vector"], rows[c]["cross_seed_soft_vector"]))
    order = sorted(feas, key=lambda c: (len(rows[c]["dominated_by"]), rows[c]["cross_seed_soft_distance"], c))
    out = {"spec": "docs/designs/SCI-AGSDR-STAGE12-A4_spec.md", "spec_sha256": A.sha256_of(SPEC),
           "script_sha256": A.sha256_of(Path(__file__).resolve()), "stage": stage, "seeds": list(seeds),
           "pool": pool, "rows": rows, "feasible_all_seeds": feas, "ranking": order}
    if stage == 1:
        out["advance_to_stage2"] = order[:TOP_STAGE2]
    else:
        out["theta_star"] = order[0] if order else None
        out["note"] = "theta_star is reported; advancing it to a ledger edge is an owner decision (design step 5)"
    name = "stage1_ranking.json" if stage == 1 else "stage2_ranking.json"
    (OUT / name).write_text(json.dumps(A._jsonable(out), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k in ("feasible_all_seeds", "ranking", "advance_to_stage2", "theta_star")}, indent=1))
    for c in order:
        print(c, "dominated_by", rows[c]["dominated_by"], "soft", round(rows[c]["cross_seed_soft_distance"], 4))
    return 0


def control() -> int:
    r = evaluate(cand_by_id("AGS_S0_06"), 0)
    ref = json.loads((A4 / "AGS_S0_06.json").read_text(encoding="utf-8"))
    res = {"C1_metrics_equal": r["metrics"] == ref["metrics"],
           "C1_gates_tau_equal": r["gates_tau"] == ref["gates_tau"],
           "C1_efrac_tau_equal": r["Efrac_tau_weighted"] == ref["Efrac_tau_weighted"],
           "C1_H05_equal": r["H05_fraction_all_E"] == ref["H05_fraction_all_E"],
           "C2_tau_E_I": [r["tau"]["E"], r["tau"]["I"]], "C2_n_E": r["tau"]["n_E"]}
    res["control_pass"] = bool(res["C1_metrics_equal"] and res["C1_gates_tau_equal"] and res["C1_efrac_tau_equal"]
                               and res["C1_H05_equal"])
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run", "rank"])
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?", type=int)
    x = ap.parse_args(argv)
    if x.cmd == "control":
        return control()
    if x.cmd == "run":
        return run_one(x.a, x.b)
    return rank(int(x.a))


if __name__ == "__main__":
    raise SystemExit(main())
