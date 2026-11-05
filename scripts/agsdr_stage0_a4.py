"""AGSDR stage 0 rerun under amendment 4 (frozen VIP drive x2.25) (spec: docs/designs/SCI-AGSDR-STAGE0-A4_spec.md).

Usage:
  python scripts/agsdr_stage0_a4.py control     # C1: scale 1.0 reproduces stored stage-0 metrics and gates (AGS_S0_06)
  python scripts/agsdr_stage0_a4.py run         # 24 candidates, seed 0; per-candidate receipts and summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402
import agsdr_stage0_correction as C  # noqa: E402
import vip_drive_probe as P  # noqa: E402

SPEC = REPO / "docs/designs/SCI-AGSDR-STAGE0-A4_spec.md"
AMENDMENT = REPO / "docs/designs/SCI-AGSDR-CALIBRATION_amendment_4.md"
OUT = REPO / "docs/receipts/agsdr_stage0_a4"
VIP_SCALE = 2.25
H07H11 = REPO / "docs/receipts/agsdr_h07h11/summary.json"


def candidates() -> list[dict]:
    out = []
    for f in sorted(A.RECEIPT_DIR.glob("candidate_AGS_S0_*_seed0.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        out.append({"candidate_id": r["candidate_id"], "values": r["values"], "builder_args": r["builder_args"],
                    "motif_gains_effective": r["motif_gains_effective"], "mu_total": A.mu_of(r["values"])})
    if len(out) != 24:
        raise RuntimeError(f"expected 24 candidates, found {len(out)}")
    return out


def evaluate(cand: dict, scale: float, tau: dict) -> dict:
    base = A.build_candidate_model(cand, 0)
    model, d0, d, cells = P.scale_vip_drive(base, scale)
    spikes, v_m = P.simulate(model, cand, 0)
    res = A.score_run(model, spikes, v_m, cand, 0)
    m = res["metrics"]
    values = {g["id"]: g["value"] for g in res["gates"]}
    tau_e, tau_i = C._single_tau(tau, "E_pre"), C._single_tau(tau, "I_pre")
    efrac_tau = C.tau_weighted_efrac(m["I_E_proxy"], m["I_I_proxy"], tau_e, tau_i)
    gates_tau = C.evaluate_gates({**values, "H09_Efrac_open_0p05_0p95": efrac_tau})
    n_def, n_e = m["E_defined_CV_count"], tau["n_E"]
    tail = round(m["E_P_CV_gt_0p5"] * n_def)
    h05_allE = tail / n_e
    c = m["class_mean_Hz"]
    return {"candidate_id": cand["candidate_id"], "vip_drive_scale": scale,
            "vip_drive_mean": float(d[cells == "VIP"].mean()),
            "class_mean_Hz": c, "viable_bands": P.viable(c), "bar_r_Hz": m["bar_r_Hz"],
            "gates_runner": {g["id"]: g["pass"] for g in res["gates"]}, "hard_pass_runner": bool(res["hard_pass"]),
            "Efrac_proxy": m["Efrac_proxy_global"], "Efrac_tau_weighted": efrac_tau,
            "gates_tau": gates_tau, "hard_pass_tau": all(gates_tau.values()),
            "H05_tail_count": tail, "H05_fraction_all_E": h05_allE, "H05_pass_all_E": h05_allE >= 0.15,
            "feasible_pre_structural": bool(all(gates_tau.values()) and h05_allE >= 0.15),
            "failed_tau": sorted(k for k, ok in gates_tau.items() if not ok) + ([] if h05_allE >= 0.15 else ["H05_all_E"]),
            "soft_distance_runner": res["soft_distance"], "metrics": m}


def control() -> int:
    tc = json.loads(C.TAU_CHECK.read_text(encoding="utf-8"))["candidates"]
    cand = next(c for c in candidates() if c["candidate_id"] == "AGS_S0_06")
    r = evaluate(cand, 1.0, tc["AGS_S0_06"])
    stored = json.loads((A.RECEIPT_DIR / "candidate_AGS_S0_06_seed0.json").read_text(encoding="utf-8"))
    corr = {x["candidate_id"]: x for x in json.loads(C.CORRECTION.read_text(encoding="utf-8"))["per_candidate"]}["AGS_S0_06"]
    res = {"C1_metrics_equal": r["metrics"] == stored["metrics"],
           "C1_gates_equal": r["gates_runner"] == {g["id"]: g["pass"] for g in stored["hard_gates"]},
           "C1_efrac_tau_equal": r["Efrac_tau_weighted"] == corr["Efrac_tau_weighted"],
           "C1_H05_allE_equal": r["H05_fraction_all_E"] == corr["H05_fraction_all_E_denominator"],
           "C1_hard_pass_tau_equal": r["hard_pass_tau"] == corr["hard_pass_tau"]}
    res["control_pass"] = all(res.values())
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tc = json.loads(C.TAU_CHECK.read_text(encoding="utf-8"))["candidates"]
    rows = []
    for cand in candidates():
        r = evaluate(cand, VIP_SCALE, tc[cand["candidate_id"]])
        (OUT / f"{cand['candidate_id']}.json").write_text(json.dumps(A._jsonable(r), indent=1, sort_keys=True,
                                                                     allow_nan=False) + "\n", encoding="utf-8")
        rows.append(r)
        print(r["candidate_id"], {k: round(v, 2) for k, v in r["class_mean_Hz"].items()}, "bands", r["viable_bands"]["all"],
              "feasible", r["feasible_pre_structural"], r["failed_tau"], flush=True)
    s = json.loads(H07H11.read_text(encoding="utf-8"))
    feas = sorted(r["candidate_id"] for r in rows if r["feasible_pre_structural"])
    after_h11 = sorted(c for c in feas if s["H11_pass"][c])
    summary = {"spec": "docs/designs/SCI-AGSDR-STAGE0-A4_spec.md", "spec_sha256": A.sha256_of(SPEC),
               "amendment_sha256": A.sha256_of(AMENDMENT), "script_sha256": A.sha256_of(Path(__file__).resolve()),
               "runner_sha256": A.sha256_of(A.THIS_FILE), "vip_drive_scale": VIP_SCALE,
               "n_candidates": len(rows),
               "viable_bands_all": sorted(r["candidate_id"] for r in rows if r["viable_bands"]["all"]),
               "vip_in_band": sorted(r["candidate_id"] for r in rows if r["viable_bands"]["VIP"]),
               "feasible_H01_H10d_tau_H05allE": feas, "feasible_after_H11": after_h11,
               "H07_status": s["H07_status"], "final_hard_feasible": None if not s["H07_determined"] else
               sorted(c for c in after_h11 if s["H07_pass"][c]),
               "count_after_H11": len(after_h11), "slots_stage1": 8}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("viable_bands_all", "feasible_H01_H10d_tau_H05allE", "feasible_after_H11",
                                              "H07_status", "count_after_H11")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run"])
    a = ap.parse_args(argv)
    return control() if a.cmd == "control" else run_all()


if __name__ == "__main__":
    raise SystemExit(main())
