"""Why balance-only raises rho and Fano: E-only and class-specific bands (spec: docs/designs/H_BALANCE_MECH_spec.md).

Usage:
  python scripts/h_balance_mech.py control           # C2 E-only leaves interneuron H at 1; C3 band arrays
  python scripts/h_balance_mech.py run KEY           # KEY = A_normal_seed0 ... B_stress_x1p5_seed1; one receipt
  python scripts/h_balance_mech.py summary           # C1 reproduction, M1/M2 verdict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import agsdr_local_sweep as A  # noqa: E402
import h_active_test as T  # noqa: E402
import h_balance_test as HB  # noqa: E402
import h_sync_test as SY  # noqa: E402
import jomission.dynamics.activity_balance as ab  # noqa: E402

SPEC = REPO / "docs/designs/H_BALANCE_MECH_spec.md"
OUT = REPO / "docs/receipts/h_balance_mech"
BAL = REPO / "docs/receipts/h_balance_test"
KEYS = [f"{c}_seed{s}" for c in T.CONDITIONS for s in (0, 1)]
CLASS_RHI = {"E": 10.0, "PV": 25.0, "SST": 12.0, "VIP": 15.0}
OFF_BAND = (0.0, 1.0e6)  # band no rate leaves: delta = 0, H stays 1, b stays 0


def band_params(cells, mode: str) -> dict:
    if mode == "e_only":
        lo = [1.0 if c == "E" else OFF_BAND[0] for c in cells]
        hi = [10.0 if c == "E" else OFF_BAND[1] for c in cells]
    elif mode == "class_rhi":
        lo = [1.0 for _ in cells]
        hi = [CLASS_RHI[c] for c in cells]
    else:
        raise ValueError(mode)
    return {"hdp_rule": ab.RULE_NAME, "hdp_rule_params": {"r_lo": lo, "r_hi": hi}, "record_weight_trace": False}


def metrics(spikes, cells) -> dict:
    m = SY.metrics(spikes, cells)
    rates = np.asarray(spikes, dtype=np.float64).sum(axis=0) / (T.DUR / 1000.0)
    m["class_rate_sd_Hz"] = {c: float(rates[cells == c].std()) for c in T.CLASSES}
    return m


def run_one(key: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cname, seed = key.rsplit("_seed", 1)
    seed = int(seed)
    m = T.build(T.CONDITIONS[cname])
    cells = np.array([r["cell_type"] for r in m.neuron_table()])
    rec = {"key": key}
    for arm, hp in {"baseline": None, "ref": HB.hdp_params(),
                    "e_only": band_params(cells, "e_only"), "class_rhi": band_params(cells, "class_rhi")}.items():
        sp, diag = HB.run(m, seed, hp)
        rec[arm] = metrics(sp, cells)
        if diag is not None:
            H = np.asarray(diag["H_final"], dtype=np.float64)
            rec[arm]["H_by_class"] = {c: {"mean": float(H[cells == c].mean()), "min": float(H[cells == c].min()),
                                          "max": float(H[cells == c].max())} for c in T.CLASSES}
        print(key, arm, round(rec[arm]["bar_r_Hz"], 2), round(rec[arm]["mean_rho"], 4),
              round(rec[arm]["pop_synchrony_fano_10ms"], 2), flush=True)
    rec["spec_sha256"] = A.sha256_of(SPEC)
    rec["script_sha256"] = A.sha256_of(Path(__file__).resolve())
    (OUT / f"{key}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def rise_fraction(r: dict, arm: str) -> float:
    den = r["ref"]["mean_rho"] - r["baseline"]["mean_rho"]
    return (r[arm]["mean_rho"] - r["baseline"]["mean_rho"]) / den


def summary() -> int:
    recs = {k: json.loads((OUT / f"{k}.json").read_text(encoding="utf-8")) for k in KEYS}
    c1 = {}
    for k, r in recs.items():
        old = json.loads((BAL / f"{k}.json").read_text(encoding="utf-8"))
        c1[k] = all(old[a][f] == r[a][f] for a in ("baseline",) for f in ("bar_r_Hz", "mean_rho", "pop_synchrony_fano_10ms")) \
            and all(old["balance"][f] == r["ref"][f] for f in ("bar_r_Hz", "mean_rho", "pop_synchrony_fano_10ms"))
    B = [k for k in KEYS if k.startswith("B_")]
    R = {k: {arm: rise_fraction(recs[k], arm) for arm in ("e_only", "class_rhi")} for k in KEYS}
    e_sd_comp = {k: recs[k]["e_only"]["class_rate_sd_Hz"]["E"] < recs[k]["baseline"]["class_rate_sd_Hz"]["E"] for k in B}
    m1 = all(R[k]["e_only"] <= 0.5 and R[k]["class_rhi"] <= 0.5 for k in B)
    m2 = all(R[k]["e_only"] > 0.5 and e_sd_comp[k] for k in B)
    if not all(c1.values()):
        verdict = "CONTROL_FAILED_NO_INTERPRETATION"
    elif m1:
        verdict = "M1_INHIBITORY_TRACKING_SUPPORTED"
    elif m2:
        verdict = "M2_HOMOGENISATION_SUPPORTED"
    else:
        verdict = "NEITHER_M1_NOR_M2"
    out = {"spec": "docs/designs/H_BALANCE_MECH_spec.md", "spec_sha256": A.sha256_of(SPEC),
           "script_sha256": A.sha256_of(Path(__file__).resolve()), "rule_sha256": A.sha256_of(Path(ab.__file__).resolve()),
           "C1_reproduces_h_balance_test": c1, "rise_fraction": R, "e_rate_sd_compressed_e_only_B": e_sd_comp,
           "M1": m1, "M2": m2, "verdict": verdict,
           "table": {k: {a: {"bar_r": round(recs[k][a]["bar_r_Hz"], 3), "rho": round(recs[k][a]["mean_rho"], 4),
                             "rho10": round(recs[k][a]["mean_rho_10ms_bins"], 4),
                             "fano": round(recs[k][a]["pop_synchrony_fano_10ms"], 3),
                             "class_Hz": {c: round(v, 2) for c, v in recs[k][a]["class_mean_Hz"].items()}}
                         for a in ("baseline", "ref", "e_only", "class_rhi")} for k in KEYS}}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("C1_reproduces_h_balance_test", "rise_fraction", "M1", "M2", "verdict")}, indent=1))
    return 0


def control() -> int:
    m = T.build(T.BASE_TONIC)
    cells = np.array([r["cell_type"] for r in m.neuron_table()])
    _, d = HB.run(m, 0, band_params(cells, "e_only"), 200.0)
    H = np.asarray(d["H_final"], dtype=np.float64)
    hp = band_params(cells, "class_rhi")["hdp_rule_params"]
    res = {"C2_interneuron_H_exactly_1": bool(np.all(H[cells != "E"] == 1.0)),
           "C2_E_H_engaged": bool(H[cells == "E"].std() > 0),
           "C3_class_rhi_values": {c: sorted({hp["r_hi"][i] for i in np.where(cells == c)[0]}) for c in T.CLASSES}}
    res["control_pass"] = bool(res["C2_interneuron_H_exactly_1"] and res["C2_E_H_engaged"]
                               and all(v == [CLASS_RHI[c]] for c, v in res["C3_class_rhi_values"].items()))
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run", "summary"])
    ap.add_argument("key", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "control":
        return control()
    if a.cmd == "run":
        return run_one(a.key)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
