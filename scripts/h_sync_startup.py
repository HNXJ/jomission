"""Sync-term start-up transient test (spec: docs/designs/H_SYNC_STARTUP_spec.md).

Usage:
  python scripts/h_sync_startup.py control
  python scripts/h_sync_startup.py run       # one process per (condition, seed), then summary
  python scripts/h_sync_startup.py key NAME  # internal
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
import jomission.dynamics.activity_balance as ab  # noqa: E402
import jomission.dynamics.activity_balance_sync as sy  # noqa: E402
import jomission.dynamics.activity_balance_sync_fixref as fx  # noqa: E402

SPEC = REPO / "docs/designs/H_SYNC_STARTUP_spec.md"
OUT = REPO / "docs/receipts/h_sync_startup"
SYNC_RECEIPTS = REPO / "docs/receipts/h_sync_test"
DUR = 4000.0
SEEDS = (0, 1, 2, 3)
BIN_STEPS = 2500  # 250 ms at dt 0.1


def hdp(rule: str, **rp) -> dict:
    return {"hdp_rule": rule, "hdp_rule_params": rp, "record_weight_trace": False}


def rate(sp) -> float:
    return float(np.asarray(sp, dtype=np.float64).sum() / sp.shape[1] / (DUR / 1000.0))


def bins(sp) -> list:
    s = np.asarray(sp, dtype=np.float64)
    return [round(float(v), 3) for v in s.reshape(-1, BIN_STEPS, s.shape[1]).sum(axis=(1, 2)) / (s.shape[1] * 0.25)]


def sigma_of(diag) -> np.ndarray:
    aux = np.asarray(diag["aux_trace"], dtype=np.float64)
    ps, pf = aux[:, :, 0].mean(1), aux[:, :, 1].mean(1)
    return np.maximum(0.0, pf - 1.5 * ps) / np.maximum(ps, 1e-3)


def run_key(key: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cname, seed = key.rsplit("_seed", 1)
    seed = int(seed)
    m = T.build(T.CONDITIONS[cname])
    base = HB.run(m, seed, None, DUR)[0]
    p_ref = rate(base)
    bal = HB.run(m, seed, hdp(ab.RULE_NAME), DUR)[0]
    syn, d = HB.run(m, seed, hdp(sy.RULE_NAME), DUR)
    sig = sigma_of(d)
    hmin_sync = float(np.asarray(d["H_trace"], dtype=np.float64).mean(axis=1).min())
    fxs, d2 = HB.run(m, seed, hdp(fx.RULE_NAME, p_ref=p_ref), DUR)
    hmin_fx = float(np.asarray(d2["H_trace"], dtype=np.float64).mean(axis=1).min())
    bb, bs = bins(bal), bins(syn)
    rec_t = next((250 * (i + 1) for i, (a, b) in enumerate(zip(bs, bb)) if b > 0 and a >= 0.8 * b), None)
    r = {"p_ref_Hz": p_ref, "rate_baseline": rate(base), "rate_balance": rate(bal), "rate_sync": rate(syn),
         "rate_fixref": rate(fxs), "rate_sync_0_2s": float(np.asarray(syn, dtype=np.float64)[:20000].sum() / syn.shape[1] / 2.0),
         "sigma_peak_0_500ms": float(sig[:5000].max()), "sigma_peak_all": float(sig.max()),
         "Hmean_min_sync": hmin_sync, "Hmean_min_fixref": hmin_fx,
         "recovery_ms": rec_t if rec_t is not None else ">4000",
         "bins250_balance": bb, "bins250_sync": bs, "bins250_fixref": bins(fxs)}
    (OUT / f"{key}.json").write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(key, "rate bal %.2f sync %.2f fixref %.2f | sigma peak(0-500) %.1f | Hmin sync %.3f fixref %.3f | recov %s"
          % (r["rate_balance"], r["rate_sync"], r["rate_fixref"], r["sigma_peak_0_500ms"],
             hmin_sync, hmin_fx, r["recovery_ms"]), flush=True)
    return 0


def verdicts(rec: dict) -> dict:
    ks = sorted(rec)
    v = {"HS1_sync_le_half_balance": {k: bool(rec[k]["rate_sync"] <= 0.5 * rec[k]["rate_balance"]) for k in ks},
         "HS2_sigma_peak_ge_10": {k: bool(rec[k]["sigma_peak_0_500ms"] >= 10.0) for k in ks},
         "HS3_fixref_ge_0p8_balance": {k: bool(rec[k]["rate_fixref"] >= 0.8 * rec[k]["rate_balance"]) for k in ks},
         "HS4_H_floor_sync_and_not_fixref": {k: bool(rec[k]["Hmean_min_sync"] <= 0.2 and rec[k]["Hmean_min_fixref"] >= 0.6)
                                             for k in ks}}
    for h in list(v):
        v[h]["all"] = all(v[h].values())
    v["startup_explanation_supported"] = bool(all(v[h]["all"] for h in list(v)))
    return v


def control() -> int:
    m = T.build(T.BASE_TONIC)
    bal = HB.run(m, 0, hdp(ab.RULE_NAME), DUR)[0]
    nul = HB.run(m, 0, hdp(sy.RULE_NAME, kappa_s=0.0), DUR)[0]
    s1 = HB.run(m, 0, hdp(sy.RULE_NAME), DUR)[0]
    s2 = HB.run(m, 0, hdp(sy.RULE_NAME), DUR)[0]
    repro = {}
    for cname in T.CONDITIONS:
        mm = m if cname.startswith("A_") else T.build(T.CONDITIONS[cname])
        for seed in (0, 1):
            sp = s1 if (cname.startswith("A_") and seed == 0) else HB.run(mm, seed, hdp(sy.RULE_NAME), DUR)[0]
            old = json.loads((SYNC_RECEIPTS / f"{cname}_seed{seed}.json").read_text(encoding="utf-8"))["sync"]["bar_r_Hz"]
            new = float(np.asarray(sp, dtype=np.float64)[:20000].sum() / sp.shape[1] / 2.0)
            repro[f"{cname}_seed{seed}"] = {"receipt": old, "now": new, "ok": bool(abs(old - new) < 1e-9)}
    ut = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         str(REPO / "tests/test_activity_balance.py"),
                         str(REPO / "tests/test_activity_balance_sync.py"),
                         str(REPO / "tests/test_activity_balance_sync_fixref.py")],
                        capture_output=True, text=True, cwd=REPO)
    res = {"null_kappa_s0_equals_balance": bool(np.array_equal(bal, nul)),
           "sync_deterministic": bool(np.array_equal(s1, s2)),
           "reproduces_sync_test_0_2s": repro, "unit_tests_pass": ut.returncode == 0,
           "unit_tests_tail": ut.stdout.strip().splitlines()[-1] if ut.stdout.strip() else ""}
    res["control_pass"] = bool(res["null_kappa_s0_equals_balance"] and res["sync_deterministic"]
                               and all(x["ok"] for x in repro.values()) and res["unit_tests_pass"])
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def run_all() -> int:
    keys = [f"{c}_seed{s}" for c in T.CONDITIONS for s in SEEDS]
    for k in keys:
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "key", k]).returncode
        if rc != 0:
            print("FAILED", k, rc)
            return rc
    rec = {k: json.loads((OUT / f"{k}.json").read_text(encoding="utf-8")) for k in keys}
    v = verdicts(rec)
    summary = {"spec": "docs/designs/H_SYNC_STARTUP_spec.md", "spec_sha256": A.sha256_of(SPEC),
               "script_sha256": A.sha256_of(Path(__file__).resolve()),
               "rule_sha256": A.sha256_of(Path(sy.__file__).resolve()),
               "probe_sha256": A.sha256_of(Path(fx.__file__).resolve()), "verdicts": v, "runs": sorted(rec)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({h: x["all"] if isinstance(x, dict) else x for h, x in v.items()}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run", "key"])
    ap.add_argument("name", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "key":
        return run_key(a.name)
    return control() if a.cmd == "control" else run_all()


if __name__ == "__main__":
    raise SystemExit(main())
