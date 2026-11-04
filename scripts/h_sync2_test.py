"""Corrected synchrony term vs rate-matched baseline (spec: docs/designs/H_SYNC2_TEST_spec.md).

Usage:
  python scripts/h_sync2_test.py control
  python scripts/h_sync2_test.py run     # select (seed 2, B) -> tune A,B -> test cells -> summary
Internal: select | tune COND | key COND_seedN
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
import jomission.dynamics.activity_balance_sync2 as s2  # noqa: E402

SPEC = REPO / "docs/designs/H_SYNC2_TEST_spec.md"
OUT = REPO / "docs/receipts/h_sync2_test"
DUR = 4000.0
GRID = (0.15, 0.5, 1.5)
TUNE_SEED = 2
TEST_SEEDS = (0, 1, 3)
TOL = 0.03
ITERS = 8


def hdp(rule: str, **rp) -> dict:
    return {"hdp_rule": rule, "hdp_rule_params": rp, "record_weight_trace": False}


def rate(sp) -> float:
    return float(np.asarray(sp, dtype=np.float64).sum() / sp.shape[1] / (DUR / 1000.0))


def cells_of(m):
    return np.array([r["cell_type"] for r in m.neuron_table()])


def mets(sp, cells) -> dict:
    return T.metrics(sp, cells, DUR)


def sigma_peak(diag) -> float:
    aux = np.asarray(diag["aux_trace"], dtype=np.float64)
    ps, pf = aux[:, :, 0].mean(1), aux[:, :, 1].mean(1)
    return float((np.maximum(0.0, pf - 1.5 * ps) / s2.DEFAULT_PARAMS["p_ref"])[:5000].max())


def tune_rm(cond: str, target: float) -> dict:
    lo, hi = 0.3, T.CONDITIONS[cond]
    its, best = [], None
    for k in range(ITERS):
        mid = 0.5 * (lo + hi)
        r = rate(HB.run(T.build(mid), TUNE_SEED, None, DUR)[0])
        its.append({"iter": k, "tonic": mid, "bar_r": r})
        if best is None or abs(r - target) < abs(best["bar_r"] - target):
            best = its[-1]
        lo, hi = (mid, hi) if r < target else (lo, mid)
    rel = abs(best["bar_r"] - target) / target
    return {"target_bar_r": target, "tonic": best["tonic"], "achieved_bar_r": best["bar_r"],
            "rel_error": rel, "matched": bool(rel <= TOL), "iterations": its}


def hc_terms(sync2_m, rm_m, base_m, cond) -> dict:
    t = {"HC1_fano_lt_rm": bool(sync2_m["pop_synchrony_fano_10ms"] < rm_m["pop_synchrony_fano_10ms"]),
         "HC2_rho_lt_rm": bool(sync2_m["mean_rho"] < rm_m["mean_rho"])}
    if cond.startswith("B_"):
        t["HC3_barr_le_0p8_baseline"] = bool(sync2_m["bar_r_Hz"] <= 0.8 * base_m["bar_r_Hz"])
    return t


def select() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cond = "B_stress_x1p5"
    m = T.build(T.CONDITIONS[cond])
    cells = cells_of(m)
    base_m = mets(HB.run(m, TUNE_SEED, None, DUR)[0], cells)
    rows, picked = [], None
    for k in GRID:
        sp = HB.run(m, TUNE_SEED, hdp(s2.RULE_NAME, kappa_s=k), DUR)[0]
        tune = tune_rm(cond, rate(sp))
        rm_m = mets(HB.run(T.build(tune["tonic"]), TUNE_SEED, None, DUR)[0], cells)
        s_m = mets(sp, cells)
        terms = hc_terms(s_m, rm_m, base_m, cond)
        ok = bool(tune["matched"] and all(terms.values()))
        rows.append({"kappa_s": k, "sync2": s_m, "rate_matched": rm_m, "tune": tune, "terms": terms, "pass": ok})
        print("grid kappa_s", k, "rate %.2f rm %.2f | fano %.2f vs %.2f | rho %.3f vs %.3f | pass %s"
              % (s_m["bar_r_Hz"], rm_m["bar_r_Hz"], s_m["pop_synchrony_fano_10ms"], rm_m["pop_synchrony_fano_10ms"],
                 s_m["mean_rho"], rm_m["mean_rho"], ok), flush=True)
        if ok and picked is None:
            picked = k
            break
    (OUT / "selection.json").write_text(json.dumps({"grid": list(GRID), "tuning_seed": TUNE_SEED,
                                                     "condition": cond, "baseline": base_m, "rows": rows,
                                                     "picked_kappa_s": picked}, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")
    return 0 if picked is not None else 2


def picked() -> float:
    return float(json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["picked_kappa_s"])


def tune_cond(cond: str) -> int:
    m = T.build(T.CONDITIONS[cond])
    t = tune_rm(cond, rate(HB.run(m, TUNE_SEED, hdp(s2.RULE_NAME, kappa_s=picked()), DUR)[0]))
    (OUT / f"tune_{cond}.json").write_text(json.dumps(t, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("tune", cond, {k: t[k] for k in ("target_bar_r", "tonic", "achieved_bar_r", "rel_error", "matched")}, flush=True)
    return 0


def run_key(key: str) -> int:
    cond, seed = key.rsplit("_seed", 1)
    seed = int(seed)
    tune = json.loads((OUT / f"tune_{cond}.json").read_text(encoding="utf-8"))
    m = T.build(T.CONDITIONS[cond])
    cells = cells_of(m)
    base = HB.run(m, seed, None, DUR)[0]
    bal = HB.run(m, seed, hdp(ab.RULE_NAME), DUR)[0]
    sp, d = HB.run(m, seed, hdp(s2.RULE_NAME, kappa_s=picked()), DUR)
    rm = HB.run(T.build(tune["tonic"]), seed, None, DUR)[0]
    f64 = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    r1 = lambda x: float(f64(x)[:10000].sum() / x.shape[1] / 1.0)  # noqa: E731  rate in [0, 1000) ms
    r = {"kappa_s": picked(), "rm_tonic": tune["tonic"], "baseline": mets(base, cells), "balance": mets(bal, cells),
         "sync2": mets(sp, cells), "rate_matched": mets(rm, cells), "sigma_peak_0_500ms": sigma_peak(d),
         "rate_0_1s": {"balance": r1(bal), "sync2": r1(sp)}}
    r["terms"] = hc_terms(r["sync2"], r["rate_matched"], r["baseline"], cond)
    r["terms"]["HC4_startup_ok"] = bool(r["rate_0_1s"]["sync2"] >= 0.5 * r["rate_0_1s"]["balance"])
    (OUT / f"{key}.json").write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(key, "rate base %.2f bal %.2f sync2 %.2f rm %.2f | fano %.2f %.2f %.2f %.2f | rho %.3f %.3f %.3f %.3f | sigmaPk %.2f"
          % tuple([r[a]["bar_r_Hz"] for a in ("baseline", "balance", "sync2", "rate_matched")]
                  + [r[a]["pop_synchrony_fano_10ms"] for a in ("baseline", "balance", "sync2", "rate_matched")]
                  + [r[a]["mean_rho"] for a in ("baseline", "balance", "sync2", "rate_matched")]
                  + [r["sigma_peak_0_500ms"]]), flush=True)
    return 0


def verdicts(rec: dict, tunes: dict) -> dict:
    out = {}
    for h in ("HC1_fano_lt_rm", "HC2_rho_lt_rm", "HC3_barr_le_0p8_baseline", "HC4_startup_ok"):
        d = {k: r["terms"][h] for k, r in rec.items() if h in r["terms"]}
        d["all"] = all(d.values())
        out[h] = d
    out["rate_matching_ok"] = {c: t["matched"] for c, t in tunes.items()}
    out["rate_matching_on_test_seeds_rel_err"] = {k: abs(r["sync2"]["bar_r_Hz"] - r["rate_matched"]["bar_r_Hz"]) / r["sync2"]["bar_r_Hz"]
                                                  for k, r in rec.items()}
    out["supported"] = bool(all(out[h]["all"] for h in ("HC1_fano_lt_rm", "HC2_rho_lt_rm", "HC3_barr_le_0p8_baseline",
                                                         "HC4_startup_ok")) and all(out["rate_matching_ok"].values()))
    return out


def control() -> int:
    m = T.build(T.BASE_TONIC)
    bal = HB.run(m, 0, hdp(ab.RULE_NAME), DUR)[0]
    nul = HB.run(m, 0, hdp(s2.RULE_NAME, kappa_s=0.0), DUR)[0]
    a = HB.run(m, 0, hdp(s2.RULE_NAME, kappa_s=0.5), DUR)[0]
    b = HB.run(m, 0, hdp(s2.RULE_NAME, kappa_s=0.5), DUR)[0]
    ut = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         *[str(REPO / f"tests/{f}") for f in ("test_activity_balance.py", "test_activity_balance_sync.py",
                                                              "test_activity_balance_sync_fixref.py",
                                                              "test_activity_balance_sync2.py")]],
                        capture_output=True, text=True, cwd=REPO)
    res = {"null_kappa_s0_equals_balance": bool(np.array_equal(bal, nul)), "sync2_deterministic": bool(np.array_equal(a, b)),
           "unit_tests_pass": ut.returncode == 0,
           "unit_tests_tail": ut.stdout.strip().splitlines()[-1] if ut.stdout.strip() else ""}
    res["control_pass"] = bool(res["null_kappa_s0_equals_balance"] and res["sync2_deterministic"] and res["unit_tests_pass"])
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    me = str(Path(__file__).resolve())
    rc = subprocess.run([sys.executable, me, "select"]).returncode
    if rc != 0:
        print("SELECTION_FAILED rc", rc)
        return rc
    conds = list(T.CONDITIONS)
    for c in conds:
        if subprocess.run([sys.executable, me, "tune", c]).returncode != 0:
            print("FAILED tune", c)
            return 1
    keys = [f"{c}_seed{s}" for c in conds for s in TEST_SEEDS]
    for k in keys:
        if subprocess.run([sys.executable, me, "key", k]).returncode != 0:
            print("FAILED", k)
            return 1
    rec = {k: json.loads((OUT / f"{k}.json").read_text(encoding="utf-8")) for k in keys}
    tunes = {c: json.loads((OUT / f"tune_{c}.json").read_text(encoding="utf-8")) for c in conds}
    v = verdicts(rec, tunes)
    summary = {"spec": "docs/designs/H_SYNC2_TEST_spec.md", "spec_sha256": A.sha256_of(SPEC),
               "script_sha256": A.sha256_of(Path(__file__).resolve()),
               "rule_sha256": A.sha256_of(Path(s2.__file__).resolve()), "picked_kappa_s": picked(),
               "verdicts": v, "runs": sorted(rec)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({h: (x["all"] if isinstance(x, dict) and "all" in x else x) for h, x in v.items()}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run", "select", "tune", "key"])
    ap.add_argument("name", nargs="?")
    a = ap.parse_args(argv)
    return {"control": control, "run": run_all, "select": select}.get(a.cmd, lambda: (tune_cond if a.cmd == "tune" else run_key)(a.name))()


if __name__ == "__main__":
    raise SystemExit(main())
