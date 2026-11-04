"""Balance + synchrony H rule vs balance-only vs baseline (spec: docs/designs/H_SYNC_TEST_spec.md).

Usage:
  python scripts/h_sync_test.py control    # null wiring, determinism, engagement, unit tests
  python scripts/h_sync_test.py run        # arms 1 to 4, conditions A and B, seeds 0 and 1
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

SPEC = REPO / "docs/designs/H_SYNC_TEST_spec.md"
OUT = REPO / "docs/receipts/h_sync_test"
BAL_RECEIPTS = REPO / "docs/receipts/h_balance_test"
TUNE_SEED = 2
RATE_MATCH_TOL = 0.03
TUNE_ITERS = 8


def sync_params(kappa_s: float | None = None) -> dict:
    rp = {} if kappa_s is None else {"kappa_s": kappa_s}
    return {"hdp_rule": sy.RULE_NAME, "hdp_rule_params": rp}


def rho_bins(spikes, bin_ms: float) -> float:
    """Mean pairwise correlation of binned counts; same pair rule as the runner."""
    S = np.asarray(spikes, dtype=np.float64)
    n_steps, n = S.shape
    n_bins = round(T.DUR / bin_ms)
    B = S.reshape(n_bins, n_steps // n_bins, n).sum(axis=1)
    rng = np.random.default_rng(A.RHO_PAIR_SEED + 0)
    pi = rng.integers(0, n, A.RHO_PAIRS)
    pj = rng.integers(0, n - 1, A.RHO_PAIRS)
    pj = pj + (pj >= pi)
    Bc = B - B.mean(axis=0)
    sd = np.sqrt((Bc ** 2).sum(axis=0))
    den = sd[pi] * sd[pj]
    num = (Bc[:, pi] * Bc[:, pj]).sum(axis=0)
    ok = den > 0
    return float((num[ok] / den[ok]).mean()) if ok.any() else float("nan")


def metrics(spikes, cells) -> dict:
    m = T.metrics(spikes, cells, T.DUR)
    m["mean_rho_10ms_bins"] = rho_bins(spikes, 10.0)
    return m


def control() -> int:
    m = T.build(T.BASE_TONIC)
    b1, _ = HB.run(m, 0, None, 200.0)
    b2, _ = HB.run(m, 0, None, 200.0)
    bal, _ = HB.run(m, 0, HB.hdp_params(), 200.0)
    nul, _ = HB.run(m, 0, sync_params(0.0), 200.0)
    bal_full, _ = HB.run(m, 0, HB.hdp_params())
    syn_full, diag = HB.run(m, 0, sync_params())
    ut = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         str(REPO / "tests/test_activity_balance.py"),
                         str(REPO / "tests/test_activity_balance_sync.py")],
                        capture_output=True, text=True, cwd=REPO)
    H = np.asarray(diag["H_final"], dtype=np.float64)
    res = {"baseline_deterministic": bool(np.array_equal(b1, b2)),
           "null_kappa_s0_equals_balance_only": bool(np.array_equal(bal, nul)),
           "engaged_differs_from_balance_only": bool(not np.array_equal(bal_full, syn_full)),
           "H_sd": float(H.std()), "unit_tests_pass": ut.returncode == 0,
           "unit_tests_tail": ut.stdout.strip().splitlines()[-1] if ut.stdout.strip() else ""}
    res["control_pass"] = bool(res["baseline_deterministic"] and res["null_kappa_s0_equals_balance_only"]
                               and res["engaged_differs_from_balance_only"] and res["H_sd"] > 0
                               and res["unit_tests_pass"])
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def tune_rate_matched() -> dict:
    """Bisection on tonic_drive_scale (HDP off) so bar_r matches balance-only, B, seed TUNE_SEED."""
    m_b = T.build(T.CONDITIONS["B_stress_x1p5"])
    sp, _ = HB.run(m_b, TUNE_SEED, HB.hdp_params())
    target = float(sp.sum() / sp.shape[1] / (T.DUR / 1000.0))
    lo, hi = T.BASE_TONIC, T.CONDITIONS["B_stress_x1p5"]
    its, best = [], None
    for k in range(TUNE_ITERS):
        mid = 0.5 * (lo + hi)
        s, _ = HB.run(T.build(mid), TUNE_SEED, None)
        r = float(s.sum() / s.shape[1] / (T.DUR / 1000.0))
        its.append({"iter": k, "tonic": mid, "bar_r": r})
        if best is None or abs(r - target) < abs(best["bar_r"] - target):
            best = its[-1]
        if r < target:
            lo = mid
        else:
            hi = mid
    rel = abs(best["bar_r"] - target) / target
    return {"target_bar_r": target, "iterations": its, "tonic": best["tonic"],
            "achieved_bar_r": best["bar_r"], "rel_error": rel, "matched": bool(rel <= RATE_MATCH_TOL)}


def verdicts(rec: dict, tune: dict) -> dict:
    ks = sorted(rec)
    g = lambda k, arm, f: rec[k][arm][f]  # noqa: E731
    out = {
        "H5_fano_sync_lt_balance": {k: bool(g(k, "sync", "pop_synchrony_fano_10ms")
                                           < g(k, "balance", "pop_synchrony_fano_10ms")) for k in ks},
        "H6_rho_sync_lt_balance": {k: bool(g(k, "sync", "mean_rho") < g(k, "balance", "mean_rho")) for k in ks},
        "H7_barr_sync_lt_baseline": {k: bool(g(k, "sync", "bar_r_Hz") < g(k, "baseline", "bar_r_Hz")) for k in ks},
        "H8_fano_sync_le_baseline_A": {k: bool(g(k, "sync", "pop_synchrony_fano_10ms")
                                              <= g(k, "baseline", "pop_synchrony_fano_10ms"))
                                       for k in ks if k.startswith("A_")},
    }
    for h in list(out):
        out[h]["all"] = all(out[h].values())
    if tune["matched"]:
        rm = [rec[k]["rate_matched"]["mean_rho"] for k in ks if k.startswith("B_")]
        bal = [rec[k]["balance"]["mean_rho"] for k in ks if k.startswith("B_")]
        spread = abs(rm[0] - rm[1])
        out["H9_balance_rho_gt_rate_matched_B"] = {"balance_rho": bal, "rate_matched_rho": rm,
                                                  "rm_seed_spread": spread,
                                                  "all": bool(min(bal) - max(rm) > spread)}
    else:
        out["H9_balance_rho_gt_rate_matched_B"] = {"all": None, "note": "rate matching failed"}
    return out


def run_all() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tune = tune_rate_matched()
    print("tuning", {k: tune[k] for k in ("target_bar_r", "tonic", "achieved_bar_r", "rel_error", "matched")},
          flush=True)
    rec, repro = {}, {}
    for cname, tonic in T.CONDITIONS.items():
        m = T.build(tonic)
        cells = np.array([r["cell_type"] for r in m.neuron_table()])
        m_rm = T.build(tune["tonic"]) if cname.startswith("B_") else None
        for seed in (0, 1):
            key = f"{cname}_seed{seed}"
            arms = {"baseline": HB.run(m, seed, None)[0],
                    "balance": HB.run(m, seed, HB.hdp_params())[0]}
            sp, diag = HB.run(m, seed, sync_params())
            arms["sync"] = sp
            if m_rm is not None:
                arms["rate_matched"] = HB.run(m_rm, seed, None)[0]
            r = {a: metrics(s, cells) for a, s in arms.items()}
            rates = sp.sum(axis=0) / (T.DUR / 1000.0)
            r["H_state_sync"] = T.h_stats(diag, rates, cells)
            r["b_state_sync"] = HB.b_stats(diag)
            r["rule_params"] = dict(sy.DEFAULT_PARAMS)
            rec[key] = r
            old = json.loads((BAL_RECEIPTS / f"{key}.json").read_text(encoding="utf-8"))["balance"]
            repro[key] = bool(all(old[f] == r["balance"][f] for f in
                                  ("bar_r_Hz", "mean_rho", "pop_synchrony_fano_10ms")))
            (OUT / f"{key}.json").write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            print(key, " | ".join("%s r %.2f rho %.3f fano %.2f" % (a, r[a]["bar_r_Hz"], r[a]["mean_rho"],
                                                                    r[a]["pop_synchrony_fano_10ms"])
                                  for a in arms), flush=True)
    v = verdicts(rec, tune)
    summary = {"spec": "docs/designs/H_SYNC_TEST_spec.md", "spec_sha256": A.sha256_of(SPEC),
               "script_sha256": A.sha256_of(Path(__file__).resolve()),
               "rule_sha256": A.sha256_of(Path(sy.__file__).resolve()),
               "balance_rule_sha256": A.sha256_of(Path(ab.__file__).resolve()),
               "balance_arm_reproduces_receipts": repro, "rate_matching": tune, "verdicts": v,
               "runs": sorted(rec)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reproduces": repro, "verdicts": {k: x["all"] for k, x in v.items()}}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["control", "run"])
    a = ap.parse_args(argv)
    return control() if a.cmd == "control" else run_all()


if __name__ == "__main__":
    raise SystemExit(main())
