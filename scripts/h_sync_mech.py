"""Mechanism test for the sync term's failure (spec: docs/designs/H_SYNC_MECH_spec.md).

Usage:
  python scripts/h_sync_mech.py control   # detector positive/negative controls, probe unit tests
  python scripts/h_sync_mech.py run       # 15 s runs: baseline, balance-only, sync x3 time scales, fixref
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

SPEC = REPO / "docs/designs/H_SYNC_MECH_spec.md"
OUT = REPO / "docs/receipts/h_sync_mech"
DUR = 15000.0
WIN_START = 3000.0
BIN_MS = 50.0
BAND_HZ = (0.2, 5.0)
SCALES = (0.3, 1.0, 3.0)
N_SUR = 1000
ALPHA = 0.05
SUR_SEED = 12345
N = 400


def pop_series(spikes) -> np.ndarray:
    """Population rate (Hz) per BIN_MS bin over the analysis window."""
    k = round(BIN_MS / T.DT)
    s = np.asarray(spikes, dtype=np.float64)[round(WIN_START / T.DT):]
    n_bins = s.shape[0] // k
    cnt = s[: n_bins * k].sum(axis=1).reshape(n_bins, k).sum(axis=1)
    return cnt / (s.shape[1] * BIN_MS / 1000.0)


def _detrend(x):
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, 1), t)


def band_stat(x, bin_ms: float = BIN_MS, band=BAND_HZ):
    """Peak-to-mean ratio of the Hann-windowed periodogram in band; returns (S, f_peak_Hz)."""
    x = _detrend(np.asarray(x, dtype=np.float64))
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), d=bin_ms / 1000.0)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or P[m].mean() == 0:
        return float("nan"), float("nan")
    i = int(np.argmax(P[m]))
    return float(P[m][i] / P[m].mean()), float(f[m][i])


def ar1_surrogates(x, n, rng):
    """AR(1) series matched to variance and lag-1 autocorrelation of x (linear, non-oscillatory null)."""
    x = _detrend(np.asarray(x, dtype=np.float64))
    sd = x.std()
    phi_hat = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    # lag-1 estimate is biased low after detrending (about -0.03 at L=240); first-order correction
    phi = float(np.clip(phi_hat + 2.0 * (1.0 + 3.0 * phi_hat) / len(x), 0.0, 0.99))
    e = rng.standard_normal((n, len(x))) * sd * np.sqrt(1 - phi ** 2)
    y = np.zeros_like(e)
    y[:, 0] = rng.standard_normal(n) * sd
    for t in range(1, len(x)):
        y[:, t] = phi * y[:, t - 1] + e[:, t]
    return y


def detect(x, rng, n_sur: int = N_SUR) -> dict:
    S, fpk = band_stat(x)
    sur = np.array([band_stat(y)[0] for y in ar1_surrogates(x, n_sur, rng)])
    p = float((1 + np.sum(sur >= S)) / (1 + n_sur))
    return {"S": S, "f_peak_Hz": fpk, "period_s": (1.0 / fpk if fpk and fpk > 0 else float("nan")),
            "p_ar1": p, "oscillation": bool(p < ALPHA)}


def stats_of(spikes, rng) -> dict:
    x = pop_series(spikes)
    d = detect(x, rng)
    d.update({"window_bar_r_Hz": float(x.mean()),
              "p2p_rel": float((np.quantile(x, 0.95) - np.quantile(x, 0.05)) / x.mean()) if x.mean() > 0 else float("nan"),
              "series_Hz_50ms": [round(float(v), 3) for v in x]})
    return d


def detector_controls() -> dict:
    rng = np.random.default_rng(SUR_SEED)
    L = round((DUR - WIN_START) / BIN_MS)
    t = np.arange(L) * BIN_MS / 1000.0
    saw = ((t / 1.5) % 1.0) - 0.5            # relaxation-like sawtooth, period 1.5 s
    pos_hits, neg_hits = 0, {0.5: 0, 0.9: 0}
    n_draw, n_sur = 100, 200
    for _ in range(n_draw):
        noise = ar1_surrogates(np.random.default_rng(int(rng.integers(1 << 31))).standard_normal(L), 1, rng)[0]
        x = noise / noise.std() * saw.std() + saw       # equal SD of signal and noise
        pos_hits += detect(x, rng, n_sur)["oscillation"]
    for phi in neg_hits:
        for _ in range(n_draw):
            y = np.zeros(L)
            e = rng.standard_normal(L)
            for i in range(1, L):
                y[i] = phi * y[i - 1] + e[i]
            neg_hits[phi] += detect(y, rng, n_sur)["oscillation"]
    res = {"positive_detection_rate": pos_hits / n_draw,
           "negative_false_positive_rate": {str(k): v / n_draw for k, v in neg_hits.items()}}
    res["pass"] = bool(res["positive_detection_rate"] >= 0.9
                       and all(v <= 0.10 for v in res["negative_false_positive_rate"].values()))
    return res


def control() -> int:
    dc = detector_controls()
    ut = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         str(REPO / "tests/test_activity_balance.py"),
                         str(REPO / "tests/test_activity_balance_sync.py"),
                         str(REPO / "tests/test_activity_balance_sync_fixref.py")],
                        capture_output=True, text=True, cwd=REPO)
    res = {"detector": dc, "unit_tests_pass": ut.returncode == 0,
           "unit_tests_tail": ut.stdout.strip().splitlines()[-1] if ut.stdout.strip() else ""}
    res["control_pass"] = bool(dc["pass"] and res["unit_tests_pass"])
    print(json.dumps(res, indent=1))
    return 0 if res["control_pass"] else 1


def hdp(rule: str, **rp) -> dict:
    # record_weight_trace False: recording only (spikes verified identical at 2000 ms); the 15 s w_trace is ~6 GB
    return {"hdp_rule": rule, "hdp_rule_params": rp, "record_weight_trace": False}


def verdicts(rec: dict) -> dict:
    keys = sorted(rec)
    h1_sync = {k: rec[k]["sync_s1"]["oscillation"] for k in keys}
    h1_bal = {k: rec[k]["balance"]["oscillation"] for k in keys}
    h1 = bool(all(h1_sync.values()) and sum(h1_bal.values()) <= 1)
    h2_cases, slopes = {}, []
    for k in keys:
        pers = [rec[k][f"sync_s{s:g}"]["period_s"] for s in SCALES]
        det = [rec[k][f"sync_s{s:g}"]["oscillation"] for s in SCALES]
        ok = bool(all(det) and pers[0] < pers[1] < pers[2])
        h2_cases[k] = {"periods_s": pers, "detected": det, "monotone_increasing": ok}
        if all(det):
            slopes.append(float(np.polyfit(np.log(SCALES), np.log(pers), 1)[0]))
    med = float(np.median(slopes)) if slopes else float("nan")
    h2 = bool(sum(c["monotone_increasing"] for c in h2_cases.values()) >= 3 and 0.5 <= med <= 1.5)
    r0 = {k: rec[k]["sync_s1"]["window_bar_r_Hz"] / rec[k]["balance"]["window_bar_r_Hz"] for k in keys}
    r1 = {k: rec[k]["fixref"]["window_bar_r_Hz"] / rec[k]["balance"]["window_bar_r_Hz"] for k in keys}
    ratio_blowup = bool(all(v <= 0.5 for v in r0.values()) and all(v >= 0.8 for v in r1.values()))
    denom_independent = bool(all(v <= 0.5 for v in r1.values()))
    if h1 and h2:
        cls = "RELAXATION_OSCILLATION_SUPPORTED"
    elif ratio_blowup and not h1:
        cls = "RATIO_BLOWUP_MONOSTABLE_COLLAPSE"
    elif not h1:
        cls = "NO_OSCILLATION_DETECTED_OTHER_MECHANISM"
    else:
        cls = "OSCILLATION_PRESENT_BUT_PERIOD_DOES_NOT_SCALE"
    return {"HM1_oscillation_in_sync_not_balance": {"sync": h1_sync, "balance": h1_bal, "all": h1},
            "HM2_period_scales_with_timescale": {"cases": h2_cases, "median_loglog_slope": med, "all": h2},
            "HM3_denominator": {"rate_ratio_sync_over_balance": r0, "rate_ratio_fixref_over_balance": r1,
                                "ratio_blowup_explains_collapse": ratio_blowup,
                                "collapse_independent_of_denominator": denom_independent},
            "classification": cls}


def run_key(key: str) -> int:
    """One (condition, seed) in its own process (memory: 15 s traces of 400 neurons)."""
    OUT.mkdir(parents=True, exist_ok=True)
    cname, seed = key.rsplit("_seed", 1)
    seed = int(seed)
    rng = np.random.default_rng(SUR_SEED + 1000 * list(T.CONDITIONS).index(cname) + seed)
    if True:
        tonic = T.CONDITIONS[cname]
        m = T.build(tonic)
        if True:
            base, _ = HB.run(m, seed, None, DUR)
            p_ref = float(pop_series(base).mean())
            r = {"p_ref_Hz": p_ref, "baseline": stats_of(base, rng)}
            r["balance"] = stats_of(HB.run(m, seed, hdp(ab.RULE_NAME), DUR)[0], rng)
            for s in SCALES:
                sp, _ = HB.run(m, seed, hdp(sy.RULE_NAME, tau_r=1000.0 * s, tau_H=1000.0 * s), DUR)
                r[f"sync_s{s:g}"] = stats_of(sp, rng)
            r["fixref"] = stats_of(HB.run(m, seed, hdp(fx.RULE_NAME, p_ref=p_ref), DUR)[0], rng)
            (OUT / f"{key}.json").write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            print(key, " | ".join("%s r %.2f p %.3f T %.2fs p2p %.2f" % (
                a, r[a]["window_bar_r_Hz"], r[a]["p_ar1"], r[a]["period_s"], r[a]["p2p_rel"])
                for a in r if a not in ("p_ref_Hz", "baseline")), flush=True)
    return 0


def run_all() -> int:
    keys = [f"{c}_seed{s}" for c in T.CONDITIONS for s in (0, 1)]
    for k in keys:
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "key", k]).returncode
        if rc != 0:
            print("FAILED", k, rc)
            return rc
    rec = {k: json.loads((OUT / f"{k}.json").read_text(encoding="utf-8")) for k in keys}
    v = verdicts({k: {a: {f: x[f] for f in ("oscillation", "period_s", "window_bar_r_Hz")}
                      for a, x in r.items() if a != "p_ref_Hz"} for k, r in rec.items()})
    summary = {"spec": "docs/designs/H_SYNC_MECH_spec.md", "spec_sha256": A.sha256_of(SPEC),
               "script_sha256": A.sha256_of(Path(__file__).resolve()),
               "rule_sha256": A.sha256_of(Path(sy.__file__).resolve()),
               "probe_sha256": A.sha256_of(Path(fx.__file__).resolve()),
               "verdicts": v, "runs": sorted(rec)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"classification": v["classification"], "HM1": v["HM1_oscillation_in_sync_not_balance"]["all"],
                      "HM2": v["HM2_period_scales_with_timescale"]["all"],
                      "median_slope": v["HM2_period_scales_with_timescale"]["median_loglog_slope"],
                      "HM3": v["HM3_denominator"]}, indent=1))
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
