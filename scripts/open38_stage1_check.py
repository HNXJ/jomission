"""OPEN-38 positive-control stage (owner 2026-11-16: ref, flat, rev x seeds 1-5) against the frozen criteria of
SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md §5-§7. No threshold here is derived from o38s outcomes.

Usage: python scripts/open38_stage1_check.py OUT_JSON [--results-dir D]

Per run (o38s_pc_{arm}_n4000_s{seed}_T8): engine, realized ratio, finite, stop state (open38_smoke_check functions).
Gates:
  ref_band   §5(b): r_ab of ref seeds 1-3 each in [0.7964, 0.8855]; seeds 4-5 reported against the same band
  flat, rev  §5(c), SEAL_T8 rule unchanged at n = 5: every d_s = r_ab(ref, s) - r_ab(arm, s) > 0 and
             mean d > 2 SD / sqrt(n)
  runs       every run present, engine from the verified export, ratio within 1 cell, finite, no stop state
Also reported, not gated: paired 95% t-interval of d (r_ab, r_gm, r_proto), crossing_2se and scorer pass per run.
A missing run makes its arm "not evaluable" (no replacement seed). Exit 0 only if every gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open38_smoke_check as sc  # noqa: E402

ARMS, SEEDS = ("ref", "flat", "rev"), (1, 2, 3, 4, 5)
BAND_SEEDS = (1, 2, 3)
SCORES = ("r_ab", "r_gm", "r_proto")


def tag(arm: str, seed: int) -> str:
    return f"o38s_pc_{arm}_n4000_s{seed}_T8"  # pc: positive-control stage; smoke tags o38s_{ref,nc}_... are kept apart


def run_checks(d: Path, t: str) -> dict:
    try:
        rec, ver, arrs = sc.load(d, t)
    except FileNotFoundError as e:
        return {"present": False, "error": str(e), "ok": False}
    jfile = (rec.get("jaxfne_file") or "").replace("\\", "/").lower()
    engine = rec["jaxfne_version"] == sc.ENGINE_VERSION and jfile.startswith(sc.ENGINE_EXPORT.lower() + "/")
    ratio = sc.ratio_ok(rec["sizes"], rec["rho"])
    fin = all(np.all(np.isfinite(a)) for a in arrs.values() if a.dtype.kind == "f") and all(
        isinstance(ver.get(s), (int, float)) and math.isfinite(ver[s]) for s in SCORES)
    rates = sc.e_rates(arrs["counts"], [str(k) for k in arrs["block_keys"]], rec["sizes"], rec["dt_ms"])
    stop = sc.stop_state(rates)
    ok = engine and all(v["ok"] for v in ratio.values()) and fin and stop is None
    return {"present": True, "engine": engine, "ratio": ratio, "finite": bool(fin), "e_rates_hz": rates, "stop": stop,
            "git_sha": rec.get("git_sha"), "git_dirty": rec.get("git_dirty"), "layout": rec.get("layout"),
            **{s: ver.get(s) for s in (*SCORES, "crossing_2se", "pass")}, "ok": bool(ok)}


def t8_rule(d: list[float]) -> dict:
    n, m = len(d), float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    ok = all(x > 0 for x in d) and m > 2 * sd / math.sqrt(n)
    return {"d_s": d, "mean": m, "sd": sd, "threshold": 2 * sd / math.sqrt(n), "ref_higher": ok}


def interval(d: list[float]) -> list[float]:
    n, m, sd = len(d), float(np.mean(d)), float(np.std(d, ddof=1))
    h = student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return [m - h, m + h]


def main(out: str, d: Path) -> int:
    runs = {a: {s: run_checks(d, tag(a, s)) for s in SEEDS} for a in ARMS}
    gates = {"runs": {"ok": all(r["ok"] for a in ARMS for r in runs[a].values()),
                      "failed": [tag(a, s) for a in ARMS for s in SEEDS if not runs[a][s]["ok"]]}}
    ref_ab = {s: runs["ref"][s].get("r_ab") for s in SEEDS}
    inb = {s: v is not None and sc.R_AB_BAND[0] <= v <= sc.R_AB_BAND[1] for s, v in ref_ab.items()}
    vals = [v for v in ref_ab.values() if v is not None]
    gates["ref_band"] = {"band": sc.R_AB_BAND, "r_ab": ref_ab, "in_band": inb,
                         "ok": all(inb[s] for s in BAND_SEEDS), "all_five_in_band": all(inb.values()),
                         "seed_mean": float(np.mean(vals)) if vals else None,
                         "seed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    for arm in ("flat", "rev"):
        if not all(runs[a][s]["present"] for a in ("ref", arm) for s in SEEDS):
            gates[arm] = {"ok": False, "status": "not evaluable (missing run)"}
            continue
        rule = t8_rule([runs["ref"][s]["r_ab"] - runs[arm][s]["r_ab"] for s in SEEDS])
        ci = {sc_: interval([runs["ref"][s][sc_] - runs[arm][s][sc_] for s in SEEDS]) for sc_ in SCORES}
        gates[arm] = {**rule, "ci95_ref_minus_arm": ci, "ok": rule["ref_higher"]}
    ok = all(g["ok"] for g in gates.values())
    report = {"pass": ok, "gates": gates, "runs": runs}
    Path(out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"pass": ok, "failed": [k for k, g in gates.items() if not g["ok"]]}))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "laminar_e2"))
    a = ap.parse_args()
    sys.exit(main(a.out, Path(a.results_dir)))
