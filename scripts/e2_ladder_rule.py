"""Apply the SEAL_LFS.md rule to the finite-size ladder.

Usage:
  python scripts/e2_ladder_rule.py

Rule (seal, not retuned): accept the smallest N whose seed-mean r_proto is within 2 seed-SD (of n4000) of n4000's
seed-mean and whose seed-mean per-unit LFP peak (sum row, `peak_hz_unit` in <TAG>_motif.json) is within 3 Hz of
n4000's for every unit. If none, T1 runs at n4000. Reads <TAG>_verdict.json and <TAG>_motif.json; n4000 seed 1 is
the earlier tag sw_n4000_dens_k1_kd1_sst15_s1_T8. Writes results/laminar_e2/lfs_rule.json and prints it.
Choices the seal leaves open, fixed before the results were read: both tests use <=; SD is n4000's seed SD with
ddof=1; seed-mean peaks are compared, not per-seed peaks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "laminar_e2"
NS = (1000, 2000, 4000)
SEEDS = (1, 2, 3)
R_SD, PEAK_HZ = 2.0, 3.0


def tag(n: int, s: int) -> str:
    return "sw_n4000_dens_k1_kd1_sst15_s1_T8" if (n, s) == (4000, 1) else f"lfs_n{n}_s{s}_T8"


def stats(n: int, d: Path = D) -> dict:
    rp, peaks = [], []
    for s in SEEDS:
        t = tag(n, s)
        rp.append(json.loads((d / f"{t}_verdict.json").read_text(encoding="utf-8"))["r_proto"])
        peaks.append(json.loads((d / f"{t}_motif.json").read_text(encoding="utf-8"))["sum_over_cells"]["peak_hz_unit"])
    units = list(peaks[0])
    return {"r_proto": rp, "mean": float(np.mean(rp)), "sd": float(np.std(rp, ddof=1)),
            "peak_mean": {u: float(np.mean([p[u] for p in peaks])) for u in units},
            "peaks": peaks}


def rule(d: Path = D) -> dict:
    st = {n: stats(n, d) for n in NS}
    ref = st[4000]
    rows = {}
    for n in NS[:-1]:
        dr = abs(st[n]["mean"] - ref["mean"])
        dp = {u: abs(st[n]["peak_mean"][u] - ref["peak_mean"][u]) for u in ref["peak_mean"]}
        rows[n] = {"r_ok": dr <= R_SD * ref["sd"], "d_r_proto": dr, "two_sd": R_SD * ref["sd"],
                   "d_peak_hz": dp, "peak_ok": all(v <= PEAK_HZ for v in dp.values())}
        rows[n]["accept"] = rows[n]["r_ok"] and rows[n]["peak_ok"]
    ok = [n for n in NS[:-1] if rows[n]["accept"]]
    out = {"stats": {str(n): st[n] for n in NS}, "tests": {str(n): rows[n] for n in rows},
           "t1_size": min(ok) if ok else 4000}
    (d / "lfs_rule.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = rule()
    print(json.dumps({"t1_size": o["t1_size"], "tests": o["tests"],
                      "means": {n: (s["mean"], s["sd"], s["peak_mean"]) for n, s in o["stats"].items()}}))
