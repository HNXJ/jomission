"""Per-run rates, stop states and the T1 axis selection (docs/laminar_motif/SEAL_T1.md).

Usage:
  python scripts/e2_select.py AXIS N [--ref-level L]     AXIS in kd | sst | sstd

Reads results/laminar_e2/t1_{AXIS}{level}_n{N}_s{seed}_T8.{npz,json} and _verdict.json for the seal's levels
and seeds 1-3. Stop state (seal): any unit E rate > 50 Hz, or the deep unit's E rate < 0.1 Hz, after the 2 s
discard. A level counts only if all 3 seeds exist, have a verdict and hit no stop state. Winner = highest
seed-mean r_proto among counted levels; ties go to the reference level. Passes = seeds whose verdict passes.
Prints one JSON line; writes results/laminar_e2/t1_{AXIS}_n{N}_selection.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "laminar_e2"
DISCARD_S = 2.0
LEVELS = {"kd": ["0p25", "0p35", "0p5", "0p71", "1", "1p41", "2"],
          "sst": ["5p3", "7p5", "10p6", "15", "21", "30", "42"],
          "sstd": ["7p5", "10p6", "15", "21", "30", "42", "60"]}
REF = {"kd": "1", "sst": "15", "sstd": "15"}
E_MAX_HZ, DEEP_E_MIN_HZ = 50.0, 0.1


def rates(tag: str, discard_s: float = DISCARD_S, d: Path = D) -> dict[str, dict[str, float]] | None:
    """Mean rate (Hz per cell) per unit and cell type after the discard; None if the run is missing or short."""
    npz, js = d / f"{tag}.npz", d / f"{tag}.json"
    if not (npz.exists() and js.exists()):
        return None
    rec = json.loads(js.read_text(encoding="utf-8"))
    z = np.load(npz)
    c = z["counts"][int(round(discard_s * 1000 / rec["dt_ms"])):]
    if c.shape[0] == 0:
        return None
    T = c.shape[0] * rec["dt_ms"] / 1000
    out: dict[str, dict[str, float]] = {}
    for k, key in enumerate(z["block_keys"]):
        u, t = str(key).split("_")
        out.setdefault(u, {})[t] = float(c[:, k].sum() / T / rec["sizes"][u][t])
    return out


def stop_state(r: dict[str, dict[str, float]], deep: str = "D") -> str | None:
    hot = [u for u in r if r[u]["E"] > E_MAX_HZ]
    if hot:
        return f"E rate > {E_MAX_HZ} Hz in {hot}"
    if deep in r and r[deep]["E"] < DEEP_E_MIN_HZ:
        return f"deep E rate < {DEEP_E_MIN_HZ} Hz"
    return None


def select(axis: str, n: int, ref: str) -> dict:
    rows = []
    for lv in LEVELS[axis]:
        seeds = []
        for s in (1, 2, 3):
            tag = f"t1_{axis}{lv}_n{n}_s{s}_T8"
            vp = D / f"{tag}_verdict.json"
            r = rates(tag)
            v = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else None
            seeds.append({"seed": s, "tag": tag, "rates": r, "stop": None if r is None else stop_state(r),
                          "r_proto": None if v is None else v["r_proto"], "pass": None if v is None else v["pass"]})
        ok = all(x["rates"] is not None and x["r_proto"] is not None and x["stop"] is None for x in seeds)
        rp = [x["r_proto"] for x in seeds if x["r_proto"] is not None]
        rows.append({"level": lv, "counts": ok, "seeds": seeds,
                     "r_proto_mean": float(np.mean(rp)) if rp else None,
                     "r_proto_sd": float(np.std(rp, ddof=1)) if len(rp) > 1 else None,
                     "n_pass": sum(bool(x["pass"]) for x in seeds)})
    counted = [r for r in rows if r["counts"]]
    win = None
    if counted:
        best = max(r["r_proto_mean"] for r in counted)
        tied = [r for r in counted if r["r_proto_mean"] == best]
        win = next((r for r in tied if r["level"] == ref), tied[0])
    out = {"axis": axis, "n": n, "ref_level": ref, "levels": rows,
           "winner": None if win is None else win["level"],
           "winner_passes_2of3": None if win is None else win["n_pass"] >= 2}
    (D / f"t1_{axis}_n{n}_selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("axis", choices=list(LEVELS))
    ap.add_argument("n", type=int)
    ap.add_argument("--ref-level", default=None)
    a = ap.parse_args()
    o = select(a.axis, a.n, a.ref_level or REF[a.axis])
    print(json.dumps({k: o[k] for k in ("axis", "n", "winner", "winner_passes_2of3")} |
                     {"table": [(r["level"], r["counts"], r["r_proto_mean"], r["n_pass"]) for r in o["levels"]]}))
