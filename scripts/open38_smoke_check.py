"""OPEN-38 spectrum smoke check (SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md §7): ref seed 1 vs nc seed 1.

Usage: python scripts/open38_smoke_check.py REF_TAG NC_TAG [--results-dir D]

Reads <D>/<TAG>.json, <TAG>.npz, <TAG>_verdict.json for both tags and writes <D>/o38s_smoke_check.json.
Checks (each PASS/FAIL with its values): engine identity (JaxFNE 0.5.9 at 351f6048); realized N_SOM/N_PV per unit
within one cell of rho; ref r_ab in the T8 band; nc equals ref (every npz array, edges, sizes, rho); every array and
score finite; no stop state (unit E rate > 50 Hz, deep-unit E rate < 0.1 Hz, after the 2 s discard). Exit 0 only if
every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ENGINE_VERSION, ENGINE_SHA = "0.5.9", "351f6048"
# Read-only `git archive` export of 351f6048 (main-vwin, 2026-11-16); every blob verified against git tree
# cabfd2f9d68ab15453349983a04ebebd654466a9 (2783 files). An export has no .git, so its runs record no SHA: the
# loaded jaxfne file must lie under this root instead.
ENGINE_EXPORT = "D:/repos/_work/_temp_jaxfne_351f6048"
R_AB_BAND = (0.7964, 0.8855)  # T8 seed mean 0.8409 +- 3 x seed SD 0.0148 (scripts/open38_t8_thresholds.py)
DISCARD_S, RATE_MAX_HZ, DEEP_MIN_HZ, DEEP_UNIT = 2.0, 50.0, 0.1, "D"
SCORES = ("r_ab", "r_gm", "r_proto")


def ratio_ok(sizes: dict, rho: dict) -> dict:
    """Per unit: |N_SOM - N_inh * rho / (1 + rho)| <= 1 cell."""
    out = {}
    for u, s in sizes.items():
        n_inh = s["PV"] + s["SOM"]
        err = abs(s["SOM"] - n_inh * rho[u] / (1.0 + rho[u]))
        out[u] = {"PV": s["PV"], "SOM": s["SOM"], "rho": rho[u], "err_cells": err, "ok": err <= 1.0 and s["PV"] > 0
                  and s["SOM"] > 0}
    return out


def e_rates(counts: np.ndarray, keys: list[str], sizes: dict, dt_ms: float) -> dict:
    """Mean E rate (Hz) per unit after the discard."""
    k0 = int(round(DISCARD_S * 1000.0 / dt_ms))
    span_s = (counts.shape[0] - k0) * dt_ms / 1000.0
    return {u: float(counts[k0:, keys.index(f"{u}_E")].sum() / (sizes[u]["E"] * span_s)) for u in sizes}


def stop_state(rates: dict) -> str | None:
    if any(r > RATE_MAX_HZ for r in rates.values()):
        return f"unit E rate > {RATE_MAX_HZ} Hz"
    if rates.get(DEEP_UNIT, 1.0) < DEEP_MIN_HZ:
        return f"deep-unit E rate < {DEEP_MIN_HZ} Hz"
    return None


def load(d: Path, tag: str):
    rec = json.loads((d / f"{tag}.json").read_text(encoding="utf-8"))
    ver = json.loads((d / f"{tag}_verdict.json").read_text(encoding="utf-8"))
    with np.load(d / f"{tag}.npz") as z:
        arrs = {k: z[k] for k in z.files}
    return rec, ver, arrs


def main(ref: str, nc: str, d: Path) -> int:
    checks = {}
    runs = {t: load(d, t) for t in (ref, nc)}
    for t, (rec, ver, arrs) in runs.items():
        src = (rec.get("jaxfne_git_sha") or "") + (rec.get("jaxfne_source") or "")
        jfile = (rec.get("jaxfne_file") or "").replace("\\", "/").lower()
        from_export = jfile.startswith(ENGINE_EXPORT.lower() + "/")
        checks[f"engine:{t}"] = {"version": rec["jaxfne_version"], "sha_field": rec.get("jaxfne_git_sha"),
                                 "jaxfne_file": rec.get("jaxfne_file"), "from_verified_export": from_export,
                                 "ok": rec["jaxfne_version"] == ENGINE_VERSION and (ENGINE_SHA in src or from_export)}
        rr = ratio_ok(rec["sizes"], rec["rho"])
        checks[f"ratio:{t}"] = {"units": rr, "ok": all(v["ok"] for v in rr.values())}
        fin = all(np.all(np.isfinite(a)) for a in arrs.values() if a.dtype.kind == "f")
        fin = fin and all(isinstance(ver.get(s), (int, float)) and np.isfinite(ver[s]) for s in SCORES)
        checks[f"finite:{t}"] = {"ok": bool(fin)}
        keys = [str(k) for k in arrs["block_keys"]]
        rates = e_rates(arrs["counts"], keys, rec["sizes"], rec["dt_ms"])
        stop = stop_state(rates)
        checks[f"stop:{t}"] = {"e_rates_hz": rates, "stop": stop, "ok": stop is None}
    rref, vref, aref = runs[ref]
    rnc, _, anc = runs[nc]
    checks["ref_r_ab_band"] = {"r_ab": vref.get("r_ab"), "band": R_AB_BAND,
                               "ok": R_AB_BAND[0] <= vref.get("r_ab", np.nan) <= R_AB_BAND[1]}
    diff = sorted(k for k in set(aref) | set(anc) if k not in aref or k not in anc or not np.array_equal(aref[k], anc[k]))
    same_meta = all(rref[k] == rnc[k] for k in ("edges", "sizes", "rho", "seed", "T_s", "dt_ms"))
    checks["nc_equals_ref"] = {"differing_arrays": diff, "metadata_equal": same_meta, "ok": not diff and same_meta}
    ok = all(c["ok"] for c in checks.values())
    report = {"ref": ref, "nc": nc, "pass": ok, "checks": checks,
              "wall_s": {t: runs[t][0]["wall_s"] for t in runs}, "git_sha": rref.get("git_sha")}
    (d / "o38s_smoke_check.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"pass": ok, "failed": [k for k, c in checks.items() if not c["ok"]]}))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("nc")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "laminar_e2"))
    a = ap.parse_args()
    sys.exit(main(a.ref, a.nc, Path(a.results_dir)))
