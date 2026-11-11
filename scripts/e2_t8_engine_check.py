"""T8 engine check (SEAL_T8): T8 ref runs vs the saved dense-delivery reference runs, array by array.

Usage: python scripts/e2_t8_engine_check.py OUT_JSON

For seed s, t8_ref_n4000_s{s}_T8.npz is compared with the dense run of the same layout and seed; per pair the
arrays present in both, those bitwise equal, and for the others the largest absolute difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "laminar_e2"
PAIRS = {1: "sw_n4000_dens_k1_kd1_sst15_s1_T8", 2: "lfs_n4000_s2_T8", 3: "lfs_n4000_s3_T8"}


def compare(a: Path, b: Path) -> dict:
    za, zb = np.load(a), np.load(b)
    common = sorted(set(za.files) & set(zb.files))
    diff = {}
    for k in common:
        x, y = za[k], zb[k]
        if x.shape != y.shape:
            diff[k] = f"shape {x.shape} vs {y.shape}"
        elif not np.array_equal(x, y):
            diff[k] = float(np.max(np.abs(x.astype(float) - y.astype(float)))) if x.dtype.kind in "fiu" else "differs"
    return {"n_common": len(common), "n_equal": len(common) - len(diff), "differing": diff,
            "only_t8": sorted(set(za.files) - set(zb.files)), "only_dense": sorted(set(zb.files) - set(za.files))}


def main(out: Path) -> dict:
    res = {}
    for s, dense in PAIRS.items():
        jd = json.loads((D / f"{dense}.json").read_text(encoding="utf-8"))
        res[s] = {"t8": f"t8_ref_n4000_s{s}_T8", "dense": dense, "dense_layout": jd["layout"], "dense_seed": jd["seed"],
                  "dense_delivery": jd.get("edge_delivery", "dense (field absent)"),
                  **compare(D / f"t8_ref_n4000_s{s}_T8.npz", D / f"{dense}.npz")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    for s, r in main(Path(sys.argv[1])).items():
        print(s, r["dense"], r["dense_layout"], r["dense_seed"], r["dense_delivery"],
              f"{r['n_equal']}/{r['n_common']} equal", r["differing"], r["only_t8"], r["only_dense"])
