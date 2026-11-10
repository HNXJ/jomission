"""Import the observed LANSA spectrolaminar maps Y (21 bins x 150 frequencies x 11 areas) with a source digest.

Usage: python scripts/import_lansa_lfp.py <spectrolaminar_data.mat> [--out docs/laminar_motif/data/lansa_lfp_Y.json]
Reads ephys_data (relative power RP(bin, f) = P / max_bin P per area; frequency index k is k Hz, 1..150, as
LANSA_Analysis_Pipeline.m bands 1:3 ... 91:150), checks shape, finiteness and range, writes one JSON. Area order
is that of lansa_density_X.json (the pipeline indexes both files with the same area axis: INFERRED from
LANSA_Analysis_Pipeline.m and lansa_port.py). Private data (owner ruling 2026-11-06). Exit 0 when written; 1 on a
failed check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mat", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/laminar_motif/data/lansa_lfp_Y.json")
    a = ap.parse_args()
    raw = a.mat.read_bytes()
    Y = np.asarray(sio.loadmat(a.mat)["ephys_data"], dtype=float)
    X = json.loads((ROOT / "docs/laminar_motif/data/lansa_density_X.json").read_text(encoding="utf-8"))
    if Y.shape != (21, 150, 11) or not np.isfinite(Y).all() or Y.min() <= 0 or Y.max() > 1.0 + 1e-12:
        print("FAIL shape/finite/range", Y.shape, float(np.nanmin(Y)), float(np.nanmax(Y)))
        return 1
    rec = {
        "schema": "jomission.lansa_lfp_Y.v1",
        "source": {"file": a.mat.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                   "pipeline": "LANSA_Analysis_Pipeline.m (ephys_data: 21 bins x 150 frequencies x 11 areas, relative power)"},
        "axes": {"bin_center_rel_L4": X["axes"]["bin_center_rel_L4"], "freq_hz": list(range(1, 151)),
                 "areas": X["axes"]["areas"]},
        "units": "relative power P(bin, f) / max over bins, per area and frequency",
        "Y": np.round(Y, 6).tolist(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec), encoding="utf-8")
    print("wrote", a.out.name, "shape", Y.shape, "sha256", rec["source"]["sha256"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
