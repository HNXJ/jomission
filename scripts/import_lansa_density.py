"""Import the LANSA relative density profiles X (21 bins x 4 types x 11 areas) with a source digest.

Usage: python scripts/import_lansa_density.py <LANSA_processed_cyto.mat> [--out docs/laminar_motif/data/lansa_density_X.json]
Reads the cytoarchitecture .mat, checks shape and finiteness, writes one JSON (values, axes, source sha256).
Private data (owner ruling 2026-11-06: jomission is private). Exit 0 when written; 1 on a failed check.
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
TYPES = ["PV", "CB", "CR", "NG"]  # axis 1 of anatomy_data (scripts in the private tree print it in this order)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mat", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/laminar_motif/data/lansa_density_X.json")
    a = ap.parse_args()
    raw = a.mat.read_bytes()
    c = sio.loadmat(a.mat)
    X = np.asarray(c["anatomy_data"], dtype=float)
    edges = np.asarray(c["standard_edges"], dtype=float).ravel()
    areas = [str(x[0]) for x in c["included_areas"][0]]
    if X.shape != (21, 4, 11) or len(areas) != 11 or edges.size != 21 or not np.isfinite(X).all():
        print("FAIL shape/finite", X.shape, len(areas), edges.size)
        return 1
    rec = {
        "schema": "jomission.lansa_density_X.v1",
        "source": {"file": a.mat.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                   "manuscript": "Laminar cellular architecture manuscript, Methods (histology, uniform space)"},
        "axes": {"bin_center_rel_L4": edges.tolist(), "types": TYPES, "areas": areas},
        "units": "cells/um3 relative laminar density (analysis scales per type); bins -0.6..0.4 of cortical depth around mid-L4 (negative = superficial)",
        "mapping_note": "CB maps to SST and CR to VIP by overlap (not measured directly); NG is the pyramidal proxy",
        "X": X.tolist(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec), encoding="utf-8")
    print("wrote", a.out.name, "shape", X.shape, "sha256", rec["source"]["sha256"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
