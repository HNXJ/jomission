"""Mean LANSA relative density per depth module (S, M, D) for each type and area.

Usage:
  python scripts/lansa_module_density.py

Module bins declared before reading values: M = mid-L4 +- 0.15 (7 bins), S = bins above (-0.6..-0.2,
9 bins), D = bins below (0.2..0.4, 5 bins). Mean over bins, so bin count does not bias a module.
Prints per-area S/D ratios of NG (pyramidal proxy), PV, CB, CR and total; writes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "docs/laminar_motif/data/lansa_density_X.json").read_text(encoding="utf-8"))
X = np.asarray(d["X"], dtype=float)  # [bin, type, area]
z = np.round(np.asarray(d["axes"]["bin_center_rel_L4"]), 3)
types, areas = d["axes"]["types"], d["axes"]["areas"]
groups = {"S": z < -0.175, "M": (z >= -0.175) & (z <= 0.175), "D": z > 0.175}
print("bins per module", {k: int(v.sum()) for k, v in groups.items()})
mean = {k: X[v].mean(axis=0) for k, v in groups.items()}  # [type, area]
cols = types + ["total"]
print("S/D ratio of mean density")
print("area   " + " ".join(f"{c:>6}" for c in cols))
rows = []
for a, name in enumerate(areas):
    r = [mean["S"][t, a] / mean["D"][t, a] for t in range(len(types))]
    r.append(mean["S"][:, a].sum() / mean["D"][:, a].sum())
    rows.append(r)
    print(f"{name:6} " + " ".join(f"{v:6.2f}" for v in r))
rows = np.array(rows)
print("median " + " ".join(f"{v:6.2f}" for v in np.median(rows, axis=0)))
print("min    " + " ".join(f"{v:6.2f}" for v in rows.min(axis=0)))
print("max    " + " ".join(f"{v:6.2f}" for v in rows.max(axis=0)))
