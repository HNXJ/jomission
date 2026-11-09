"""E0 estimators on a single-unit stack run, beside the E0 result (finite-size check of a small CAMINOS unit).

Usage:
  python scripts/e2_unit_check.py TAG

Reads results/laminar_e2/<TAG>.npz (counts of one unit) and results/laminar_e0/e0_result.json; prints both.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("JAXFNE_SITE"):
    sys.path.insert(0, os.environ["JAXFNE_SITE"])
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from jomission.laminar import caminos_analysis as an  # noqa: E402

tag = sys.argv[1]
z = np.load(ROOT / "results" / "laminar_e2" / f"{tag}.npz")
rec = json.loads((ROOT / "results" / "laminar_e2" / f"{tag}.json").read_text(encoding="utf-8"))
(unit,) = rec["units"]
est = an.analyze_run(z["counts"], rec["dt_ms"] / 1000.0, rec["sizes"][unit])
print(json.dumps({"tag": tag, "sizes": rec["sizes"][unit], "estimates": est}, default=float))
e0 = json.loads((ROOT / "results" / "laminar_e0" / "e0_result.json").read_text(encoding="utf-8"))
print(json.dumps({k: e0[k] for k in e0 if k in ("means", "mean", "summary", "verdict", "estimates_mean")}, default=float)[:800])
