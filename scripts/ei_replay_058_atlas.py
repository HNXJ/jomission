"""Build the V3 suite atlas for SCI-EI-REPLAY-058 from the sealed regime construction.

Usage (from the repo root): python scripts/ei_replay_058_atlas.py

The construction is the one scripts/whole_system_ei_regime_arm.py builds at g = 0 (through
the render script's build_from_driver, so it is the circuit that runs). The raster is the
final 1000 ms probe, results/viz/SCI-EI-REPLAY-058/probe_final.npz. No membrane or
plasticity trace is supplied, so the atlas reports those panels as unavailable.
Writes results/viz/SCI-EI-REPLAY-058/atlas/. Exit 0 on success.
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import render_visualization_contract as RVC  # noqa: E402
from jomission.visualization import jaxfne_suite_atlas as atlas  # noqa: E402

LINEAGE = "SCI-EI-REPLAY-058"
DRIVER = "scripts/whole_system_ei_regime_arm.py"
PROBE = ROOT / "results" / "viz" / LINEAGE / "probe_final.npz"
OUT = ROOT / "results" / "viz" / LINEAGE / "atlas"


def main() -> int:
    model, _index, _rf, _enf, _drv = RVC.build_from_driver(DRIVER, g=0.0)
    spikes = np.load(PROBE)["spikes"]
    signals = types.SimpleNamespace(spikes=spikes, V_m=None)
    manifest = atlas.build_suite_atlas(
        model,
        signals,
        out_dir=str(OUT),
        dt_ms=1.0,
        seed=0,
        title=f"{LINEAGE} suite atlas",
    )
    print("atlas written", OUT, "panels", len(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
