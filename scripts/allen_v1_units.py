"""Map the Allen V1 replica onto the shared scorer's units and write the scorer inputs (read-only on run outputs).

Units are the four E layers (L2/3, L4, L5, L6): the replica records recurrent conductance only onto E layers (g_nS, two
columns per layer: AMPA, GABA), so these are the units the RWS readout sees. Interneurons (L1 Inh, PV, SST, VIP) are not
units; they act through the conductances onto E. Depth is depth_rel_mid_L4 from the figure record (thickness units).
Crossing is scored on the shared scorer's contact grid (e2_motif_figure.CHANNELS), the grid its K matrix uses.
The scorer itself (scripts/e2_scorer.py) is unchanged; this module only writes its inputs.

Usage: python scripts/allen_v1_units.py TAG [--src results/allen_v1] [--dst results/allen_v1/scorer]
Then:  python scripts/e2_scorer.py TAG --results-dir <dst>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

N_UNITS = 4


def depth_map(fig: dict) -> tuple[list[str], dict[str, float]]:
    """(units, depth_rel_L4) from a figure record: the E layers in order, with their depth relative to mid-L4."""
    units = list(fig["layers"])
    depths = list(fig["depth_rel_mid_L4"])
    if len(units) != N_UNITS or len(depths) != N_UNITS:
        raise ValueError(f"expected {N_UNITS} E-layer units, got {len(units)} layers and {len(depths)} depths")
    return units, dict(zip(units, (float(d) for d in depths)))


def _jsonable(o):
    if isinstance(o, np.bool_):
        return bool(o)
    return float(o)


def write_inputs(tag: str, src: Path, dst: Path) -> dict:
    """Write dst/<tag>.json (record + units, depth_rel_L4), dst/<tag>_motif.json (crossing on CHANNELS) and a symlink
    dst/<tag>.npz to the run's npz. Returns the summary printed by main()."""
    dst.mkdir(parents=True, exist_ok=True)
    rec = json.loads((src / f"{tag}.json").read_text(encoding="utf-8"))
    fig = json.loads((src / f"{tag}_motif.json").read_text(encoding="utf-8"))
    units, depth = depth_map(fig)
    (dst / f"{tag}.json").write_text(json.dumps(dict(rec, units=units, depth_rel_L4=depth), indent=1), encoding="utf-8")
    link = dst / f"{tag}.npz"
    if link.is_symlink() or link.exists():  # a dangling link is not link.exists(); replace it either way
        link.unlink()
    os.symlink((src / f"{tag}.npz").resolve(), link)
    discard = float(fig["discard_s"])
    f, S, K = sc.unit_spectra(tag, discard, dst)
    res = mf.score(f, S, K, mf.CHANNELS, mf.BANDS[mf.FIG_BANDS])
    motif = {"tag": tag, "units": units, "discard_s": discard, "sum_over_cells": {"scores": {mf.FIG_BANDS: res}}}
    (dst / f"{tag}_motif.json").write_text(json.dumps(motif, indent=1, default=_jsonable), encoding="utf-8")
    return {"tag": tag, "units": units, "depth_rel_mid_L4": depth, "n_windows": int(S.shape[0]),
            "crossing_2se": res["crossing_2se"], "discard_s": discard}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Write shared-scorer inputs for an Allen V1 replica run (4 E-layer units).")
    ap.add_argument("tag")
    ap.add_argument("--src", default=str(ROOT / "results" / "allen_v1"))
    ap.add_argument("--dst", default=str(ROOT / "results" / "allen_v1" / "scorer"))
    a = ap.parse_args(argv)
    print(json.dumps(write_inputs(a.tag, Path(a.src), Path(a.dst)), default=_jsonable))


if __name__ == "__main__":
    main()
