"""Per-unit absolute band power of E2 runs (readout only): where the model's gamma and alpha-beta come from.

Usage:
  python scripts/e2_unit_power.py OUT.json TAG [TAG ...]

Power = the real part of the window-mean unit auto-spectrum (diagonal of e2_scorer.unit_spectra, the scorer's
own input), summed over 1 Hz bins in each band: ab 10-30 Hz, gm 50-150 Hz, lo 1-10 Hz, tot 1-150 Hz (image bands
of the scorer; edges inclusive, so lo and ab share the 10 Hz bin). Units are those of the run (S, M, D for the 3-unit stack). Writes {tag: {unit: {band: power,
gm_frac, ab_frac}}} plus the max |imag|/|real| of the diagonal as a check that the real part is the power.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

BANDS = {"lo": (1, 10), "ab": (10, 30), "gm": (50, 150), "tot": (1, 150)}


def band_power(f: np.ndarray, S: np.ndarray, units: list[str]) -> dict:
    """f [F], S [window, F, U, U] -> {unit: {band: power, gm_frac, ab_frac}, _imag: max |imag|/|real|}."""
    d = np.diagonal(S.mean(axis=0), axis1=1, axis2=2)  # [F, U]
    out: dict = {"_imag": float(np.abs(d.imag).max() / np.abs(d.real).max())}
    p = d.real
    for i, u in enumerate(units):
        b = {k: float(p[(f >= lo) & (f <= hi), i].sum()) for k, (lo, hi) in BANDS.items()}
        b["gm_frac"] = b["gm"] / b["tot"]
        b["ab_frac"] = b["ab"] / b["tot"]
        out[u] = b
    return out


def main(out: Path, tags: list[str], d: Path = mf.D) -> dict:
    res = {}
    for tag in tags:
        rec = json.loads((d / f"{tag}.json").read_text(encoding="utf-8"))
        motif = json.loads((d / f"{tag}_motif.json").read_text(encoding="utf-8"))
        f, S, _ = sc.unit_spectra(tag, float(motif["discard_s"]), d)
        units = list(rec["units"]) if "units" in rec else list(rec["sizes"])
        res[tag] = band_power(f, S, units)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
