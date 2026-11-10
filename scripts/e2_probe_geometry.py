"""Probe geometry readout for a saved E2 run: tilted and laterally offset virtual probe (T6, post hoc, no simulation).

Usage:
  python scripts/e2_probe_geometry.py TAG [--extent L] [--results-dir D] [--out-dir D]

Geometry (ASSUMED, from the packet): contact k at arc position s_k = CHANNELS[k] sits at
  x_k = offset + s_k sin(theta),  z_k = s_k cos(theta)   (projected depth, rel. mid-L4).
Each unit is a sheet at its depth z_u with lateral half-width L (--extent, default 0.5):
  K[k, u] = exp(-(z_k - z_u)^2 / (2 SIGMA^2)) * exp(-max(0, |x_k| - L)^2 / (2 SIGMA^2)).
The lateral factor depends on the contact only (every unit has the same L). Grid: theta in {0, 30, 60, 90} deg
x offset in {0, 0.25, 0.5}. Spectra: the sum-over-cells row, from e2_scorer.unit_spectra.

Per geometry:
  M_arc (primary; dispatcher ruling 2026-11-10, ASSUMED): e2_motif_figure.score slope over contact arc position
    s_k, i.e. what an experimenter who does not know the tilt measures. Verdict bands.
  M_z (secondary): the same score slope over projected depth z_k; NaN when the z span is 0 (theta 90).
  Crossing and zero depth: from the z-based score (as the packet's score() on contacts ordered by z_k).
  r_proto: contact power P(c, f) interpolated linearly over z_k (np.interp) onto the 21 bin centres CHANNELS.
    Bins outside [min z_k, max z_k] are not covered, not extrapolated. R = P / max over covered bins (ASSUMED:
    normalisation over covered bins, which equals e2_scorer.rp_map at theta 0). r_proto = mean over f of
    Pearson r over covered bins against PROTO (dispatcher ruling 2026-11-10, option 3, ASSUMED).
    Fewer than 5 covered bins, or collapsed z (theta 90): r_proto and delta_r NaN with a reason. M is NaN
    at theta 90 (contrast slope over zero depth range).
  Identical-unit control (ASSUMED): every unit gets the mean over units of the sum-row auto-spectra, cross
    terms zero, at the same geometry, with the same bin rule. M_ctrl, r_ctrl; delta_r = r_proto - r_ctrl.
Writes <TAG>_probe_geometry.json (one row per geometry; NaN as null) and <TAG>_probe_geometry.png
(M and delta_r vs theta, one line per offset).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as es  # noqa: E402

THETAS_DEG = (0, 30, 60, 90)
OFFSETS = (0.0, 0.25, 0.5)
MIN_BINS = 5
F = es.F  # 1..150 Hz; frequency index = Hz on the 1 Hz grid
FREQ = np.arange(501.0)  # full 0..500 Hz grid of unit_spectra
BANDS = mf.BANDS[mf.FIG_BANDS]
NAN = float("nan")
RESULTS = Path("/Users/hamednejat/workspace/dev/jomission/results/laminar_e2")
OUT_DIR = Path("/Users/hamednejat/workspace/dev/jomission/results/viz/paper_w5")


def contacts(theta_deg: float, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """(x_k, z_k) of the 21 contacts. cos(90 deg) is set to exactly 0 so the depths collapse exactly."""
    th = np.deg2rad(theta_deg)
    c = np.cos(th)
    c = 0.0 if abs(c) < 1e-12 else c
    s = mf.CHANNELS
    return offset + s * np.sin(th), s * c


def kernel(zu: np.ndarray, x: np.ndarray, z: np.ndarray, extent: float) -> np.ndarray:
    """K[contact, unit] per the packet: depth Gaussian times a lateral factor (per contact)."""
    depth = np.exp(-((z[:, None] - zu[None, :]) ** 2) / (2 * mf.SIGMA ** 2))
    lat = np.exp(-np.maximum(0.0, np.abs(x) - extent) ** 2 / (2 * mf.SIGMA ** 2))
    return depth * lat[:, None]


def identical_units(S: np.ndarray) -> np.ndarray:
    """S [window, f, U, U] -> every unit given the mean over units of the sum-row auto-spectra; cross terms zero."""
    U = S.shape[-1]
    d = np.einsum("wfuu->wf", S).real / U
    out = np.zeros(S.shape, dtype=complex)
    idx = np.arange(U)
    out[:, :, idx, idx] = d[:, :, None]
    return out


def r_proto_bins(P: np.ndarray, z: np.ndarray) -> tuple[float, int, str | None]:
    """P [contact, f] at 1..150 Hz, rows in ascending z -> (r_proto, n_bins_covered, reason). See module docstring."""
    cov = (mf.CHANNELS >= z[0] - 1e-12) & (mf.CHANNELS <= z[-1] + 1e-12)
    n = int(cov.sum())
    if z[-1] - z[0] == 0:
        return NAN, n, "projected depths collapse (z range 0)"
    if n < MIN_BINS:
        return NAN, n, f"projected depths cover < {MIN_BINS} bins"
    # weights W[bin, contact] of linear interpolation over z; contacts are distinct and ascending
    W = np.array([np.interp(mf.CHANNELS[cov], z, e) for e in np.eye(z.size)]).T
    Pb = W @ P
    R = Pb / Pb.max(axis=0, keepdims=True)
    r, _ = es._r_over_bins(R, es.PROTO[cov])
    return r, n, None


def geometry_row(S: np.ndarray, zu: np.ndarray, theta: float, offset: float, extent: float,
                 f: np.ndarray = FREQ) -> dict:
    """One geometry: real units and identical-unit control. S [window, f, U, U], zu [U]."""
    x, z = contacts(theta, offset)
    K = kernel(zu, x, z, extent)
    o = np.argsort(z, kind="stable")
    z, K = z[o], K[o]
    collapsed = z[-1] - z[0] == 0
    Sc = identical_units(S)
    s = mf.CHANNELS[o]  # arc positions, in the same contact order as K
    arc = mf.score(f, S, K, s, BANDS)
    arc_c = mf.score(f, Sc, K, s, BANDS)
    sc = None if collapsed else mf.score(f, S, K, z, BANDS)
    sc_c = None if collapsed else mf.score(f, Sc, K, z, BANDS)
    P = mf.probe_power(S.mean(axis=0), K)[:, F]
    Pc = mf.probe_power(Sc.mean(axis=0), K)[:, F]
    r, n, reason = r_proto_bins(P, z)
    rc, _, _ = r_proto_bins(Pc, z)
    return {"theta_deg": float(theta), "offset": float(offset), "z_min": float(z[0]), "z_max": float(z[-1]),
            "n_bins_covered": n,
            "M_arc": arc["contrast_M"], "jackknife_se_arc": arc["jackknife_se"], "M_arc_ctrl": arc_c["contrast_M"],
            "M_z": sc["contrast_M"] if sc else NAN, "jackknife_se_z": sc["jackknife_se"] if sc else None,
            "M_z_ctrl": sc_c["contrast_M"] if sc_c else NAN,
            "crossing_2se": sc["crossing_2se"] if sc else None,
            "zero_depth_first": sc["zero_depth_first"] if sc else None,
            "r_proto": r, "r_ctrl": rc, "delta_r": r - rc, "reason": reason}


def run(tag: str, extent: float = 0.5, results_dir=RESULTS) -> dict:
    rd = Path(results_dir)
    motif = json.loads((rd / f"{tag}_motif.json").read_text(encoding="utf-8"))
    rec = json.loads((rd / f"{tag}.json").read_text(encoding="utf-8"))
    units = list(rec["units"]) if "units" in rec else list(rec["sizes"])
    depth = rec.get("depth_rel_L4") or mf.cs.layout("e2_3unit")[3]
    zu = np.array([depth[u] for u in units])
    f, S, _ = es.unit_spectra(tag, float(motif["discard_s"]), rd)
    rows = [geometry_row(S, zu, th, off, extent, f) for th in THETAS_DEG for off in OFFSETS]
    return {"tag": tag, "discard_s": float(motif["discard_s"]), "units": units, "unit_depth_rel_L4": zu.tolist(),
            "n_windows": int(S.shape[0]), "extent": extent, "sigma": mf.SIGMA, "bands": mf.FIG_BANDS,
            "rows": rows}


def clean(o):
    """NaN and inf become null so the JSON is standard."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def figure(out: dict, path: Path) -> None:
    fig, (ax_m, ax_d) = plt.subplots(1, 2, figsize=(10, 4))
    for off in OFFSETS:
        rows = [r for r in out["rows"] if r["offset"] == off]
        th = [r["theta_deg"] for r in rows]
        ax_m.plot(th, [r["M_arc"] for r in rows], "o-", label=f"offset {off:g}")
        ax_m.plot(th, [r["M_z"] for r in rows], "o--", alpha=0.5, mfc="w")
        ax_d.plot(th, [r["delta_r"] for r in rows], "o-", label=f"offset {off:g}")
    for ax, ylab in ((ax_m, "contrast M (solid M_arc, dashed M_z)"), (ax_d, "delta r_proto (vs identical-unit control)")):
        ax.axhline(0, c="k", lw=0.8)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel(ylab)
        ax.set_xticks(THETAS_DEG)
        ax.spines[["top", "right"]].set_visible(False)
    ax_m.legend(frameon=False)
    fig.suptitle(f"{out['tag']}: probe geometry, L = {out['extent']:g}; M_z and delta_r are NaN at theta 90 "
                 "(depth collapse)", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Probe geometry readout for a saved E2 run (T6).")
    ap.add_argument("tag")
    ap.add_argument("--extent", type=float, default=0.5)
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    out = run(a.tag, a.extent, a.results_dir)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{a.tag}_probe_geometry.json").write_text(json.dumps(clean(out), indent=1), encoding="utf-8")
    figure(out, od / f"{a.tag}_probe_geometry.png")
    for r in out["rows"]:
        print(json.dumps(clean({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})))


if __name__ == "__main__":
    main()
