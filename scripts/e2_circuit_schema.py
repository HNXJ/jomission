"""Data and circuit schema figure (Herrera 2026 layout, conjugated to the CAMINOS stack): what the model is built from.

Usage:
  python scripts/e2_circuit_schema.py TAG

Row 1 (data, no simulation): A LANSA cell densities per type vs depth (median over 11 areas, / max per type);
B observed relative-power map, mean of 11 areas (PROTO); C observed map, V1.
Row 2 (the model of run TAG, built, not simulated): D the column, one dot per DOT cells of each unit, placed
uniformly in the unit's depth bins and at random lateral positions, with the unit's within-unit loops (E<->PV,
E<->SOM) and inter-unit E projections drawn as arrows; E built cells per type per depth bin (unit count spread
uniformly over its bins, / max per type, bars) against LANSA (lines). LANSA densities are relative per type, so
only depth profiles are compared, never fractions across types; CR (VIP) has no model counterpart.
Writes results/viz/schema/{TAG}_schema.png. Reads results/laminar_e2/{TAG}.json and docs/laminar_motif/data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jomission.laminar import caminos_stack as cs  # noqa: E402

DATA = ROOT / "docs" / "laminar_motif" / "data"
RES = ROOT / "results" / "laminar_e2"
OUT = ROOT / "results" / "viz" / "schema"
TYPE_COLOR = {"E": "#8e24aa", "PV": "#1565c0", "SOM": "#ff9800"}  # as e2_paper_block.py
LANSA_STYLE = {"NG": ("NG (pyramidal proxy)", "#8e24aa", "-"), "PV": ("PV", "#1565c0", "--"),
               "CB": ("CB (not SST-specific)", "#ff9800", "-."), "CR": ("CR, not modelled", "#9e9e9e", ":")}
LANSA_TO_MODEL = {"NG": "E", "PV": "PV", "CB": "SOM"}
DOT = 20  # cells per dot in panel D
PT = 7


def lansa() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    d = json.loads((DATA / "lansa_density_X.json").read_text(encoding="utf-8"))
    y = json.loads((DATA / "lansa_lfp_Y.json").read_text(encoding="utf-8"))
    X = np.asarray(d["X"], dtype=float)  # [bin, type, area]
    z = np.round(np.asarray(d["axes"]["bin_center_rel_L4"]), 3)
    return z, d["axes"]["types"], np.median(X, axis=2), np.asarray(y["Y"], dtype=float), y["axes"]["areas"]


def built_profile(rec: dict, z: np.ndarray) -> dict[str, np.ndarray]:
    """Cells per type per depth bin: each unit's count spread uniformly over its MODULE_BINS bins."""
    out = {t: np.zeros(z.size) for t in TYPE_COLOR}
    for m, sel in cs.MODULE_BINS.items():
        b = sel(z)
        for t in TYPE_COLOR:
            out[t][b] += rec["sizes"][m][t] / b.sum()
    return out


def _map(ax, R, z, title):
    f = np.arange(1, R.shape[1] + 1)
    im = ax.pcolormesh(f, z, R, cmap="viridis", vmin=0, vmax=1, shading="nearest")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("frequency (Hz)")
    ax.set_title(title, fontsize=PT + 0.5)
    return im


def _arrow(ax, a, b, color, rad=0.25, lw=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4))


def main(tag: str, out: Path = OUT) -> Path:
    rec = json.loads((RES / f"{tag}.json").read_text(encoding="utf-8"))
    z, types, Xm, Y, areas = lansa()
    plt.rcParams.update({"font.size": PT, "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(6.7677, 6.4), constrained_layout=True)  # text width 489.10307 pt: scale 1.000 at print
    # 2 x 3: A | B | C on top, D (two columns) | E on the bottom. A 2 x 4 grid was tried for an E/F split and
    # squeezed E and F to 0.13 in each, because panel D's text overflows its axes and constrained_layout then
    # starves the neighbours; one E panel with both encodings is the smaller layout that stays readable.
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.15, 1.15], height_ratios=[1, 1])
    # A: densities
    ax = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(types):
        lab, col, ls = LANSA_STYLE[t]
        ax.plot(Xm[:, i] / Xm[:, i].max(), z, color=col, ls=ls, lw=1.5, label=lab)
    ax.set_ylim(z[-1] + 0.03, z[0] - 0.03)
    ax.set_xlabel("density / max (median of 11 areas)")
    ax.set_ylabel("depth from mid-L4 (fraction of thickness)")
    ax.set_title("cell densities (LANSA)", fontsize=PT + 0.5)
    ax.legend(frameon=False, fontsize=PT - 0.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              handlelength=1.5, columnspacing=0.8)
    # B, C: observed maps
    axb = fig.add_subplot(gs[0, 1], sharey=ax)
    _map(axb, Y.mean(axis=2), z, "observed motif (mean of 11 areas)")
    axc = fig.add_subplot(gs[0, 2], sharey=ax)
    im = _map(axc, Y[:, :, areas.index("V1")], z, "observed motif, V1")
    for a in (axb, axc):
        a.tick_params(labelleft=False)
    fig.colorbar(im, ax=axc, shrink=0.8, label="relative power (P / max over depth)")
    # D: model column
    axd = fig.add_subplot(gs[1, 0:2], sharey=ax)
    rng = np.random.default_rng(0)
    half = 0.025  # half a bin
    for m, sel in cs.MODULE_BINS.items():
        zb = z[sel(z)]
        lo, hi = zb.min() - half, zb.max() + half
        axd.axhspan(lo, hi, xmin=0, xmax=0.62, color="#f5f5f5" if m != "M" else "#eeeeee", zorder=0)
        for t in TYPE_COLOR:
            n = int(round(rec["sizes"][m][t] / DOT))
            axd.scatter(rng.uniform(0.02, 0.98, n) * 0.6, rng.uniform(lo, hi, n), s=1.6, color=TYPE_COLOR[t],
                        lw=0, zorder=2)
        zc = rec["depth_rel_L4"][m]
        # loops at x 0.72 (E), 0.86 (PV above, SOM below)
        pe, pp, ps = (0.72, zc), (0.88, zc - 0.06), (0.88, zc + 0.06)
        for p, t in ((pe, "E"), (pp, "PV"), (ps, "SOM")):
            axd.scatter(*p, s=55, color=TYPE_COLOR[t], zorder=3)
        for src, dst in ((pe, pp), (pp, pe), (pe, ps), (ps, pe)):
            _arrow(axd, src, dst, "#424242")
        n = sum(rec["sizes"][m].values())
        pl = rec.get("p_local")
        pl = pl.get(m) if isinstance(pl, dict) else pl
        w = {k.split(":")[0]: v for k, v in (rec.get("modifiers", {}).get("w") or {}).items() if k.endswith(":" + m)}
        wtxt = ", ".join(f"{r} x{v:g}" for r, v in w.items())
        axd.text(0.94, zc, f"{m}  N {n:,}\nE {rec['sizes'][m]['E']:,}  PV {rec['sizes'][m]['PV']:,}  "
                 f"SOM {rec['sizes'][m]['SOM']:,}\np {pl:.3f}" + (f"\n{wtxt}" if wtxt else "")
                 if pl is not None else f"{m}  N {n:,}", va="center", fontsize=PT - 0.5)
    units = list(rec["sizes"])
    for a, b in zip(units[:-1], units[1:]):
        za, zb_ = rec["depth_rel_L4"][a], rec["depth_rel_L4"][b]
        _arrow(axd, (0.68, za + 0.02), (0.68, zb_ - 0.02), TYPE_COLOR["E"], rad=-0.3, lw=0.8)
        _arrow(axd, (0.66, zb_ - 0.02), (0.66, za + 0.02), TYPE_COLOR["E"], rad=-0.3, lw=0.8)
    axd.set_xlim(0, 1.25)
    axd.set_xticks([])
    axd.spines["bottom"].set_visible(False)
    taus = rec.get("tau_sst_e_ms") or {}
    taus = taus if isinstance(taus, dict) else {"all": taus}  # older records: one scalar
    axd.set_title(f"model column ({tag})", fontsize=PT + 0.5, loc="left")
    axd.set_xlabel(f"one dot per {DOT} cells; loops E<->PV, E<->SOM;\n"
                   f"purple arrows: E projections between units;\n"
                   f"SOM->E tau {', '.join(f'{k} {v:g}' for k, v in taus.items())} ms",
                   fontsize=PT - 1, loc="left", linespacing=1.15)
    # E: built cell counts as bars, against the LANSA profiles as lines, one colour per type. The bars are drawn
    # at alpha 0.55 with a darker edge so the two encodings stay separable and the bars clear contrast on white;
    # each type's line style differs from the others, so the pair is readable without relying on hue alone.
    bp = built_profile(rec, z)
    off = {"E": -0.012, "PV": 0.0, "SOM": 0.012}
    axe = fig.add_subplot(gs[1, 2], sharey=ax)
    for t, col in TYPE_COLOR.items():
        axe.barh(z + off[t], bp[t] / bp[t].max(), height=0.011, color=col, alpha=0.55,
                 edgecolor=col, lw=0.25, label=f"{t} built")
    for i, t in enumerate(types):
        if t in LANSA_TO_MODEL:
            lab, col, ls = LANSA_STYLE[t]
            axe.plot(Xm[:, i] / Xm[:, i].max(), z, color=col, ls=ls, lw=1.6,
                     label=lab.replace(" (not SST-specific)", ""))
    axe.set_xlabel("cells / max per type")
    axe.set_title("built vs LANSA", fontsize=PT + 0.5)
    axe.tick_params(labelleft=False)
    axe.legend(frameon=False, fontsize=PT - 1, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
               handlelength=1.3, columnspacing=0.5, borderaxespad=0.01)
    for a, letter in zip((ax, axb, axc, axd, axe), "ABCDE"):
        a.text(-0.06, 1.06, letter, transform=a.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}_schema.png"
    fig.savefig(path, dpi=300)
    return path


if __name__ == "__main__":
    print(main(sys.argv[1]))
