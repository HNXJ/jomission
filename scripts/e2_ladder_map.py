"""Print-size ladder map for the supplement (Fig S3).

The diagnostic figure from `e2_motif_figure.py` is drawn at 14 x 11 in with 11 pt type and saved as a 6-panel
montage; placed at 0.32\\linewidth it prints at 2.17 in, so its type lands near 1.7 pt. This script draws the one
panel the ladder argument uses (the relative-power map with each contact's peak frequency) at the exact size it
prints, so the type stays at its declared point size.

    python3 scripts/e2_ladder_map.py lfs_n1000_s1_T8 --out-dir results/viz/ladder
    python3 scripts/e2_ladder_map.py --all --out-dir results/viz/ladder

Each panel is 2.166 in wide (0.32 of the 6.7677 in text width) and 1.55 in tall. Fonts: 6 pt ticks, 6.5 pt labels,
7 pt title. Reads `results/laminar_e2/<tag>.npz` and `<tag>.json` (the same sources as e2_motif_figure.py) and
writes `<tag>_laddermap.png`. Values are not recomputed: the map is the per-cell-mean RWS probe power divided by
its maximum over depth at each frequency (not per contact), the same quantity e2_motif_figure.py plots in its
middle column.

The Fig S1 reference map in the supplement is made with the non-default flags
`--frac 0.78 --height 2.1 --suffix refmap`, i.e.
`python3 scripts/e2_ladder_map.py sw_n4000_dens_k1_kd1_sst15_s1_T8 --out-dir results/viz/s1 --frac 0.78 --suffix refmap --height 2.1`
(0.78 x the text width is the width that figure prints at, so its declared type sizes are its printed ones).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.e2_motif_figure as mf  # the scoring, RWS, probe and spectrum definitions: one home
import jomission.laminar.caminos_stack as cs

ROOT = mf.ROOT
D = mf.D
PT_TICK, PT_LABEL, PT_TITLE = 6.0, 6.5, 7.0
PANEL_W_IN = 0.32 * 489.10307 / 72.27  # 2.1664 in: the width this panel is printed at
PANEL_H_IN = 1.55


def one(tag: str, discard_s: float, out_dir: Path, frac: float = 0.32, suffix: str = "laddermap",
        height_in: float | None = None) -> Path:
    z = np.load(D / f"{tag}.npz")
    rec = json.loads((D / f"{tag}.json").read_text(encoding="utf-8"))
    units = list(rec.get("units", rec["sizes"]))
    depth = rec.get("depth_rel_L4") or cs.layout("e2_3unit")[3]
    dt_ms = rec["dt_ms"]
    x = mf.rws(z["g_nS"].astype(np.float64), dt_ms, rec, units)[int(round(discard_s * 1000.0 / dt_ms)):]
    k = int(round(1.0 / dt_ms))
    n1 = x.shape[0] // k
    x = x[: n1 * k].reshape(n1, k, -1).mean(axis=1)
    fs, nper = 1000.0, min(1000, n1)
    n_e = np.array([rec["sizes"][u]["E"] for u in units], dtype=float)
    zu = np.array([depth[u] for u in units])
    K = np.exp(-((mf.CHANNELS[:, None] - zu[None, :]) ** 2) / (2 * mf.SIGMA ** 2))
    f, S = mf.cross_spectra(x / n_e[None], fs, nper)
    P = mf.probe_power(S.mean(axis=0), K)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": PT_TICK, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.labelsize": PT_TICK, "ytick.labelsize": PT_TICK,
                         "pdf.fonttype": 42, "savefig.dpi": 300})
    w_in = frac * 489.10307 / 72.27
    h_in = height_in if height_in is not None else PANEL_H_IN * (frac / 0.32)
    fig, ax = plt.subplots(figsize=(w_in, h_in), layout="constrained")
    R = P / P.max(axis=0, keepdims=True)
    s = (f >= 0) & (f <= 150)
    ys = mf.CHANNELS
    edges = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[:-1] + ys[1:]) / 2,
                            [ys[-1] + (ys[-1] - ys[-2]) / 2]])
    im = ax.imshow(R[:, s], aspect="auto", cmap=mf.PARULA, vmin=0.3, vmax=1.0, interpolation="nearest",
                   extent=(0, 150, edges[-1], edges[0]))
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels(["0", "50", "100", "150"])
    ax.set_xlabel("Frequency (Hz)", fontsize=PT_LABEL, labelpad=1.5)
    ax.set_ylabel("Depth (rel. L4)", fontsize=PT_LABEL, labelpad=1.5)
    # no title: the surrounding \panel{N1000 s1}{...} already prints the clean label, and an embedded
    # "lfs_n1000_s1, s1" repeated the seed and carried an internal tag into the figure
    ax.tick_params(length=2, pad=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, ticks=[0.3, 1.0])
    cb.ax.tick_params(labelsize=PT_TICK, length=2)
    cb.set_label("rel. power", fontsize=PT_LABEL, labelpad=1.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_{suffix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--all", action="store_true", help="every ladder run lfs_n*_s*_T8 plus the n4000 s1 reference")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--discard", type=float, default=2.0)
    ap.add_argument("--frac", type=float, default=0.32, help="fraction of the text width this panel prints at")
    ap.add_argument("--suffix", default="laddermap", help="output filename suffix")
    ap.add_argument("--height", type=float, default=None, help="panel height in inches (default scales with frac)")
    a = ap.parse_args(argv)
    tags = ([t for t in ("lfs_n1000_s1_T8", "lfs_n1000_s2_T8", "lfs_n1000_s3_T8",
                         "lfs_n2000_s1_T8", "lfs_n2000_s2_T8", "lfs_n2000_s3_T8",
                         "sw_n4000_dens_k1_kd1_sst15_s1_T8", "lfs_n4000_s2_T8", "lfs_n4000_s3_T8")]
            if a.all else ([a.tag] if a.tag else []))
    if not tags:
        ap.error("give a tag or --all")
    for t in tags:
        p = one(t, a.discard, a.out_dir, a.frac, a.suffix, a.height)
        print(f"{t} -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
