"""Seed robustness as difference maps (paper Fig 2B): per N, the scorer's relative-power map R [21 contacts x 150 Hz]
of each seed, the pairwise differences R_a - R_b, their SSIM and Pearson r, and each seed's SSIM and r against the
observed map PROTO.

Usage:
  python scripts/e2_seed_diff.py

Runs: the nine ladder runs (e2_ladder_rule.tag). R is e2_scorer.rp_map of the sum-over-cells spectra (as the
verdict). SSIM: Wang et al. 2004, Gaussian window sigma 1.5 (truncate 3.5, as the common 11-sample window), data
range 1 (R is in [0, 1]), K1 0.01, K2 0.03, mean over the whole map with reflect padding at the borders (the
11-sample window spans about half of the 21 depth rows, so border handling matters; a valid-region-only SSIM is
lower for model vs PROTO, 0.011 vs 0.079 for n4000 s1, review). Pearson r over all 21 x 150 pixels.
Writes results/viz/paper_w6/seed_diff.json and seed_diff.png. Post hoc readout; no simulation.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_ladder_rule as lr  # noqa: E402
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "viz" / "paper_w6"


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0, sigma: float = 1.5) -> float:
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    f = lambda x: gaussian_filter(x, sigma, truncate=3.5, mode="reflect")  # noqa: E731
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def rmap(tag: str) -> np.ndarray:
    motif = json.loads((mf.D / f"{tag}_motif.json").read_text(encoding="utf-8"))
    _, S, K = sc.unit_spectra(tag, float(motif["discard_s"]), mf.D)
    return sc.rp_map(S.mean(axis=0), K)


def main(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    maps = {n: {s: rmap(lr.tag(n, s)) for s in lr.SEEDS} for n in lr.NS}
    res: dict = {}
    pairs = list(itertools.combinations(lr.SEEDS, 2))
    for n in lr.NS:
        res[str(n)] = {
            "pairs": {f"{a}-{b}": {"ssim": ssim(maps[n][a], maps[n][b]), "r": pearson(maps[n][a], maps[n][b])}
                      for a, b in pairs},
            "vs_proto": {str(s): {"ssim": ssim(maps[n][s], sc.PROTO), "r": pearson(maps[n][s], sc.PROTO)}
                         for s in lr.SEEDS}}
    (out / "seed_diff.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    fig, axes = plt.subplots(len(lr.NS), len(pairs), figsize=(10, 7.5), sharex=True, sharey=True,
                             constrained_layout=True)
    ext = [float(sc.F[0]), float(sc.F[-1]), float(mf.CHANNELS[-1]), float(mf.CHANNELS[0])]
    for i, n in enumerate(lr.NS):
        for j, (a, b) in enumerate(pairs):
            ax = axes[i, j]
            im = ax.imshow(maps[n][a] - maps[n][b], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, extent=ext)
            p = res[str(n)]["pairs"][f"{a}-{b}"]
            ax.set_title(f"N {n}: seed {a} - seed {b}\nSSIM {p['ssim']:.2f}, r {p['r']:.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel("depth rel. mid-L4")
            if i == len(lr.NS) - 1:
                ax.set_xlabel("frequency (Hz)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="difference of relative power")
    fig.savefig(out / "seed_diff.png", dpi=200)
    return res


if __name__ == "__main__":
    print(json.dumps(main()))
