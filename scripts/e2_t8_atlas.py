"""T8 V3 atlas (SEAL_T8 visualization): seed-mean relative-power SL map of each arm beside the observed map.

Usage: python scripts/e2_t8_atlas.py

Per arm, the scorer's map R (e2_scorer.rp_map of the sum-row spectra, 2 s discard) of seeds 1-3 averaged; titles
give the seed means of r_ab, r_gm, r_proto from the verdict files. Writes results/viz/t8/V3_t8_atlas.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "viz" / "t8"
ARMS = {"ref": "reference", "dflat": "density flat", "drev": "density S<->D",
        "rflat": "SOM/PV flat 0.79", "rrev": "SOM/PV reversed"}
SEEDS = (1, 2, 3)


def arm_map(arm: str, d: Path = mf.D) -> tuple[np.ndarray, dict[str, float]]:
    maps, vs = [], []
    for s in SEEDS:
        tag = f"t8_{arm}_n4000_s{s}_T8"
        _, S, K = sc.unit_spectra(tag, 2.0, d)
        maps.append(sc.rp_map(S.mean(axis=0), K))
        vs.append(json.loads((d / f"{tag}_verdict.json").read_text(encoding="utf-8")))
    return np.mean(maps, axis=0), {k: float(np.mean([v[k] for v in vs])) for k in ("r_ab", "r_gm", "r_proto")}


def main(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6})
    # drawn at print size (full text width, 6.8 in)
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.6), sharex=True, sharey=True, constrained_layout=True)
    ext = [float(sc.F[0]), float(sc.F[-1]), float(mf.CHANNELS[-1]), float(mf.CHANNELS[0])]
    panels = [("observed, 11-area mean", sc.PROTO)]
    for a, name in ARMS.items():
        R, m = arm_map(a)
        panels.append((f"{name}\nab {m['r_ab']:.2f}, gm {m['r_gm']:.2f}, r {m['r_proto']:.2f}", R))
    for ax, (title, R), letter in zip(axes.flat, panels, "ABCDEF"):
        im = ax.imshow(R, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=ext)
        ax.set_title(title, fontsize=6.5)
        ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    for ax in axes[:, 0]:
        ax.set_ylabel("depth rel. mid-L4")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="relative power\n(P / max over depth)")
    path = out / "V3_t8_atlas.png"
    fig.savefig(path, dpi=300)
    return path


if __name__ == "__main__":
    print(main())
