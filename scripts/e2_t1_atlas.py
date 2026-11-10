"""T1 V3 atlas (SEAL_T1 visualization): the seed-mean relative-power SL map per level of one axis, beside PROTO.

Usage:
  python scripts/e2_t1_atlas.py AXIS N

For each level of AXIS (e2_select.LEVELS), the scorer's map R (e2_scorer.rp_map of the sum-row spectra) of every
seed with a verdict, averaged over the seeds present; the title gives the seeds present and the seed-mean r_proto,
r_ab, r_gm from the verdict files. A level with no finished seed is drawn empty. Writes
results/viz/t1/V3_{AXIS}_n{N}_atlas.png. Readout only; selection stays with e2_select.py.
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
import e2_select as sel  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "viz" / "t1"
SEEDS = (1, 2, 3)


def level_maps(axis: str, n: int, level: str, d: Path = mf.D) -> tuple[list[int], list[np.ndarray], list[dict]]:
    seeds, maps, verdicts = [], [], []
    for s in SEEDS:
        tag = f"t1_{axis}{level}_n{n}_s{s}_T8"
        vf = d / f"{tag}_verdict.json"
        if not vf.exists():
            continue
        motif = json.loads((d / f"{tag}_motif.json").read_text(encoding="utf-8"))
        _, S, K = sc.unit_spectra(tag, float(motif["discard_s"]), d)
        seeds.append(s)
        maps.append(sc.rp_map(S.mean(axis=0), K))
        verdicts.append(json.loads(vf.read_text(encoding="utf-8")))
    return seeds, maps, verdicts


def main(axis: str, n: int, out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    levels = sel.LEVELS[axis]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True, constrained_layout=True)
    ext = [float(sc.F[0]), float(sc.F[-1]), float(mf.CHANNELS[-1]), float(mf.CHANNELS[0])]
    panels = [("observed (PROTO)", sc.PROTO)]
    for lv in levels:
        seeds, maps, vs = level_maps(axis, n, lv)
        if not seeds:
            panels.append((f"{axis} {lv.replace('p', '.')}: no run yet", None))
            continue
        m = {k: float(np.mean([v[k] for v in vs])) for k in ("r_proto", "r_ab", "r_gm")}
        ref = " (ref)" if lv == sel.REF[axis] else ""
        panels.append((f"{axis} {lv.replace('p', '.')}{ref}, seeds {','.join(map(str, seeds))}\n"
                       f"r {m['r_proto']:.2f}, ab {m['r_ab']:.2f}, gm {m['r_gm']:.2f}", np.mean(maps, axis=0)))
    for ax, (title, R) in zip(axes.flat, panels):
        if R is not None:
            im = ax.imshow(R, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=ext)
        ax.set_title(title, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("depth rel. mid-L4")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="relative power (P / max over depth)")
    fig.suptitle(f"T1 V3 atlas, axis {axis}, N {n}: seed-mean SL map per level", fontsize=10)
    path = out / f"V3_{axis}_n{n}_atlas.png"
    fig.savefig(path, dpi=180)
    return path


if __name__ == "__main__":
    print(main(sys.argv[1], int(sys.argv[2])))
