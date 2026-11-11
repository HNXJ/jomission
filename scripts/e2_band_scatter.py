"""Band-correlation scatter of every scored E2 run (reply to review 4, 2026-11-10).

Usage: python scripts/e2_band_scatter.py

One point per verdict file in results/laminar_e2: r_ab (10-30 Hz) against r_gm (50-150 Hz), grouped by test
(finite-size ladder, T1 selection levels, T1 confirmation, exploratory weight runs, T8 arms). The shaded region is
the band part of the pass rule (both >= 0.6). Writes results/viz/paper_w7/band_scatter.png and a JSON of the
plotted values next to it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "laminar_e2"
OUT = ROOT / "results" / "viz" / "paper_w7"
# group: (pattern on tag, label, colour, marker, size)
GROUPS = [
    ("ladder", r"^(lfs_n\d+|sw_n4000_dens_k1_kd1_sst15)_s\d_T8$", "finite-size ladder", "#9e9e9e", "o", 14),
    ("t1_sel", r"^t1_(kd|sst|sstd)[0-9p]+_n4000_s\d_T8$", "T1 levels, selection seeds", "#1565c0", "o", 8),
    ("t1_conf", r"^t1_confirm_.*_T12$", "T1 winner, fresh seeds", "#1565c0", "D", 26),
    ("weights", r"^e2x_", "middle-unit weights, exploratory", "#ff9800", "^", 20),
    ("t8_ref", r"^t8_ref_", "T8 reference", "#000000", "s", 26),
    ("t8_dens", r"^t8_(dflat|drev)_", "T8 density flat / S\u2194D", "#00acc1", "v", 26),
    ("t8_ratio", r"^t8_(rflat|rrev)_", "T8 SOM/PV flat / reversed", "#e53935", "P", 30),
]


def main() -> None:
    pts = {g[0]: [] for g in GROUPS}
    seen = set()  # T1 reference levels reuse earlier runs under a second file name; plot each run once
    for f in sorted(D.glob("*_verdict.json")):
        v = json.loads(f.read_text(encoding="utf-8"))
        if v["tag"] in seen:
            continue
        seen.add(v["tag"])
        for key, pat, *_ in GROUPS:
            if re.search(pat, v["tag"]):
                pts[key].append({"tag": v["tag"], "r_ab": v["r_ab"], "r_gm": v["r_gm"], "pass": v["pass"]})
                break
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    ax.add_patch(plt.Rectangle((0.6, 0.6), 0.4, 0.4, color="#cfb87c", alpha=0.35, lw=0, zorder=0))
    ax.text(0.98, 0.97, "both \u2265 0.6", ha="right", va="top", fontsize=6.5)
    ax.axhline(0, color="#e0e0e0", lw=0.5, zorder=0)
    ax.axvline(0, color="#e0e0e0", lw=0.5, zorder=0)
    for key, _, label, col, mk, sz in GROUPS:
        p = pts[key]
        ax.scatter([q["r_ab"] for q in p], [q["r_gm"] for q in p], s=sz, c=col, marker=mk, lw=0.3,
                   edgecolors="white", label=f"{label} (n = {len(p)})", zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("alpha-beta depth correlation $r_{ab}$ (10\u201330 Hz)")
    ax.set_ylabel("gamma depth correlation $r_{gm}$ (50\u2013150 Hz)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="lower left", handletextpad=0.3, borderaxespad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "band_scatter.png", dpi=300)
    (OUT / "band_scatter.json").write_text(json.dumps(pts, indent=1) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in pts.items()}, "pass:", sum(q["pass"] for v in pts.values() for q in v))


if __name__ == "__main__":
    main()
