"""T6 score summary across seeds (paper Fig 6C): M_arc and delta_r by tilt and offset, seed mean +- SD per N.

Usage:
  python scripts/e2_probe_summary.py

Reads results/viz/paper_w5/<TAG>_probe_geometry.json for the nine ladder runs (e2_ladder_rule.tag), writes
results/viz/paper_w5/probe_summary.json and probe_summary.png. A NaN row value (theta 90 r, collapsed z) stays NaN
in the mean (np.nanmean over seeds only when every seed is finite; else NaN). Post hoc readout; no simulation.
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
import e2_ladder_rule as lr  # noqa: E402

D = Path(__file__).resolve().parents[1] / "results" / "viz" / "paper_w5"
KEYS = ("M_arc", "delta_r")


def load(n: int, d: Path = D) -> dict:
    """{(theta, offset): {key: [per-seed values]}}"""
    out: dict = {}
    for s in lr.SEEDS:
        rows = json.loads((d / f"{lr.tag(n, s)}_probe_geometry.json").read_text(encoding="utf-8"))["rows"]
        for r in rows:
            cell = out.setdefault((r["theta_deg"], r["offset"]), {k: [] for k in KEYS})
            for k in KEYS:
                cell[k].append(np.nan if r[k] is None else float(r[k]))
    return out


def summarise(cells: dict) -> list[dict]:
    rows = []
    for (th, off), v in sorted(cells.items()):
        row = {"theta_deg": th, "offset": off}
        for k in KEYS:
            a = np.array(v[k])
            fin = np.all(np.isfinite(a))
            row[k] = {"mean": float(a.mean()) if fin else None, "sd": float(a.std(ddof=1)) if fin else None,
                      "n": int(a.size)}
        rows.append(row)
    return rows


def main(d: Path = D) -> dict:
    out = {str(n): summarise(load(n, d)) for n in lr.NS}
    (d / "probe_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, constrained_layout=True)
    offs = sorted({r["offset"] for r in out[str(lr.NS[0])]})
    for j, n in enumerate(lr.NS):
        for i, k in enumerate(KEYS):
            ax = axes[i, j]
            for off, mk in zip(offs, ("o", "s", "^")):
                rr = [r for r in out[str(n)] if r["offset"] == off and r[k]["mean"] is not None]
                ax.errorbar([r["theta_deg"] for r in rr], [r[k]["mean"] for r in rr],
                            yerr=[r[k]["sd"] for r in rr], marker=mk, capsize=3, label=f"offset {off}")
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_title(f"N = {n}, {k}", fontsize=10)
            if j == 0:
                ax.set_ylabel("M_arc (contrast)" if k == "M_arc" else "r_proto - r_ctrl")
            if i == 1:
                ax.set_xlabel("tilt theta (deg)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Probe geometry, seed mean +- SD (seeds 1-3); theta 90 delta_r undefined", fontsize=10)
    fig.savefig(d / "probe_summary.png", dpi=200)
    return out


if __name__ == "__main__":
    o = main()
    for n, rows in o.items():
        print(n, [(r["theta_deg"], r["offset"], r["M_arc"]["mean"], r["delta_r"]["mean"]) for r in rows])
