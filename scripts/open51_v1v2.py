"""OPEN-51 V1/V2 from count-validated reconstructed rasters (A1 batch outputs), per manifests/visualization_contract.json.

V1 = first 1000 ms, V2 = last 1000 ms, same conventions: rows are cells in E, PV, SOM order (the model's contiguous
cell order; E0/E1 have one column, no area or layer axis); one dot per spike. Every title carries the protocol label
"count-validated reconstructed raster". Before drawing, each raster is re-checked against its counts file and the
counts file against the original (np.array_equal); a mismatch stops the script.

Usage: python -B open51_v1v2.py OUT_DIR LINEAGE SALVAGE_KIND RUN_DIR:STEM [RUN_DIR:STEM ...]
  SALVAGE_KIND is laminar_e0 or laminar_e1; one panel row per run.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SALVAGE = Path("F:/repo_salvage_2026-11-10/jomission/E_repos__wt_jomission-handoff.loose/results")
LABEL = "count-validated reconstructed raster (OPEN-51, protocol 5d586da + A1)"
DT_MS, WIN = 0.1, 10000  # 1000 ms
COL = {"E": "#2b6cb0", "PV": "#c53030", "SOM": "#38a169"}

out, lineage, kind = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
runs = [a.rsplit(":", 1) for a in sys.argv[4:]]
out.mkdir(parents=True, exist_ok=True)
checks = []
data = []
for run_dir, stem in runs:
    rd = Path(run_dir)
    rec = json.loads(next(rd.glob(f"{stem}.receipt.json")).read_text(encoding="utf-8"))
    sizes = rec["result"]["sizes"]
    r = np.load(rd / f"{stem}_raster.npz")
    step, cell, shape = r["step"], r["cell"], tuple(r["shape"])
    counts = np.load(rd / f"{stem}_counts.npy")
    bounds = np.cumsum([0, sizes["E"], sizes["PV"], sizes["SOM"]])
    rebuilt = np.zeros((shape[0], 3), np.int64)
    for k in range(3):
        m = (cell >= bounds[k]) & (cell < bounds[k + 1])
        np.add.at(rebuilt[:, k], step[m], 1)
    orig = np.load(SALVAGE / kind / f"{stem}_counts.npy")
    ok = bool(np.array_equal(rebuilt, counts) and np.array_equal(counts, orig))
    checks.append({"run": stem, "raster_matches_counts_and_original": ok, "n_spikes": int(step.size), "sizes": sizes})
    if not ok:
        raise SystemExit(f"{stem}: raster or counts do not match the original; no panel drawn")
    data.append((stem, step, cell, bounds, shape[0]))

for vname, which in (("V1", "first"), ("V2", "last")):
    fig, axes = plt.subplots(len(data), 1, figsize=(6.8, 1.9 * len(data) + 0.5), dpi=150, squeeze=False,
                             constrained_layout=True)
    for ax, (stem, step, cell, bounds, n) in zip(axes[:, 0], data):
        a = 0 if which == "first" else n - WIN
        m = (step >= a) & (step < a + WIN)
        for k, p in enumerate(("E", "PV", "SOM")):
            mm = m & (cell >= bounds[k]) & (cell < bounds[k + 1])
            ax.scatter((step[mm] - a) * DT_MS + a * DT_MS, cell[mm], s=0.15, lw=0, color=COL[p], rasterized=True)
        ax.set_ylim(bounds[-1], 0)
        ax.set_yticks([(bounds[k] + bounds[k + 1]) / 2 for k in range(3)], ["E", "PV", "SOM"])
        for b in bounds[1:-1]:
            ax.axhline(b, color="0.6", lw=0.4)
        ax.set_xlim(a * DT_MS, (a + WIN) * DT_MS)
        ax.set_title(stem, fontsize=8, loc="left")
    axes[-1, 0].set_xlabel("time (ms)")
    fig.suptitle(f"{lineage} {vname}: {which} 1000 ms; {LABEL}", fontsize=8)
    name = "raster_initial_1000ms.png" if vname == "V1" else "raster_final.png"
    fig.savefig(out / name)
    plt.close(fig)
(out / "v1v2_checks.json").write_text(json.dumps({"label": LABEL, "checks": checks}, indent=1) + "\n",
                                      encoding="utf-8", newline="\n")
print(json.dumps(checks))
