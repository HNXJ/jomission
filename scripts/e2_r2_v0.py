"""V0 (contract stage V0): rule-based construction schematic for the R2 P0 layout.

Labelled rule-based. Reads only the construction (unit sizes, depths, the 33 rules and their realised edge counts);
no edge list is read, and the sealed construction path is not changed.

Units: S, M and D are units (layers) of one area 'C' in jomission/laminar/caminos_stack.py (rows(), area "C").
Depths are dimensionless fractions of cortical thickness relative to L4 (layout depth_rel_L4), not millimetres.

Not drawn (contract V0 content, stated on the figure and in the receipt): x and y placement, O and X connections,
FF against FB direction, excitatory against inhibitory projection sign, and connection authority g or w.

Usage: python scripts/e2_r2_v0.py ROOT OUTDIR
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from jomission.laminar import caminos_stack as cs  # noqa: E402

LAY = "e2_3unit_n4000_dens_k1"
SEED = 7
sizes, rho, p_inter, depth = cs.layout(LAY)
mods = cs.variant(LAY)[1]
tau_sst_e = cs.sst_taus(LAY, sizes)
p_local = cs.local_p(LAY, sizes)
tdrive = cs.drive_entries(LAY, sizes)
model, rows = cs.construct(SEED, sizes, p_inter, p_local, tau_sst_e_ms=tau_sst_e, k_scale=mods.get("k", 1.0),
                           deep_size=mods.get("dsz", 1.0), gaba_scale=mods.get("gi", 1.0), triple_drive=tdrive,
                           w_scales=mods.get("w"))
counts = cs.edge_table(model)  # rule name -> realised edge count (no edge list read)
rule_names = [r[0] for r in model.static["adex"]["rules"]]
assert set(counts) == set(rule_names), "rule table and counts disagree"

NOT_DRAWN = ("Not drawn (contract V0): x/y placement, O and X connections, FF/FB direction, E/I projection sign, "
             "connection authority g or w.")
table = {"layout": LAY, "seed": SEED, "rule_based": True, "units": ["S", "M", "D"],
         "note": "Rule-based construction schematic. Counts are the realised edges per construction rule; "
                 "the edge list is not drawn. Depths are fractions of cortical thickness relative to L4.",
         "unit_sizes": sizes, "rho": rho, "depth_rel_L4": depth, "rules": counts, "not_drawn": NOT_DRAWN}
(OUT / "V0_rule_table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")

units = ["S", "M", "D"]  # unit order follows the layout: superficial, middle, deep
fig, ax = plt.subplots(figsize=(13, 10.5), dpi=150)
ax.set_xlim(0, 13)
ax.set_ylim(0, 10.5)
ax.axis("off")
xs = {"S": 0.6, "M": 4.9, "D": 9.2}
colour = {"E": "#c44e52", "PV": "#2a9d8f", "SOM": "#8d6cc6"}
BOX_Y0, BOX_H = 3.9, 4.9
for u in units:
    x0 = xs[u]
    ax.add_patch(FancyBboxPatch((x0, BOX_Y0), 3.2, BOX_H, boxstyle="round,pad=0.05", fc="#f7f7f7", ec="#555"))
    ax.text(x0 + 1.6, BOX_Y0 + BOX_H + 0.25,
            f"unit {u}  (rho {rho[u]:g}, depth {depth[u]:+.2f} rel. L4, thickness fraction)",
            ha="center", fontsize=8.0)
    for i, p in enumerate(["E", "PV", "SOM"]):
        n = sizes[u][p]
        y = 8.0 - i * 1.25
        ax.add_patch(FancyBboxPatch((x0 + 0.3, y - 0.3), 2.6, 0.6, boxstyle="round,pad=0.03",
                                    fc=colour[p], ec="none", alpha=0.85))
        ax.text(x0 + 1.6, y, f"{p}  N={n}", ha="center", va="center", color="white", fontsize=10)

# intra-unit rule counts printed under each block, below the box
for u in units:
    lines = []
    for key in sorted(k for k in counts if k.endswith(f"_{u}")):
        lines.append(f"{key[:-len(u) - 1]}: {counts[key]:,}")
    ax.text(xs[u] + 1.6, BOX_Y0 - 0.2, "intra-unit rules\n" + "\n".join(lines), ha="center", va="top",
            fontsize=6.4, family="monospace")

# cross-unit rule counts as a block of text along the bottom
xl = sorted(k for k in counts if k.startswith("x_"))
ax.text(0.3, 2.1, "cross-unit rules (name: realised edges)", fontsize=8, weight="bold")
cols = 4
for i, k in enumerate(xl):
    col, row = divmod(i, (len(xl) + cols - 1) // cols)
    ax.text(0.3 + col * 3.2, 1.8 - row * 0.26, f"{k}: {counts[k]:,}", fontsize=6.2, family="monospace")

ax.text(0.3, 0.25, NOT_DRAWN, fontsize=7, color="#333")
ax.text(12.9, 0.05, "RULE-BASED CONSTRUCTION SCHEMATIC (no edge list drawn). Layout " + LAY + ", seed " + str(SEED),
        ha="right", fontsize=7, color="#333")
fig.savefig(OUT / "V0_rule_based_schematic.png", bbox_inches="tight")
print("written", OUT / "V0_rule_based_schematic.png")
print("rules", len(counts), "total edges", sum(counts.values()))
