"""Analytic construction of candidate O/X transfer authorities. No simulation.

    python scripts/o_authority_parameterization.py [--out results/o_authority_parameterization.json]

WS-REALIZE-R4-PARAM measured what the O relation realized and returned
PARAMETER_REALIZATION_UNQUALIFIED. This script inverts that measurement. With transfer
authority g defined per receiving neuron as

    g_p = (sum of |w| over p-afferents of cell i) / (sum of |w| over local excitatory afferents of i)

the per-edge weight a projection needs is

    w_p = g_p * K_local / N_afferents,p

so per-edge weight follows from convergence instead of being shared across every projection.
Two distinct artifacts in the current realization follow from ignoring that:

  fan-in     a 28-afferent feedback projection and a 15-afferent feedforward projection both
             carry w = 0.5, so their authorities differ by 28/15 for no stated reason
  hierarchy  a target area reached by two upstream areas receives exactly twice the afferents
             of one reached by a single upstream area, so frontal cells carry twice the
             long-range authority of sensory cells at the same per-edge weight

This script chooses no g. It tabulates w_p(g) and reports which conventions are ambiguous
rather than resolving them.

Read-only: the model is constructed and measured, never modified, and no simulation is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jomission.tfne import architecture, nf, realize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "o_authority_parameterization.json"

#: TFNE relation per (pre layer, post layer), as realized. ff and fb come from the O[fffb]
#: clauses, lat from X[lat].
RELATION = {("L2", "L4"): "ff", ("L3", "L4"): "ff", ("L6", "L1"): "fb", ("L3", "L3"): "lat"}

#: Normalized authorities to tabulate. Every value is below 1, so K_long < K_local holds at
#: initialization. The grid is a menu, not a proposal.
G_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75)


def build():
    chain, rules, external, decl = architecture.build()
    n = nf.normalize(chain, rules, external)
    cfg = realize.declare_projections(realize.declare_mechanisms(realize.scaffold(n)[0], n), n)
    return realize.construct(cfg)


def measure(model) -> dict:
    idx, (area, lay, cls) = realize.index_map(model)
    area, lay, cls = np.asarray(area), np.asarray(lay), np.asarray(cls)
    el = model.params["edge_list"]
    pre, post, w = np.asarray(el.pre), np.asarray(el.post), np.asarray(el.weight)
    same_area = area[pre] == area[post]

    # K_local: summed local excitatory afferent weight per receiving E cell, checked per layer
    # so that using one reference for every target population is established, not assumed.
    m = same_area & (w > 0)
    k_local = np.zeros(area.size)
    np.add.at(k_local, post[m], np.abs(w[m]))
    by_layer = {}
    for L in sorted(set(lay[cls == "E"].tolist())):
        sel = np.flatnonzero((cls == "E") & (lay == L))
        by_layer[L] = {"n": int(sel.size), "K_local_exc_mean": float(k_local[sel].mean()),
                       "min": float(k_local[sel].min()), "max": float(k_local[sel].max())}
    e_cells = np.flatnonzero(cls == "E")
    K_local = float(k_local[e_cells].mean())

    # A receiving neuron sums every long-range afferent it has, so convergence is counted per
    # target cell across all source areas, not per source-area identity. It is not uniform, so
    # convergence classes are kept separate rather than averaged into one N.
    rel_of_edge = np.array([RELATION.get((lay[a], lay[b]), "") for a, b in zip(pre, post)])
    channels = {}
    for rel in ("ff", "fb", "lat"):
        mask = (~same_area) & (rel_of_edge == rel)
        if not mask.any():
            continue
        cnt = np.bincount(post[mask], minlength=area.size)
        k = np.zeros(area.size)
        np.add.at(k, post[mask], np.abs(w[mask]))
        recv = np.flatnonzero(cnt > 0)
        classes = {}
        for N in sorted(set(cnt[recv].tolist())):
            sel = recv[cnt[recv] == N]
            classes[str(N)] = {
                "N_afferents": int(N),
                "n_target_cells": int(sel.size),
                "target_areas": sorted(set(area[sel].tolist())),
                "K_realized": float(k[sel].mean()),
                "g_realized": float(k[sel].mean() / K_local),
            }
        channels[rel] = {
            "post_layers": sorted(set(lay[recv].tolist())),
            "source_motifs": sorted({f"{lay[a]}.E -> {lay[b]}.E"
                                     for a, b in zip(pre[mask], post[mask])}),
            "w_realized": float(np.abs(w[mask]).mean()),
            "n_target_cells": int(recv.size),
            "convergence_uniform": len(classes) == 1,
            "convergence_classes": classes,
        }

    return {"K_local": K_local, "K_local_by_layer": by_layer,
            "K_local_layer_spread": max(abs(v["K_local_exc_mean"] - K_local)
                                        for v in by_layer.values()),
            "channels": channels, "n_E": int(e_cells.size)}


def parameterize(meas: dict) -> dict:
    """w_p(g) under the two policies that differ once convergence is non-uniform."""
    K, channels = meas["K_local"], meas["channels"]
    out = {}
    for rel, ch in channels.items():
        cls = ch["convergence_classes"]
        # Policy EQUAL_G: every target cell of this channel gets the same authority g, so the
        # per-edge weight differs by convergence class. Hierarchy position stops mattering.
        equal_g = {c: {"N_afferents": v["N_afferents"], "target_areas": v["target_areas"],
                       "w_of_g": {f"{g:g}": g * K / v["N_afferents"] for g in G_GRID}}
                   for c, v in cls.items()}
        # Policy EQUAL_W: one per-edge weight for the whole channel, pinned so that the
        # least-converged class reaches g. More-converged targets then exceed g in proportion.
        N_ref = min(v["N_afferents"] for v in cls.values())
        equal_w = {"pinned_to_N": N_ref,
                   "w_of_g": {f"{g:g}": g * K / N_ref for g in G_GRID},
                   "g_actual_by_class": {c: v["N_afferents"] / N_ref for c, v in cls.items()}}
        out[rel] = {"equal_g_per_cell": equal_g, "equal_w_per_edge": equal_w}
    return out


def main(out: Path | None = None) -> None:
    meas = measure(build())
    par = parameterize(meas)
    K = meas["K_local"]
    lo = min(v["K_local_exc_mean"] for v in meas["K_local_by_layer"].values())
    hi = max(v["K_local_exc_mean"] for v in meas["K_local_by_layer"].values())
    rec = {
        "schema": "jomission.o_authority_parameterization.v1",
        "lineage": "WS-REALIZE-R4-PARAM (measurement); this file is its analytic inverse",
        "read_only": "no simulation; the model is constructed and measured, never modified",
        "chooses_nothing": ("g_FF, g_FB and g_X are model-definition decisions. This file "
                            "tabulates w_p(g), states which conventions are ambiguous, and "
                            "proposes no value. Nothing here is fitted to the observed "
                            "oscillator"),
        "definition": ("g_p = (sum |w| over p-afferents of cell i) / (sum |w| over local "
                       "excitatory afferents of i), per receiving neuron. Inverted: "
                       "w_p = g_p * K_local / N_afferents,p"),
        "K_local": K,
        "K_local_evidence": (f"layer-wise means span {lo:.4f}-{hi:.4f}, a spread of "
                             f"{meas['K_local_layer_spread']:.4f} about the mean, so one reference "
                             "serves every target population. Cell-to-cell spread within a layer "
                             "is about +/-0.08 and is not collapsed here"),
        "K_local_by_layer": meas["K_local_by_layer"],
        "measured": meas["channels"],
        "construction": par,
        "g_grid": list(G_GRID),
        "open_conventions": {
            "convergence_is_not_uniform": (
                "ff targets split 35 afferents (V4_1, V4_2, one upstream area) against 70 (FEF, "
                "PFC, two upstream V4 instances); fb splits 28 (V1_1, V1_2) against 56 (V4_1, "
                "V4_2, reached by both FEF and PFC); lat is uniform at 20. So a single per-edge "
                "weight gives frontal cells exactly twice the feedforward authority of sensory "
                "cells. Whether that doubling is intended hierarchy or an unintended artifact of "
                "the fan-in is a model-definition decision. Both policies are tabulated"),
            "ff_is_two_motifs": (
                "L4.E receives both L2.E->L4.E and L3.E->L4.E. The channel totals above pool "
                "them, which is what the receiving neuron experiences; whether the two motifs "
                "should additionally carry different per-edge weights is not decided here"),
            "delay": (
                "delay is a second unqualified parameter and is not parameterized here. The edge "
                "list uses delay_storage uniform at 0 steps; jaxfne.emitters.resolve_edge_delay_"
                "steps supports delay_storage 'per_edge' with a per-edge delay_steps array, so "
                "distinguishing local recurrence from inter-area communication needs no engine "
                "change"),
            "targets": (
                "every long-range projection is E->E. Whether O should also reach PV, SST or VIP "
                "is a separate model-definition question, untouched here"),
        },
    }
    dest = (ROOT / out) if out and not Path(out).is_absolute() else (out or OUT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")

    print("K_local = %.4f  (local excitatory afferent weight per E cell; layer spread %.4f)"
          % (K, meas["K_local_layer_spread"]))
    print("\nrealized today, at the frozen w = 0.5:")
    print("  %-4s %-6s %5s %6s %9s %8s %8s  %s"
          % ("chan", "target", "N_aff", "cells", "w", "K", "g", "target areas"))
    for rel, ch in meas["channels"].items():
        for v in ch["convergence_classes"].values():
            print("  %-4s %-6s %5d %6d %9.4f %8.4f %8.2f  %s"
                  % (rel, ",".join(ch["post_layers"]) + ".E", v["N_afferents"],
                     v["n_target_cells"], ch["w_realized"], v["K_realized"], v["g_realized"],
                     ", ".join(v["target_areas"])))
    print("\nEQUAL_G: same authority g at every target cell; per-edge weight by convergence class")
    print("  %-4s %5s" % ("chan", "N_aff") + "".join("%10g" % g for g in G_GRID))
    for rel, p in par.items():
        for v in p["equal_g_per_cell"].values():
            print("  %-4s %5d" % (rel, v["N_afferents"])
                  + "".join("%10.5f" % v["w_of_g"][f"{g:g}"] for g in G_GRID))
    print("\nEQUAL_W: one per-edge weight per channel, pinned so the least-converged class hits g")
    print("  %-4s %5s" % ("chan", "N_ref") + "".join("%10g" % g for g in G_GRID)
          + "   g at the other class")
    for rel, p in par.items():
        e = p["equal_w_per_edge"]
        mult = sorted(e["g_actual_by_class"].values())
        print("  %-4s %5d" % (rel, e["pinned_to_N"])
              + "".join("%10.5f" % e["w_of_g"][f"{g:g}"] for g in G_GRID)
              + "   %s" % (" ".join("%.0fx g" % m for m in mult)))
    print("\nwrote", dest.relative_to(ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(Path(a.out) if a.out else None)
