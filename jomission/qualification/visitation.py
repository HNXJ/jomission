"""B0-B3 basin-accessibility analysis: visitation distance + separatrix work.

Parent: S9 ACTIVE_BASIN_FAIL (ef32fcc/1848e27). No plant changes: all
quantities are analysis of sealed S9 artifacts (B0/B1) plus bounded new
protocol runs (B2: amplitude bisection <= 2 levels, duration ladder) that
vary only preparation, never the plant.

Visitation distance (predeclared): normalized coarse coordinates
  z = (rE/40, rPV/80, rSST/80, wE/wE*, auxG/auxG*, H)
  d(t) = ||z_obs(t) - z*||
FP refs: S7 roots (rates), FP wpath means (E-pathway mean for wE*),
global aux* = edge-mean of r_pre*TAU_ACT/1000 over the C-min edge list
(VIP r*=0), H*=1. Normalizers are band maxima (40/80/80); w/aux/H are
scale-free ratios. NEAR threshold: min-d < 0.25 (predeclared).

B0 (microstate mapping): only rest-consistent (V=c,u=b*c) constructions
are admissible native microstates (S9 demonstration); the S8 FP has no
demonstrated active-microstate realization -- recorded, not re-litigated.
B1 (saved trajectories): d(t) over sealed S9 release chunks (P0/P7,
4x5s + carry snapshots) and prep endpoints (rate subspace).
B2 (new, bounded): L1 amps {2.5,3.0,3.5} (8 s prep, 1 s-chunk d(t));
L2 = one midpoint of the flip interval; duration ladder at first-sync
amp T in {0.5,1,2,4,8} s drive then 20 s release (1 s chunks, S9 capture
criterion adjudicates each release).
B3 verdict (lineage-global, predeclared aggregation):
  any B2 release CAPTURED -> BASIN_TINY_OR_INACCESSIBLE (narrow works);
  elif min-d < 0.25 on any non-capturing trajectory -> same (approach w/o capture);
  elif all boundary trajectories min-d >= 0.25 with direct silence<->sync
    jumps -> COARSE_FIXED_POINT_OFF_NATIVE_MANIFOLD;
  else UNRESOLVED.
"""

from __future__ import annotations

import numpy as np

CANDIDATES: tuple[float, ...] = (50.0, 54.0, 57.0, 61.0)

TAU_ACT = 20.0
NEAR_D = 0.25

L1_AMPS = (2.5, 3.0, 3.5)
LADDER_T = (0.5, 1.0, 2.0, 4.0, 8.0)

NORM = {"rE": 40.0, "rPV": 80.0, "rSST": 80.0}


def scale_tag(s: float) -> str:
    return ("%.1f" % s).replace(".", "p")


def fp_ref(s, model=None):
    """FP reference vector + normalizers for candidate s (sealed JSONs)."""
    import json

    from jomission.qualification import basin as B

    fp = B.fp_tables(s)
    tag = scale_tag(s)
    geo = json.load(open(f"results/s7_geometry_s{tag}.json"))
    root = geo["cortical"][0]
    wE_star = float(np.mean([fp["wpath"][t] for t in ("EE", "EPV", "ESST", "EVIP")]))
    if model is not None:
        tbl = model.neuron_table()
        cls = np.array([str(r["cell_type"]) for r in tbl])
        el = model.params["edge_list"]
        pre = np.asarray(el.pre)
        rmap = {"E": fp["r"]["E"], "PV": fp["r"]["PV"],
                "SST": fp["r"]["SST"], "VIP": 0.0}
        aux_star = float(np.mean([rmap.get(c, 0.0) * TAU_ACT / 1000.0
                                  for c in cls[pre]]))
    else:
        aux_star = None
    return {"rE": root["rE"], "rPV": root["rPV"], "rSST": root["rSST"],
            "wE_star": wE_star, "aux_star": aux_star, "H_star": 1.0}


def dist(obs, ref):
    """Normalized coarse distance (missing keys ignored pairwise)."""
    terms = []
    for k, n in (("rE", NORM["rE"]), ("rPV", NORM["rPV"]), ("rSST", NORM["rSST"])):
        if k in obs and k in ref:
            terms.append(((obs[k] - ref[k]) / n) ** 2)
    if "wE" in obs and ref.get("wE_star"):
        terms.append(((obs["wE"] - ref["wE_star"]) / ref["wE_star"]) ** 2)
    if "aux" in obs and ref.get("aux_star"):
        terms.append(((obs["aux"] - ref["aux_star"]) / ref["aux_star"]) ** 2)
    if "H" in obs:
        terms.append((obs["H"] - 1.0) ** 2)
    assert terms
    return float(np.sqrt(sum(terms) / len(terms)))
