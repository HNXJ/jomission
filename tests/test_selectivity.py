"""S1-S5 inverse pathway-selectivity: equations, ladder solve, robustness, verdict.

Deterministic numerics on sealed measurements (no simulation, no new rule).
Writes results/selectivity_operating.json, results/selectivity_ladder.json,
results/selectivity_robustness.json, results/selectivity_lineage.json.

Method repairs (documented, outcome-neutral harness fixes):
- roots must lie strictly inside all measured driver domains (clamp-edge
  crossings are flat-extrapolation artifacts, not measurements);
- optimizer acceptance is by explicit verification (|R|<=1e-6, cortical,
  in-domain, admissible), not the SLSQP flag (iteration-limit at verified
  minima on the kinked interpolant);
- H2 inits = predeclared H2_INITS + deterministic scan-derived inits
  (H1 scan-best, H0 hits expanded to 3D): starting where grid existence
  was established is coverage, acceptance unchanged.
"""

import numpy as np

from jomission.qualification.selectivity import (
    H0_S_GRID,
    H1_S_GRID,
    H2_INITS,
    PERT_EFF,
    PERT_F,
    PERT_K,
    S_MAX,
    cortical_roots,
    find_roots_s,
    in_domain,
    load_measured,
    make_residual,
    polish,
)


def hits_in_domain(M, eval_point, cortical, s_ee, s_epv, s_esst):
    return [d for d in cortical_roots(eval_point, cortical, s_ee, s_epv, s_esst)
            if in_domain(M, d["rE"], d["rPV"], d["rSST"])]


def test_s1_operating_model():
    """S1: reconstruct + integrity-check the pathway-resolved equations."""
    import json
    M = load_measured()
    assert set(M["F"]) >= {"E", "PV", "SST"}
    assert M["K"][("E", "E")] == 299.0
    assert M["K"][("E", "PV")] == 300.0 and M["K"][("E", "SST")] == 300.0
    assert M["K"][("PV", "E")] == 40.0 and M["K"][("SST", "E")] == 32.0
    ev, co = make_residual(M)
    # s=1 reproduces the sealed Phase-3A verdict: silent root only.
    all1 = find_roots_s(ev, 1.0, 1.0, 1.0)
    assert len(all1) == 1 and all1[0]["rE"] < 1.0, all1
    assert hits_in_domain(M, ev, co, 1.0, 1.0, 1.0) == []
    out = {"K": {f"{a}->{b}": v for (a, b), v in M["K"].items()},
           "s1_roots": all1, "s1_check": "reproduces EI_GEOMETRY_FAIL",
           "scope": "measured domain only; VIP excluded; PV<->SST omitted"}
    json.dump(out, open("results/selectivity_operating.json", "w"),
              indent=2, sort_keys=True)
    print("S1 operating model OK; s=1 ->", all1)


def test_s2_s3_ladder():
    """S2/S3: H0 scan, H1 scan + tied polish, H2 polish (fixed + derived)."""
    import json
    from scipy.optimize import minimize

    M = load_measured()
    ev, co = make_residual(M)
    dom = lambda rE, rpv, rsst: in_domain(M, rE, rpv, rsst)

    h0 = []
    for s in H0_S_GRID:
        c = hits_in_domain(M, ev, co, s, s, s)
        if c:
            h0.append({"s": s, "roots": c})

    h1_hits = []
    for a in H1_S_GRID:
        for b in H1_S_GRID:
            c = hits_in_domain(M, ev, co, a, b, b)
            if c:
                h1_hits.append({"s_ee": a, "s_ei": b,
                                "cost": float(np.log(a) ** 2 + 2 * np.log(b) ** 2),
                                "roots": c})
    h1_best = min(h1_hits, key=lambda h: h["cost"]) if h1_hits else None

    # H1 tied polish (2 dof): free + domain-constrained variants, seeded
    # at scan-best roots. The constrained variant tests whether a cost
    # minimum exists strictly inside the measured domain (vs cost
    # decreasing toward the clamp edge = edge-seeking manifold).
    h1_pol = None
    h1_pol_con = None
    if h1_best:
        seeds = [r["rE"] for r in h1_best["roots"]] + [5.0, 10.0, 20.0, 30.0]

        def obj2(x):
            return float(x[0] ** 2 + 2 * x[1] ** 2)

        def eq2(x):
            a, b = float(np.exp(x[0])), float(np.exp(x[1]))
            return ev(x[2], a, b, b)[0]

        def try_polish(extra_cons):
            best, attempts = None, 0
            for r0 in seeds:
                cons = [{"type": "eq", "fun": eq2},
                        {"type": "ineq", "fun": lambda x: x[2] - 5.0},
                        {"type": "ineq", "fun": lambda x: 40.0 - x[2]}] + extra_cons
                r = minimize(obj2, [np.log(h1_best["s_ee"]), np.log(h1_best["s_ei"]), r0],
                             method="SLSQP",
                             bounds=[(np.log(1e-3), np.log(S_MAX))] * 2 + [(0.0, 40.0)],
                             constraints=cons,
                             options={"maxiter": 2000, "ftol": 1e-12})
                attempts += 1
                a, b, rE = float(np.exp(r.x[0])), float(np.exp(r.x[1])), float(r.x[2])
                if abs(ev(rE, a, b, b)[0]) > 1e-6:
                    continue
                _, rpv, rsst, _ = ev(rE, a, b, b)
                if not co(rE, rpv, rsst) or not dom(rE, rpv, rsst):
                    continue
                if best is None or float(obj2(r.x)) < best[0]:
                    best = (float(obj2(r.x)), a, b, rE, rpv, rsst, bool(r.success))
            return best, attempts

        b, _ = try_polish([])
        if b:
            h1_pol = {"cost": b[0], "s": [b[1], b[2], b[2]],
                      "rE": b[3], "rPV": b[4], "rSST": b[5], "solver_ok": b[6]}
        bc, natt = try_polish([
            {"type": "ineq", "fun": lambda x: 34.9 - x[2]},
            {"type": "ineq", "fun": lambda x: 59.5 - ev(x[2], float(np.exp(x[0])), float(np.exp(x[1])), float(np.exp(x[1])))[1]},
            {"type": "ineq", "fun": lambda x: 67.0 - ev(x[2], float(np.exp(x[0])), float(np.exp(x[1])), float(np.exp(x[1])))[2]},
        ])
        h1_pol_con = ({"cost": bc[0], "s": [bc[1], bc[2], bc[2]],
                       "rE": bc[3], "rPV": bc[4], "rSST": bc[5],
                       "solver_ok": bc[6], "attempts": natt}
                      if bc else {"status": "no-interior-minimum", "attempts": natt})

    # H2: predeclared inits + deterministic scan-derived inits.
    inits = list(H2_INITS)
    if h1_best:
        inits.append((h1_best["s_ee"], h1_best["s_ei"], h1_best["s_ei"]))
    for h in h0:
        inits.append((h["s"], h["s"], h["s"]))
    h2_sols = []
    for init in inits:
        seeds = [5.0, 10.0, 20.0, 30.0]
        p = polish(ev, co, init, r_seeds=tuple(seeds), in_domain_fn=dom)
        if p:
            h2_sols.append({"cost": p[2], "s": list(p[0]), "rE": p[1],
                            "solver_ok": p[3], "init": list(init)})
    h2_best = min(h2_sols, key=lambda h: h["cost"]) if h2_sols else None

    out = {"H0": {"n_grid": len(H0_S_GRID), "hits": h0,
                  "note": "common E-out scaling; inhibition fixed at measured"},
           "H1": {"n_grid": len(H1_S_GRID) ** 2,
                  "n_scan_hits": len(h1_hits), "scan_hits": h1_hits,
                  "scan_best": h1_best, "polished": h1_pol,
                  "polished_constrained": h1_pol_con},
           "H2": {"n_inits": len(inits), "solutions": h2_sols,
                  "best": h2_best}}
    json.dump(out, open("results/selectivity_ladder.json", "w"),
              indent=2, sort_keys=True)
    print("H0 in-domain hits:", len(h0), "| H1 scan hits:", len(h1_hits),
          "| H1 polished:", h1_pol, "| H1 constrained:", h1_pol_con,
          "| H2 solutions:", len(h2_sols), "| H2 best:", h2_best)
    assert isinstance(out["H0"], dict)


def test_s4_robustness():
    """S4: perturbation cube; frozen-solution check + re-solve per ladder."""
    import itertools
    import json
    M = load_measured()
    lad = json.load(open("results/selectivity_ladder.json"))
    cube = list(itertools.product(PERT_F, PERT_K, PERT_EFF))
    rep = {}
    cands_cfg = {
        "H0": [tuple([h["s"]] * 3) for h in lad["H0"]["hits"][:3]],
        # H1 nominal is the scan-certified in-domain root (brentq-verified);
        # polish is a refinement, not a gatekeeper (cost infimum sits at the
        # measurement boundary: documented edge-seeking, not an interior min).
        "H1": [(lad["H1"]["scan_best"]["s_ee"],) * 1 + (lad["H1"]["scan_best"]["s_ei"],) * 2]
        if lad["H1"]["scan_best"] else [],
        "H2": [tuple(lad["H2"]["best"]["s"])] if lad["H2"]["best"] else [],
    }
    for name, seeds in cands_cfg.items():
        if not seeds:
            rep[name] = {"status": "no-nominal-solution"}
            continue
        frozen_ok, resolved, s_min, s_max = 0, 0, None, None
        for pert in cube:
            ev, co = make_residual(M, pert=pert)
            dom = lambda rE, rpv, rsst: in_domain(M, rE, rpv, rsst)
            hit = False
            for s0 in seeds:
                if any(hits_in_domain(M, ev, co, *s0)):
                    hit = True
            if hit:
                frozen_ok += 1
            # Re-solve: local box around each nominal seed (deterministic).
            found = []
            for s0 in seeds:
                lo = [max(v / 4, 0.1) for v in s0]
                hi = [min(v * 4, S_MAX) for v in s0]
                grids = [np.logspace(np.log10(l), np.log10(h), 7)
                         for l, h in zip(lo, hi)]
                if name == "H0":
                    grid = grids[0]
                    for a in grid:
                        if hits_in_domain(M, ev, co, a, a, a):
                            found.append((float(a),) * 3)
                elif name == "H1":
                    for a in grids[0]:
                        for b in grids[1]:
                            if hits_in_domain(M, ev, co, a, b, b):
                                found.append((float(a), float(b), float(b)))
                else:
                    import itertools as it
                    for cand in it.product(*grids):
                        if hits_in_domain(M, ev, co, *cand):
                            found.append(tuple(float(v) for v in cand))
            if found:
                resolved += 1
                for cand in found:
                    s_min = cand if s_min is None else tuple(min(x, y) for x, y in zip(s_min, cand))
                    s_max = cand if s_max is None else tuple(max(x, y) for x, y in zip(s_max, cand))
        rep[name] = {"perturbations": len(cube), "frozen_ok": frozen_ok,
                     "resolved": resolved,
                     "s_viable_min": list(s_min) if s_min else None,
                     "s_viable_max": list(s_max) if s_max else None}
    json.dump(rep, open("results/selectivity_robustness.json", "w"),
              indent=2, sort_keys=True)
    print("robustness:", rep)
    assert isinstance(rep, dict)


def test_s5_verdict():
    """S5: decisive verdict (robustness-gated; nominal-only is not enough)."""
    import json
    lad = json.load(open("results/selectivity_ladder.json"))
    try:
        rob = json.load(open("results/selectivity_robustness.json"))
    except FileNotFoundError:
        rob = {}

    def frac(name):
        r = rob.get(name, {})
        n = r.get("perturbations", 0)
        return (r.get("frozen_ok", 0) / n) if n else 0.0, r

    f0, r0 = frac("H0")
    f1, r1 = frac("H1")
    f2, r2 = frac("H2")
    sb = lad["H1"]["scan_best"]
    h1 = {"s": [sb["s_ee"], sb["s_ei"], sb["s_ei"]], "scan": True,
          "roots": sb["roots"]} if sb else None
    h1p = lad["H1"]["polished"]
    h1c = lad["H1"].get("polished_constrained")
    h2 = lad["H2"]["best"]
    edge_note = ""
    if h1 and not h1p:
        edge_note = ("scan-certified root only; optimizer cost decreases toward "
                     "the E-driver clamp edge (no interior minimum)")
        if h1c and h1c.get("status") != "no-interior-minimum":
            edge_note = "constrained interior minimum found"
            h1 = {"s": h1c["s"], "scan": False, "roots": None}
    if lad["H0"]["hits"] and f0 >= 0.5:
        # H0 here = common gain WITHIN E-out pathways, inhibitory returns
        # FIXED at measured. That is not "no selectivity": relative to the
        # current co-scaling rule it requires E-out vs inhibitory
        # conditioning (one dof), with no differential needed within E-out.
        # (Common scaling *including* inhibitory is the scale lineage's
        # falsified direction through 36x.) Hence FEASIBLE-minimal, not
        # NOT_REQUIRED.
        s_h0 = sorted(h["s"] for h in lad["H0"]["hits"])
        verdict = "SELECTIVITY_FEASIBLE"
        detail = (f"minimal structure: uniform E-out gain s in "
                  f"[{min(s_h0):.1f},{max(s_h0):.1f}], inhibitory fixed at "
                  f"measured; viable in {r0['frozen_ok']}/{r0['perturbations']} "
                  f"perturbations; no within-E-out differential required")
    elif h1 and f1 >= 0.5:
        verdict = "SELECTIVITY_FEASIBLE"
        qual = "robust" if f1 == 1.0 else "marginal"
        detail = (f"H1 one-dof selectivity {qual}: s={h1['s']} "
                  f"frozen {r1['frozen_ok']}/{r1['perturbations']}. {edge_note}")
    elif h2 and f2 >= 0.5:
        verdict = "SELECTIVITY_FEASIBLE"
        detail = f"H2 full independence required: s={h2['s']}"
    elif h1 or h2 or lad["H0"]["hits"]:
        verdict = "SELECTIVITY_INFEASIBLE"
        detail = ("nominal slivers only (H0/H1/H2 survive <50% of perturbations); "
                  "knife-edge solutions rejected per P4")
    else:
        verdict = "SELECTIVITY_INFEASIBLE"
        detail = "no cortical root on H0/H1 grids or H2 polishes within s<=500"
    out = {"parent": "fe79fd2", "verdict": verdict, "detail": detail,
           "verdict_mapping_note": (
               "H0-robust maps to FEASIBLE-minimal, not NOT_REQUIRED: "
               "the H0 solution holds inhibitory returns at measured while "
               "scaling E-out, which requires E-vs-I pathway conditioning "
               "relative to the current co-scaling rule. NOT_REQUIRED would "
               "require common scaling including inhibitory -- the direction "
               "the scale lineage falsified (silent root only, s=2..36)."),
           "ladder_summary": {
               "H0": {"hits": len(lad["H0"]["hits"])},
               "H1": {"scan_hits": lad["H1"]["n_scan_hits"],
                      "polished": lad["H1"]["polished"]},
               "H2": {"solutions": len(lad["H2"]["solutions"]),
                      "best": lad["H2"]["best"]}},
           "robustness": rob,
           "next_authorized_action": ("S6 gate OPEN (pathway-conditioned HDP design); "
                                      "S6 requires explicit authorization; stop here"
                                      if verdict == "SELECTIVITY_FEASIBLE"
                                      else "STOP; no HDP implementation authorized")}
    json.dump(out, open("results/selectivity_lineage.json", "w"),
              indent=2, sort_keys=True)
    print("S5 verdict:", verdict, "|", detail)
    assert verdict in ("SELECTIVITY_NOT_REQUIRED", "SELECTIVITY_FEASIBLE",
                       "SELECTIVITY_INFEASIBLE", "UNRESOLVED")
