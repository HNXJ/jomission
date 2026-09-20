"""S1-S5 inverse pathway-selectivity analysis: predeclared operating model + solver.

Parent authority: fe79fd2 (model frozen: JaxFNE 0.4.24, jomission_sat_ee_v0
shape, measured F/K/rheobases, zero tonic, C-min architecture, no
homeostatic HDP, all retired axes stand). No simulation, no new rule.

Operating equations (all components measured at s=1; NOT transplanted gains)::

    rPV(rE;s)  = F_PV(K_EPV * s_EPV * i_EPV(rE))
    rSST(rE;s) = F_SST(K_ESST * s_ESST * i_ESST(rE))
    R(rE;s)    = F_E(K_EE*s_EE*i_EE(rE)
                     + K_PVE*i_PVE(rPV) + K_SSTE*i_SSTE(rSST)) - rE

- i_EE/i_EPV/i_ESST: per-edge current interpolants vs E-driver rate,
  sealed battery_b1 (E-driver star, rule active).
- i_PVE/i_SSTE: sealed batteries B2/B3 (inhibitory returns, rule active).
- F_E/F_PV/F_SST: sealed single-cell F-I (rule-independent emitter property).
- K: fresh structural convergence from build_cmin (no sim).
- Inhibitory returns FIXED at measured: inhibition is the control system to
  be retained; selectivity applies to E-out pathways only. (Differs from the
  scale lineage, which co-scaled inhibition too -- H0 here is strictly more
  excitation-favorable, so its failure would strengthen the case further.)
- VIP excluded from the map (sealed silent scope); recorded, not deleted.

Hypothesis ladder (predeclared):
  H0: s_EE = s_EPV = s_ESST = s (common E-out scaling; 1 dof)
  H1: (s_EE, s_EI) with s_EPV = s_ESST = s_EI (2 dof)
  H2: (s_EE, s_EPV, s_ESST) independent (3 dof)

Objective (inverse design, not grid search): min ||log s||^2 subject to a
cortical root of R in GEO_BANDS. Coarse log-grid existence scan, then local
constrained polish (SLSQP on (log s, rE), equality R=0, cortical
inequalities), multiple fixed inits. Deterministic: no randomness.

Admissibility (predeclared): each s in (0, 500] (engine w_ceiling);
cortical membership per GEO_BANDS (rE 5..40, rPV>=2, rSST>=1, max<80).

Robustness S4 (predeclared perturbations, re-solve + frozen-solution check):
  F current-axis shift {-10%, 0, +10%}; K per-class {-1, 0, +1} count;
  efficacy interpolants {x0.95, x1.0, x1.05}. Feasible region S_viable =
  min/max viable s per ladder over the perturbation cube.

Verdict S5: SELECTIVITY_NOT_REQUIRED (reserved: would require common
scaling *including* inhibitory -- the scale lineage's falsified direction) |
SELECTIVITY_FEASIBLE (robust; fewest dof preferred; H0-robust with fixed
inhibition counts as FEASIBLE-minimal since it needs E-vs-I conditioning) |
SELECTIVITY_INFEASIBLE (nothing admissible/robust) | UNRESOLVED.
Only FEASIBLE opens the S6 gate (not this lineage).
"""

from __future__ import annotations

import numpy as np

H0_S_GRID = tuple(float(x) for x in np.logspace(np.log10(0.5), np.log10(500), 60))

H1_S_GRID = tuple(float(x) for x in np.logspace(np.log10(0.5), np.log10(500), 24))

H2_INITS = (
    (1.0, 1.0, 1.0),
    (4.44, 4.44, 4.44),
    (10.0, 1.0, 1.0),
    (10.0, 10.0, 10.0),
    (30.0, 5.0, 2.0),
    (100.0, 10.0, 10.0),
)

# H2 polish additionally starts from scan-derived inits (deterministic rule:
# H1 scan-best s and each H0 hit s expanded to 3D), documented in the test.
# Starting the local optimizer where grid existence was established is
# coverage, not solution selection: the acceptance criterion (verified
# in-domain cortical root) is unchanged.

S_MAX = 500.0
R_GRID = tuple(float(x) for x in np.arange(0.0, 40.5, 0.5))

# Strict interior margin below each measured driver-domain maximum.
# A root AT a clamp edge relies on flat extrapolation, not measurement.
DOMAIN_MARGIN = {"ee": 0.1, "pve": 0.5, "sste": 0.5}

PERT_F = (-0.10, 0.0, 0.10)
PERT_K = (-1, 0, 1)
PERT_EFF = (0.95, 1.0, 1.05)


def load_measured():
    """Sealed s=1 measurements: F, current interpolants, fresh K."""
    import json

    from jomission.qualification.cmin import build_cmin
    from jomission.qualification.ei_geometry import GEO_BANDS

    b1 = json.load(open("results/battery_b1.json"))
    b2 = json.load(open("results/battery_b2.json"))
    b3 = json.load(open("results/battery_b3.json"))
    F = {c: (np.array(v["I"]), np.array(v["r"])) for c, v in b1["F"].items()}

    def curve(rows, key):
        xs = np.array([r["driver_rate"] for r in rows])
        ys = np.array(
            [
                r["classes"][key]["I"] if key in r["classes"] else r["classes"]["E"]["I"]
                for r in rows
            ]
        )
        o = np.argsort(xs)
        return xs[o], ys[o]

    x_ee, i_ee = curve(b1["rows"], "E")
    x_epv, i_epv = curve(b1["rows"], "PV")
    x_esst, i_esst = curve(b1["rows"], "SST")
    x_pve, i_pve = curve(b2["rows"], "E")
    x_sste, i_sste = curve(b3["rows"], "E")

    m = build_cmin()
    el = m.params["edge_list"]
    tbl = m.neuron_table()
    cls = np.array([str(r["cell_type"]) for r in tbl])
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    K = {}
    for cpost in ("E", "PV", "SST"):
        for cpre in ("E", "PV", "SST"):
            sel = (cls[post] == cpost) & (cls[pre] == cpre)
            K[(cpre, cpost)] = float(sel.sum() / max((cls == cpost).sum(), 1))
    return {
        "F": F,
        "GEO": dict(GEO_BANDS),
        "K": K,
        "curves": {
            "ee": (x_ee, i_ee),
            "epv": (x_epv, i_epv),
            "esst": (x_esst, i_esst),
            "pve": (x_pve, i_pve),
            "sste": (x_sste, i_sste),
        },
    }


def driver_domains(M):
    """Measured driver-rate support maxima (no-extrapolation boundary)."""
    C = M["curves"]
    return {
        "ee": float(C["ee"][0].max()),
        "pve": float(C["pve"][0].max()),
        "sste": float(C["sste"][0].max()),
    }


def in_domain(M, rE, rPV, rSST):
    """Strict interior of all three measured driver domains."""
    D = driver_domains(M)
    return (
        rE < D["ee"] - DOMAIN_MARGIN["ee"]
        and rPV < D["pve"] - DOMAIN_MARGIN["pve"]
        and rSST < D["sste"] - DOMAIN_MARGIN["sste"]
    )


def make_residual(M, pert=(0.0, 0, 1.0)):
    """Residual closure under perturbation (f_shift, k_shift, eff_gain)."""
    f_shift, k_shift, eff = pert
    F = {c: (Ii * (1.0 + f_shift), rr) for c, (Ii, rr) in M["F"].items()}
    K = {k: v + k_shift for k, v in M["K"].items()}
    C = {k: (x, y * eff) for k, (x, y) in M["curves"].items()}
    GEO = M["GEO"]

    def F_of(c, I):
        Ii, rr = F[c]
        return float(np.interp(max(I, 0.0), Ii, rr))

    def eval_point(rE, s_ee, s_epv, s_esst):
        iEE = float(np.interp(rE, *C["ee"]))
        iEPV = float(np.interp(rE, *C["epv"]))
        iESST = float(np.interp(rE, *C["esst"]))
        rPV = F_of("PV", K[("E", "PV")] * s_epv * iEPV)
        rSST = F_of("SST", K[("E", "SST")] * s_esst * iESST)
        iPVE = float(np.interp(min(rPV, C["pve"][0].max()), *C["pve"]))
        iSSTE = float(np.interp(min(rSST, C["sste"][0].max()), *C["sste"]))
        Ie = K[("E", "E")] * s_ee * iEE + K[("PV", "E")] * iPVE + K[("SST", "E")] * iSSTE
        return F_of("E", Ie) - rE, rPV, rSST, Ie

    def cortical(rE, rPV, rSST):
        return (
            GEO["rE_lo"] <= rE <= GEO["rE_hi"]
            and rPV >= GEO["rPV_min"]
            and rSST >= GEO["rSST_min"]
            and max(rE, rPV, rSST) < GEO["sat_max"]
        )

    return eval_point, cortical


def find_roots_s(eval_point, s_ee, s_epv, s_esst):
    from scipy.optimize import brentq

    grid = np.array(R_GRID)
    vals = [eval_point(x, s_ee, s_epv, s_esst) for x in grid]
    R = np.array([v[0] for v in vals])
    roots = []
    for a, b, ra, rb in zip(grid[:-1], grid[1:], R[:-1], R[1:]):
        if ra == 0.0:
            roots.append(float(a))
        elif ra * rb < 0.0:
            roots.append(float(brentq(lambda x: eval_point(x, s_ee, s_epv, s_esst)[0], a, b)))
    detail = []
    for rt in sorted(set(round(x, 3) for x in roots)):
        _, rpv, rsst, Ie = eval_point(rt, s_ee, s_epv, s_esst)
        detail.append({"rE": rt, "rPV": rpv, "rSST": rsst, "I_E": Ie})
    return detail


def cortical_roots(eval_point, cortical, s_ee, s_epv, s_esst):
    return [
        d
        for d in find_roots_s(eval_point, s_ee, s_epv, s_esst)
        if cortical(d["rE"], d["rPV"], d["rSST"])
    ]


def polish(
    eval_point, cortical, init_s, fix=None, r_seeds=(5.0, 10.0, 20.0, 30.0), in_domain_fn=None
):
    """SLSQP: min ||log s||^2 s.t. R(rE;s)=0 + cortical membership.

    fix maps axis index -> value (H0/H1 reductions). Acceptance is by
    explicit verification (|R|<=1e-6, cortical, in-domain, admissible),
    NOT by the solver flag: SLSQP reports iteration-limit at verified
    minima on this kinked interpolant landscape. Returns (s, rE, cost,
    solver_ok) or None. r_seeds are deterministic rE starting points.
    """
    from scipy.optimize import minimize

    init_s = tuple(float(v) for v in init_s)
    free = [i for i in range(3) if not (fix and i in fix)]
    x0_base = [np.log(init_s[i]) for i in free]

    def unpack(x):
        s = list(init_s)
        for j, i in enumerate(free):
            s[i] = float(np.exp(np.clip(x[j], np.log(1e-3), np.log(S_MAX))))
        if fix:
            for i, v in fix.items():
                s[i] = float(v)
        return tuple(s), float(np.clip(x[-1], 0.0, 40.0))

    def obj(x):
        s, _ = unpack(x)
        return float(sum(np.log(v) ** 2 for v in s))

    def eq(x):
        s, rE = unpack(x)
        return eval_point(rE, *s)[0]

    cons = [{"type": "eq", "fun": eq}]
    # Cortical membership as inequalities g(x) >= 0 (GEO_BANDS values).

    def mk_ineq(kind):
        def g(x):
            s, rE = unpack(x)
            _, rpv, rsst, _ = eval_point(rE, *s)
            if kind == "lo":
                return rE - 5.0
            if kind == "hi":
                return 40.0 - rE
            if kind == "pv":
                return rpv - 2.0
            if kind == "sst":
                return rsst - 1.0
            return 79.0 - max(rE, rpv, rsst)

        return g

    for kind in ("lo", "hi", "pv", "sst", "sat"):
        cons.append({"type": "ineq", "fun": mk_ineq(kind)})
    bounds = [(np.log(1e-3), np.log(S_MAX))] * len(free) + [(0.0, 40.0)]
    best = None
    for r0 in r_seeds:
        x0 = x0_base + [float(r0)]
        r = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        s, rE = unpack(r.x)
        if abs(eval_point(rE, *s)[0]) > 1e-6:
            continue
        _, rpv, rsst, _ = eval_point(rE, *s)
        if not cortical(rE, rpv, rsst):
            continue
        if in_domain_fn is not None and not in_domain_fn(rE, rpv, rsst):
            continue
        if any(v <= 0 or v > S_MAX for v in s):
            continue
        cost = float(sum(np.log(v) ** 2 for v in s))
        if best is None or cost < best[2]:
            best = (s, rE, cost, bool(r.success))
    if best is None:
        return None
    return best
