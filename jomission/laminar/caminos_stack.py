"""Stacks of CAMINOS E-PV-SST units coupled by adjacent-unit E projections (E2; PLAN_LANSA_E2.md).

A unit is a CAMINOS module (caminos_e0.py) whose cell counts may be smaller than the published 8000 E /
1500 inhibitory: every within-unit block keeps the published in-degree (560 E inputs, 105 inhibitory
inputs per cell, split by rho) through p = 0.07 * 8000 / N_E, so E/I = 8000:1500 is kept in every unit.
Adjacent units (i, i+1) are coupled both ways from E to E, PV and SST with in-degree K_INTER per
neighbour (p = K_INTER / N_E of the source unit), the local weights and the 2 ms delay; an end unit takes
2 K_INTER from its one neighbour, so every unit receives 2 K_INTER inter-unit E inputs per cell. The 3-unit
stack keeps the owner's fixed p 0.01 per adjacent pair instead.

JaxFNE 0.5.9 refuses column() for the AdEx family, so the network is built with construct_adex_model and
per-neuron rows whose layer label is the unit name; connection selectors then compile per unit.
"""

from __future__ import annotations

import numpy as np

from jomission.laminar import caminos_e0 as c0

POPS = c0.POPS  # E, PV, SOM
K_E = c0.NE * 0.07  # 560 E inputs per cell (CAMINOS, p 0.07)
INH_PER_E = c0.N_INH / c0.NE  # 1500 / 8000
K_INTER = 80.0  # E inputs per cell from each adjacent unit (owner ruling 2026-11-09: p 0.01 of 8000)

# First stack, three full units (owner rulings 2026-11-09)
MODULES = ("S", "M", "D")
RHO = {"S": 0.25, "M": 1.0, "D": 2.0}
P_LOCAL = 0.07
P_INTER = 0.01


def unit_sizes(n_e: int, rho: float) -> dict[str, int]:
    n_inh = int(round(n_e * INH_PER_E))
    n_sst = int(round(n_inh * rho / (1.0 + rho)))
    return {"E": int(n_e), "PV": n_inh - n_sst, "SOM": n_sst}


def module_sizes(rho: dict[str, float] = RHO) -> dict[str, dict[str, int]]:
    return {m: c0.sizes(c0.rho_to_nsst(rho[m])) for m in MODULES}


MEAN_NE_BIN = 1600  # mean E cells per bin in the 21-bin stack (smallest bin keeps p <= 0.77)
RHO_LO, RHO_HI = 0.25, 2.0  # ramp ends at the top and bottom bin (owner rulings C1/C4, 2026-11-09)


def lansa_bins() -> tuple[np.ndarray, np.ndarray]:
    """(bin centers rel. mid-L4, median over areas of total LANSA density per bin)."""
    import json
    from pathlib import Path

    d = json.loads((Path(__file__).resolve().parents[2] / "docs/laminar_motif/data/lansa_density_X.json")
                   .read_text(encoding="utf-8"))
    X = np.asarray(d["X"], dtype=float)  # [bin, type, area]
    return np.round(np.asarray(d["axes"]["bin_center_rel_L4"]), 3), np.median(X.sum(axis=1), axis=1)


def layout(name: str) -> tuple[dict[str, dict[str, int]], dict[str, float], float | None, dict[str, float]]:
    """(sizes, rho, p_inter, depth) for a named layout. depth is the unit's position rel. mid-L4."""
    if name == "e2_3unit":
        return module_sizes(), dict(RHO), P_INTER, {"S": -0.4, "M": 0.0, "D": 0.3}
    if name.startswith("unit") and name[4:].isdigit():  # one small unit at rho 1: finite-size check against E0
        return {"U": unit_sizes(int(name[4:]), 1.0)}, {"U": 1.0}, None, {"U": 0.0}
    if name == "e2_21bin":
        z, dens = lansa_bins()
        frac = (z - z[0]) / (z[-1] - z[0])
        rho = RHO_LO * (RHO_HI / RHO_LO) ** frac  # log-linear ramp, low superficial, high deep
        n_e = np.round(MEAN_NE_BIN * dens / dens.mean()).astype(int)
        names = [f"b{i:02d}" for i in range(z.size)]
        sizes = {nm: unit_sizes(int(n_e[i]), float(rho[i])) for i, nm in enumerate(names)}
        return sizes, {nm: float(rho[i]) for i, nm in enumerate(names)}, None, {nm: float(z[i]) for i, nm in enumerate(names)}
    raise ValueError(f"unknown layout {name!r}")


def neuron_rows(sizes: dict[str, dict[str, int]]) -> list[dict]:
    """Rows in unit order, each E, PV, SOM; area 'C', layer = unit name."""
    rows: list[dict] = []
    for m in sizes:
        for t in POPS:
            start = len(rows)
            rows += [{"neuron_id": start + i, "area": "C", "layer": m, "cell_type": t} for i in range(sizes[m][t])]
    return rows


def build_config(seed: int, sizes: dict[str, dict[str, int]], p_inter: float | None = None,
                 p_local: dict[str, float] | None = None):
    """p_local: per-unit within-unit probability (default K_E / N_E). p_inter: fixed inter-unit probability
    (the 3-unit stack's 0.01); None means K_INTER / N_E of the source unit."""
    import jaxfne as jtfne

    names = list(sizes)
    n = sum(sum(s.values()) for s in sizes.values())
    tot = {t: sum(sizes[m][t] for m in names) for t in POPS}
    cfg = (
        jtfne.Configuration()
        .network(name="caminos_stack", n=n, cell_types={t: tot[t] / n for t in POPS})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": c0.TAU_E_MS, "reversal_mV": c0.REV_E_MV})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": c0.TAU_I_MS, "reversal_mV": c0.REV_I_MV})
    )
    for m in names:
        p = (p_local or {}).get(m, min(1.0, K_E / sizes[m]["E"]))
        for name, pre, post, weight, mech in c0.RULES:
            cfg = cfg.connections(
                name=f"{name}_{m}", source={"cell_type": pre, "layer": m}, target={"cell_type": post, "layer": m},
                probability=float(p), weight=weight, mechanism=mech, delay_ms=c0.DELAY_MS, sampling="bernoulli",
            )
    n_nb = {m: (len(names) > 1) * (2 - (m in (names[0], names[-1]))) for m in names}
    for i in range(len(names) - 1):
        for a, b in ((names[i], names[i + 1]), (names[i + 1], names[i])):
            # in-degree 2 K_INTER per target cell: an end unit takes 2 K_INTER from its one neighbour
            p = p_inter if p_inter is not None else min(1.0, 2 * K_INTER / n_nb[b] / sizes[a]["E"])
            if p <= 0:
                continue
            for post, weight in (("E", c0.Q_E_E), ("PV", c0.Q_E_INH), ("SOM", c0.Q_E_INH)):
                cfg = cfg.connections(
                    name=f"x_{a}_{b}_e_{post.lower()}", source={"cell_type": "E", "layer": a},
                    target={"cell_type": post, "layer": b}, probability=float(p), weight=weight,
                    mechanism="ampa", delay_ms=c0.DELAY_MS, sampling="bernoulli",
                )
    return cfg.connectivity(edge_seed=int(seed))


def construct(seed: int, sizes: dict[str, dict[str, int]], p_inter: float | None = None,
              p_local: dict[str, float] | None = None):
    from jaxfne._construct_adex import construct_adex_model

    rows = neuron_rows(sizes)
    model = construct_adex_model(build_config(seed, sizes, p_inter, p_local), neuron_rows=rows,
                                 positions=np.zeros((len(rows), 3)))
    labels = tuple(model.params["emitter"].labels)
    if labels != tuple(r["cell_type"] for r in rows):
        raise RuntimeError("emitter labels do not follow the row order")
    return model, rows


def block_slices(sizes: dict[str, dict[str, int]]) -> dict[tuple[str, str], slice]:
    out, i = {}, 0
    for m in sizes:
        for t in POPS:
            out[(m, t)] = slice(i, i + sizes[m][t])
            i += sizes[m][t]
    return out


def lfp_weights(model, rows, names) -> np.ndarray:
    """[N, 2 * n_units] summed outgoing weight (nS) onto E cells of each unit, AMPA then GABA columns.

    Conductance onto unit m's E cells is the exponential filter of spikes @ W[:, col] delayed by 2 ms;
    the external Poisson drive is not included (unrecorded, stationary).
    """
    st = model.static["adex"]
    pre = np.asarray(st["pre"])
    post = np.asarray(st["post"])
    w = np.abs(np.asarray(st["weight_nS"], dtype=np.float64))
    rec = np.asarray(st["receptor"])
    rnames = list(st["receptor_names"])
    lay = np.array([r["layer"] for r in rows])
    typ = np.array([r["cell_type"] for r in rows])
    W = np.zeros((len(rows), 2 * len(names)))
    for k, m in enumerate(names):
        on_e = (lay[post] == m) & (typ[post] == "E")
        for j, rname in enumerate(("ampa", "gaba")):
            sel = on_e & (rec == rnames.index(rname))
            np.add.at(W[:, 2 * k + j], pre[sel], w[sel])
    return W


def edge_table(model) -> dict[str, int]:
    st = model.static["adex"]
    rid = np.asarray(st["rule_id"])
    names = [r[0] for r in st["rules"]]
    counts = np.bincount(rid, minlength=len(names))
    return {names[i]: int(counts[i]) for i in range(len(names))}
