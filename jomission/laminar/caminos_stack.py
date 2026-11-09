"""E2 stack: three CAMINOS E-PV-SST modules (S, M, D) coupled by adjacent-module E projections.

Owner rulings 2026-11-09 (docs/laminar_motif/PLAN_LANSA_E2.md): one CAMINOS unit per module (8000 E,
1500 inhibitory), rho = N_SST / N_PV of 0.25 (S), 1 (M), 2 (D); every within-module value is E0's
(caminos_e0.py); adjacent modules (S-M, M-D) are coupled in both directions from E to E, PV and SST
with p 0.01, the local weights and the 2 ms delay. JaxFNE 0.5.9 refuses column() for the AdEx
family, so the network is built through construct_adex_model with per-neuron rows carrying the
module as the layer label; connection selectors then compile per module.
"""

from __future__ import annotations

import numpy as np

from jomission.laminar import caminos_e0 as c0

MODULES = ("S", "M", "D")
RHO = {"S": 0.25, "M": 1.0, "D": 2.0}
P_LOCAL = 0.07
P_INTER = 0.01
ADJACENT = (("S", "M"), ("M", "S"), ("M", "D"), ("D", "M"))
POPS = c0.POPS  # E, PV, SOM


def module_sizes(rho: dict[str, float] = RHO) -> dict[str, dict[str, int]]:
    return {m: c0.sizes(c0.rho_to_nsst(rho[m])) for m in MODULES}


def neuron_rows(sizes: dict[str, dict[str, int]]) -> list[dict]:
    """Rows in module order S, M, D, each E, PV, SOM; area 'C', layer = module."""
    rows: list[dict] = []
    for m in MODULES:
        for t in POPS:
            start = len(rows)
            rows += [{"neuron_id": start + i, "area": "C", "layer": m, "cell_type": t} for i in range(sizes[m][t])]
    return rows


def build_config(seed: int, sizes: dict[str, dict[str, int]], p_inter: float = P_INTER):
    import jaxfne as jtfne

    n = sum(sum(s.values()) for s in sizes.values())
    tot = {t: sum(sizes[m][t] for m in MODULES) for t in POPS}
    cfg = (
        jtfne.Configuration()
        .network(name="caminos_e2_stack", n=n, cell_types={t: tot[t] / n for t in POPS})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": c0.TAU_E_MS, "reversal_mV": c0.REV_E_MV})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": c0.TAU_I_MS, "reversal_mV": c0.REV_I_MV})
    )
    for m in MODULES:
        for name, pre, post, weight, mech in c0.RULES:
            cfg = cfg.connections(
                name=f"{name}_{m}", source={"cell_type": pre, "layer": m}, target={"cell_type": post, "layer": m},
                probability=P_LOCAL, weight=weight, mechanism=mech, delay_ms=c0.DELAY_MS, sampling="bernoulli",
            )
    if p_inter > 0:
        for a, b in ADJACENT:
            for post, weight in (("E", c0.Q_E_E), ("PV", c0.Q_E_INH), ("SOM", c0.Q_E_INH)):
                cfg = cfg.connections(
                    name=f"x_{a}{b}_e_{post.lower()}", source={"cell_type": "E", "layer": a},
                    target={"cell_type": post, "layer": b}, probability=float(p_inter), weight=weight,
                    mechanism="ampa", delay_ms=c0.DELAY_MS, sampling="bernoulli",
                )
    return cfg.connectivity(edge_seed=int(seed))


def construct(seed: int, sizes: dict[str, dict[str, int]], p_inter: float = P_INTER):
    from jaxfne._construct_adex import construct_adex_model

    rows = neuron_rows(sizes)
    model = construct_adex_model(build_config(seed, sizes, p_inter), neuron_rows=rows,
                                 positions=np.zeros((len(rows), 3)))
    labels = tuple(model.params["emitter"].labels)
    if labels != tuple(r["cell_type"] for r in rows):
        raise RuntimeError("emitter labels do not follow the row order")
    return model, rows


def block_slices(sizes: dict[str, dict[str, int]]) -> dict[tuple[str, str], slice]:
    out, i = {}, 0
    for m in MODULES:
        for t in POPS:
            out[(m, t)] = slice(i, i + sizes[m][t])
            i += sizes[m][t]
    return out


def lfp_weights(model, rows) -> np.ndarray:
    """[N, 2 * n_modules] summed outgoing weight (nS) onto E cells of each module, AMPA then GABA columns.

    Conductance onto module m's E cells is the exponential filter of spikes @ W[:, col] delayed by
    2 ms; the external Poisson drive is not included (unrecorded, stationary).
    """
    st = model.static["adex"]
    pre = np.asarray(st["pre"])
    post = np.asarray(st["post"])
    w = np.abs(np.asarray(st["weight_nS"], dtype=np.float64))
    rec = np.asarray(st["receptor"])
    names = list(st["receptor_names"])
    lay = np.array([r["layer"] for r in rows])
    typ = np.array([r["cell_type"] for r in rows])
    W = np.zeros((len(rows), 2 * len(MODULES)))
    for k, m in enumerate(MODULES):
        on_e = (lay[post] == m) & (typ[post] == "E")
        for j, rname in enumerate(("ampa", "gaba")):
            sel = on_e & (rec == names.index(rname))
            np.add.at(W[:, 2 * k + j], pre[sel], w[sel])
    return W


def edge_table(model, rows) -> dict[str, int]:
    st = model.static["adex"]
    rid = np.asarray(st["rule_id"])
    names = [r[0] for r in st["rules"]]
    counts = np.bincount(rid, minlength=len(names))
    return {names[i]: int(counts[i]) for i in range(len(names))}
