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

import math

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


def _prob(x: float, what: str) -> float:
    """x as a connection probability. Raises when x > 1: a clip at 1.0 would silently change the in-degree."""
    if x <= 1.0:
        return x
    raise ValueError(f"{what} = {x:.6g} > 1: connection probability would clip at 1.0")


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


DEEP_UNIT = "D"  # the 3-unit stack's deep unit
DEEP_E = ("C", DEEP_UNIT, "E")  # (area, layer, cell type) of the deep unit's E cells
E_SIZE_FIELDS = ("C_pF", "gL_nS", "a_nS", "b_pA")  # scale with membrane area; tau_m unchanged


DRIVE_UNITS = ("a", "S", "M", "D")  # a = every unit; S, M, D = the 3-unit stack's units


def _drive_token(rest: str, tok: str) -> tuple[str, float]:
    """(unit, pA) of an ipvUV / isomUV token; rest is the text after the key. 'm' is minus, 'p' the point."""
    if not rest or rest[0] not in DRIVE_UNITS:
        raise ValueError(f"drive modifier {tok!r}: unit must be one of {'/'.join(DRIVE_UNITS)}")
    try:
        value = float(rest[1:].replace("m", "-").replace("p", "."))
    except ValueError:
        raise ValueError(f"drive modifier {tok!r}: {rest[1:]!r} is not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"drive modifier {tok!r}: value must be finite")
    return rest[0], value


def variant(name: str) -> tuple[str, dict]:
    """Split a layout name into (base, modifiers). Modifiers are '_'-separated tokens after the base
    (e2_3unit, e2_21bin, unitNNNN); 'p' is the decimal point, 'm' a minus sign (ipv, isom only):
      sstNN  SST->E decay NN ms on its own receptor, equal charge (build_config)
      sstdNN SST->E decay NN ms in the deep unit (DEEP_UNIT) only; overrides sstNN there (sst_taus)
      isomUV constant current V pA on the SOM cells of unit U (drive_entries); U in a (every unit), S, M, D;
             V signed: isomD50 = +50 pA on deep SOM, isomam10 = -10 pA on every SOM. Stored as {U: V};
             a unit given twice is refused, and so is an overlap of a with a named unit (drive_entries)
      ipvUV  as isomUV, for the PV cells
      rhosX  every unit's rho times X (3-unit stack; applied after rhod, so PV+SOM per unit is unchanged)
      rhodX  deep unit's N_SST / N_PV set to X (3-unit stack; default RHO["D"] 2)
      giX    every GABA rule weight (pv_e, pv_pv, sst_e, sst_pv) times X, after the sstNN charge rescale
      nNNNN  mean E cells per unit (3-unit stack)
      dens   E cells per unit proportional to LANSA total density of the unit's bins (3-unit stack)
      kX     inter-unit in-degree K_INTER * X
      kdB    local in-degree K_E * (density / mean density)^B per unit: denser units get more synapses
             per cell (3-unit stack; see local_p)
      dszX   deep-unit E cells X times larger: C, gL, a, b scaled by X, synapses and drive unchanged
    """
    toks = name.split("_")
    nb = 2 if name.startswith("e2_") else 1
    base, mods = "_".join(toks[:nb]), {}
    for t in toks[nb:]:
        # longer keys first: sstd before sst, kd before k (the other keys do not share a prefix)
        for key in ("sstd", "isom", "rhos", "rhod", "dens", "sst", "ipv", "dsz", "gi", "kd", "n", "k"):
            if t.startswith(key):
                rest = t[len(key):]
                if key in ("isom", "ipv"):
                    unit, value = _drive_token(rest, t)
                    if unit in mods.get(key, {}):
                        raise ValueError(f"{key} declared twice for unit {unit!r} in {name!r}")
                    mods.setdefault(key, {})[unit] = value
                else:
                    if key in mods:
                        raise ValueError(f"{key} declared twice in {name!r}")
                    v = 1.0 if key == "dens" and not rest else float(rest.replace("p", "."))
                    if not math.isfinite(v):
                        raise ValueError(f"modifier {t!r} in {name!r}: value must be finite")
                    mods[key] = v
                break
        else:
            raise ValueError(f"unknown layout modifier {t!r} in {name!r}")
    return base, mods


def local_p(name: str, sizes: dict[str, dict[str, int]]) -> dict[str, float] | None:
    """Per-unit within-unit p for the 'kdB' modifier (None otherwise): in-degree K_E * rel^B, rel = the
    unit's LANSA density over the mean of the 3 units; applies to every local rule (E and inhibitory)."""
    mods = variant(name)[1]
    if "kd" not in mods:
        return None
    dens = unit_density(MODULE_BINS)
    mean = np.mean([dens[m] for m in sizes])
    return {m: _prob(K_E * (dens[m] / mean) ** mods["kd"] / sizes[m]["E"], f"local p (kd) of unit {m}") for m in sizes}


def sst_taus(name: str, units) -> dict[str, float] | None:
    """Per-unit SST->E decay (ms) from 'sstNN' and 'sstdNN'; units without one keep CAMINOS gaba. None if
    neither modifier is given."""
    mods = variant(name)[1]
    taus = {m: mods.get("sstd", mods.get("sst")) if m == DEEP_UNIT else mods.get("sst") for m in units}
    taus = {m: t for m, t in taus.items() if t is not None}
    return taus or None


def drive_entries(name: str, units) -> list[dict] | None:
    """tensor_triple_drive entries of the ipvUV (PV) and isomUV (SOM) modifiers, or None. One entry per
    (unit, cell type) in layer = unit, area 'C'; a = every unit in units. A (unit, cell type) covered twice
    (by a and a named unit) is refused rather than summed."""
    mods = variant(name)[1]
    out: dict[tuple[str, str], float] = {}
    for key, cell in (("ipv", "PV"), ("isom", "SOM")):
        for u, v in mods.get(key, {}).items():
            for m in (list(units) if u == "a" else [u]):
                if m not in units:
                    raise ValueError(f"{key}{u} names unit {m!r}, which is not in {list(units)}")
                if (m, cell) in out:
                    raise ValueError(f"{name!r}: {cell} cells of unit {m!r} are driven twice")
                out[(m, cell)] = v
    if not out:
        return None
    return [{"area": "C", "layer": m, "cell_type": cell, "drive": v} for (m, cell), v in out.items()]


def unit_density(depth_groups: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean over the group's bins of the median-over-areas total LANSA density."""
    z, dens = lansa_bins()
    return {m: float(dens[g(z)].mean()) for m, g in depth_groups.items()}


MODULE_BINS = {"S": lambda z: z < -0.175, "M": lambda z: (z >= -0.175) & (z <= 0.175), "D": lambda z: z > 0.175}


def layout(name: str) -> tuple[dict[str, dict[str, int]], dict[str, float], float | None, dict[str, float]]:
    """(sizes, rho, p_inter, depth) for a named layout (see variant). depth is the unit's position rel.
    mid-L4. The 3-unit stack keeps the owner's fixed p 0.01 unless n, dens or k is given; then the
    inter-unit in-degree is K_INTER-based (p None)."""
    name, mods = variant(name)
    if "rhos" in mods and name != "e2_3unit":
        raise ValueError(f"rhos applies to the 3-unit stack (e2_3unit), not {name!r}")
    if name == "e2_3unit":
        depth = {"S": -0.4, "M": 0.0, "D": 0.3}
        rho = {**RHO, DEEP_UNIT: mods.get("rhod", RHO[DEEP_UNIT])}
        rho = {m: r * mods.get("rhos", 1.0) for m, r in rho.items()}
        if not ({"n", "dens", "k"} & set(mods)):
            return module_sizes(rho), rho, P_INTER, depth
        n_mean = mods.get("n", c0.NE)
        dens = unit_density(MODULE_BINS) if "dens" in mods else {m: 1.0 for m in MODULES}
        mean = np.mean(list(dens.values()))
        sizes = {m: unit_sizes(int(round(n_mean * dens[m] / mean)), rho[m]) for m in MODULES}
        return sizes, rho, None, depth
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
                 p_local: dict[str, float] | None = None, tau_sst_e_ms: float | dict[str, float] | None = None,
                 k_scale: float = 1.0, gaba_scale: float = 1.0, triple_drive: list[dict] | None = None):
    """p_local: per-unit within-unit probability (default K_E / N_E). p_inter: fixed inter-unit probability
    (the 3-unit stack's 0.01); None means K_INTER / N_E of the source unit. tau_sst_e_ms (one value, or per
    unit): SST->E synapses on their own receptor with this decay ('gaba_sst' when one value, else
    'gaba_sstNN'), weight scaled by TAU_I / tau so the charge per spike is unchanged; units without one keep
    CAMINOS (gaba, 8 ms). gaba_scale multiplies every GABA rule weight after that rescale. triple_drive:
    tensor_triple_drive entries (drive_entries), stored as config metadata; None writes nothing."""
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
    taus = tau_sst_e_ms if isinstance(tau_sst_e_ms, dict) else (
        {} if tau_sst_e_ms is None else {m: tau_sst_e_ms for m in names})
    distinct = sorted({float(t) for t in taus.values()})
    rname = {t: "gaba_sst" if len(distinct) == 1 else f"gaba_sst{t:g}".replace(".", "p") for t in distinct}
    for t in distinct:
        cfg = cfg.mechanisms(name=rname[t], kind="receptor", params={"tau_ms": t, "reversal_mV": c0.REV_I_MV})
    for m in names:
        p = (p_local or {}).get(m)
        if p is None:
            p = _prob(K_E / sizes[m]["E"], f"local p of unit {m}")
        for name, pre, post, weight, mech in c0.RULES:
            scale = gaba_scale if mech == "gaba" else 1.0  # read before the SST receptor replaces the mechanism
            if name == "sst_e" and m in taus:
                weight, mech = weight * c0.TAU_I_MS / taus[m], rname[float(taus[m])]
            cfg = cfg.connections(
                name=f"{name}_{m}", source={"cell_type": pre, "layer": m}, target={"cell_type": post, "layer": m},
                probability=float(p), weight=weight * scale, mechanism=mech, delay_ms=c0.DELAY_MS, sampling="bernoulli",
            )
    n_nb = {m: (len(names) > 1) * (2 - (m in (names[0], names[-1]))) for m in names}
    for i in range(len(names) - 1):
        for a, b in ((names[i], names[i + 1]), (names[i + 1], names[i])):
            # in-degree 2 K_INTER per target cell: an end unit takes 2 K_INTER from its one neighbour
            p = p_inter if p_inter is not None else _prob(
                2 * K_INTER * k_scale / n_nb[b] / sizes[a]["E"], f"inter-unit p {a}->{b}")
            if p <= 0:
                continue
            for post, weight in (("E", c0.Q_E_E), ("PV", c0.Q_E_INH), ("SOM", c0.Q_E_INH)):
                cfg = cfg.connections(
                    name=f"x_{a}_{b}_e_{post.lower()}", source={"cell_type": "E", "layer": a},
                    target={"cell_type": post, "layer": b}, probability=float(p), weight=weight,
                    mechanism="ampa", delay_ms=c0.DELAY_MS, sampling="bernoulli",
                )
    if triple_drive:
        cfg = cfg.update_metadata(tensor_triple_drive=list(triple_drive))
    return cfg.connectivity(edge_seed=int(seed))


def construct(seed: int, sizes: dict[str, dict[str, int]], p_inter: float | None = None,
              p_local: dict[str, float] | None = None, tau_sst_e_ms: float | dict[str, float] | None = None,
              k_scale: float = 1.0, deep_size: float = 1.0, gaba_scale: float = 1.0,
              triple_drive: list[dict] | None = None):
    from jaxfne._construct_adex import construct_adex_model
    from jaxfne.emitters_adex import ADEX_CELL_TYPE_DEFAULTS

    rows = neuron_rows(sizes)
    over = None
    if deep_size != 1.0:
        over = {DEEP_E: {f: ADEX_CELL_TYPE_DEFAULTS["E"][f] * deep_size for f in E_SIZE_FIELDS}}
    model = construct_adex_model(build_config(seed, sizes, p_inter, p_local, tau_sst_e_ms, k_scale, gaba_scale,
                                              triple_drive),
                                 neuron_rows=rows, positions=np.zeros((len(rows), 3)), param_overrides=over)
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


def lfp_weights(model, rows, names) -> tuple[np.ndarray, list[tuple[str, str]], np.ndarray, np.ndarray]:
    """(W [N, n_units * n_receptors], columns [(unit, receptor)], tau_ms, reversal_mV per column): summed
    outgoing weight (nS) onto E cells of each unit, one column per declared receptor, unit-major.

    Conductance onto unit m's E cells is the exponential filter of spikes @ W[:, col] delayed by 2 ms;
    the external Poisson drive is not included (unrecorded, stationary).
    """
    st = model.static["adex"]
    pre = np.asarray(st["pre"])
    post = np.asarray(st["post"])
    w = np.abs(np.asarray(st["weight_nS"], dtype=np.float64))
    rec = np.asarray(st["receptor"])
    rnames = list(st["receptor_names"])
    tau = np.asarray(st["receptor_tau_ms"], dtype=np.float64)
    rev = np.asarray(st["receptor_reversal_mV"], dtype=np.float64)
    lay = np.array([r["layer"] for r in rows])
    typ = np.array([r["cell_type"] for r in rows])
    cols = [(m, r) for m in names for r in rnames]
    W = np.zeros((len(rows), len(cols)))
    for c, (m, rname) in enumerate(cols):
        sel = (lay[post] == m) & (typ[post] == "E") & (rec == rnames.index(rname))
        np.add.at(W[:, c], pre[sel], w[sel])
    ri = [rnames.index(r) for _, r in cols]
    return W, cols, tau[ri], rev[ri]


def edge_table(model) -> dict[str, int]:
    st = model.static["adex"]
    rid = np.asarray(st["rule_id"])
    names = [r[0] for r in st["rules"]]
    counts = np.bincount(rid, minlength=len(names))
    return {names[i]: int(counts[i]) for i in range(len(names))}
