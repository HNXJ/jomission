"""Allen V1 core column (Herrera et al. 2026) in CAMINOS/JaxFNE form (docs/laminar_motif/ALLEN_V1_REPLICA_SPEC.md).

19 Allen populations, core counts and pmax from docs/laminar_motif/data/allen_v1_blocks.json
(scripts/allen_v1_import.py). Point-neuron probability p_eff = pmax * g(pre ei, post ei); in-degree K = p_eff * N_pre
at full size, kept under scale (p = K / N_pre_scaled). CAMINOS AdEx cells and conductance synapses; weights W1
(each population's mean recurrent conductance per receptor equals a CAMINOS cell's: 560 E, 105 inhibitory inputs).
Wiring H removes PV->SST and SST->SST within a layer (one-way SST->PV); A0 keeps every Allen block.
VIP and L1 Inh take the SOM parameter set and label (V1: undriven). Built with construct_adex_model on neuron rows
whose layer label is the population key; the CAMINOS stack helpers are reused by import.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jomission.laminar import caminos_e0 as c0
from jomission.laminar import caminos_stack as cs

DATA = Path(__file__).resolve().parents[2] / "docs" / "laminar_motif" / "data" / "allen_v1_blocks.json"
K_CAMINOS_E = c0.NE * 0.07  # 560
K_CAMINOS_I = c0.N_INH * 0.07  # 105
SILENCE = {"VT_mV": 500.0, "Vpeak_mV": 1000.0}  # spec amendment: no spike below E_AMPA 0 mV
AREA = "C"
E_LAYERS = ("L2/3", "L4", "L5", "L6")  # layers with E cells (LFP sources)


def load() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def layer_of(pop: str) -> str:
    return pop.split("_")[0]


def kind(pop: str) -> str:
    """E, PV, SST or VIP (L1_Inh counts as VIP: Htr3a, cell_type_synaptic 'vip' in the Allen naming table)."""
    s = pop.split("_")[1]
    return {"Exc": "E", "ET": "E", "IT": "E", "NP": "E", "PV": "PV", "SST": "SST", "VIP": "VIP", "Inh": "VIP"}[s]


def label(pop: str) -> str:
    return {"E": "E", "PV": "PV", "SST": "SOM", "VIP": "SOM"}[kind(pop)]


def tag(pop: str) -> str:
    return pop.replace("/", "")


def swap_map(data: dict, a: str = "L4", b: str = "L6") -> dict[str, str]:
    """Population of layer a <-> population of layer b with the same kind (one per kind in L4 and L6)."""
    pops = data["populations"]
    out = {p: p for p in pops}
    for k in ("E", "PV", "SST", "VIP"):
        pa = [p for p in pops if layer_of(p) == a and kind(p) == k]
        pb = [p for p in pops if layer_of(p) == b and kind(p) == k]
        if len(pa) != 1 or len(pb) != 1:
            raise ValueError(f"swap needs one {k} population in {a} and {b}")
        out[pa[0]], out[pb[0]] = pb[0], pa[0]
    return out


def full_sizes(data: dict, swap: bool = False) -> dict[str, int]:
    m = swap_map(data) if swap else {p: p for p in data["populations"]}
    return {p: int(data["pops"][m[p]]["N"]) for p in data["populations"]}


def p_eff(data: dict, wiring: str = "H", swap: bool = False) -> dict[tuple[str, str], float]:
    """Point-neuron probability per (pre, post). swap: within-layer blocks of L4 and L6 exchanged (P Suppl Fig 9)."""
    if wiring not in ("H", "A0"):
        raise ValueError(f"wiring {wiring!r}; expected 'H' or 'A0'")
    pops = data["populations"]
    m = swap_map(data) if swap else {p: p for p in pops}
    out = {}
    for pre in pops:
        for post in pops:
            src_pre, src_post = (m[pre], m[post]) if layer_of(pre) == layer_of(post) else (pre, post)
            ei = data["pops"][pre]["ei"] + data["pops"][post]["ei"]
            p = data["pmax"][src_pre][src_post] * data["g"][ei]
            if wiring == "H" and layer_of(pre) == layer_of(post) and kind(post) == "SST" and kind(pre) in ("PV", "SST"):
                p = 0.0
            out[(pre, post)] = p
    return out


def in_degree(p: dict, n_full: dict[str, int]) -> dict[tuple[str, str], float]:
    return {(a, b): v * n_full[a] for (a, b), v in p.items()}


def weights_w1(K: dict, pops: list[str]) -> dict[tuple[str, str], float]:
    """nS per synapse: E pre -> Q * 560 / K_E(post); inhibitory pre -> Q_INH * 105 / K_I(post)."""
    w = {}
    for post in pops:
        ke = sum(K[(a, post)] for a in pops if kind(a) == "E")
        ki = sum(K[(a, post)] for a in pops if kind(a) != "E")
        q_e = c0.Q_E_E if kind(post) == "E" else c0.Q_E_INH
        for a in pops:
            if kind(a) == "E":
                w[(a, post)] = q_e * K_CAMINOS_E / ke if ke > 0 else 0.0
            else:
                w[(a, post)] = c0.Q_INH * K_CAMINOS_I / ki if ki > 0 else 0.0
    return w


def plan(wiring: str = "H", scale: float = 1.0, swap: bool = False, data: dict | None = None) -> dict:
    """Sizes, per-block probability (in-degree kept under scale) and W1 weights."""
    data = data or load()
    pops = data["populations"]
    n_full = full_sizes(data, swap)
    sizes = {p: max(1, int(round(n_full[p] * scale))) for p in pops}
    pe = p_eff(data, wiring, swap)
    K = in_degree(pe, n_full)
    prob = {k: min(1.0, K[k] / sizes[k[0]]) for k in pe}
    capped = sorted(f"{a}->{b}" for (a, b) in pe if K[(a, b)] > sizes[a])
    return {"pops": pops, "sizes": sizes, "n_full": n_full, "p_eff": pe, "K": K, "prob": prob,
            "w": weights_w1(K, pops), "capped": capped, "wiring": wiring, "scale": scale, "swap": swap}


def neuron_rows(pl: dict) -> list[dict]:
    rows: list[dict] = []
    for p in pl["pops"]:
        start = len(rows)
        rows += [{"neuron_id": start + i, "area": AREA, "layer": tag(p), "cell_type": label(p)}
                 for i in range(pl["sizes"][p])]
    return rows


def rule_name(pre: str, post: str) -> str:
    return f"{tag(pre)}__{tag(post)}"


def build_config(seed: int, pl: dict):
    import jaxfne as jtfne

    pops, sizes = pl["pops"], pl["sizes"]
    n = sum(sizes.values())
    tot: dict[str, int] = {}
    for p in pops:
        tot[label(p)] = tot.get(label(p), 0) + sizes[p]
    cfg = (
        jtfne.Configuration()
        .network(name="allen_v1_replica", n=n, cell_types={t: v / n for t, v in tot.items()})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": c0.TAU_E_MS, "reversal_mV": c0.REV_E_MV})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": c0.TAU_I_MS, "reversal_mV": c0.REV_I_MV})
    )
    for (pre, post), p in pl["prob"].items():
        if p <= 0:
            continue
        cfg = cfg.connections(
            name=rule_name(pre, post),
            source={"cell_type": label(pre), "layer": tag(pre)}, target={"cell_type": label(post), "layer": tag(post)},
            probability=float(p), weight=float(pl["w"][(pre, post)]),
            mechanism="ampa" if kind(pre) == "E" else "gaba", delay_ms=c0.DELAY_MS, sampling="bernoulli",
        )
    return cfg.connectivity(edge_seed=int(seed))


def silence_targets(spec: str | None, pops: list[str]) -> list[str]:
    """None; 'PV' | 'SST' | 'VIP' (kind, all layers); 'layer:L4'; 'isolate:L4' (all other layers)."""
    if not spec:
        return []
    if spec in ("E", "PV", "SST", "VIP"):
        return [p for p in pops if kind(p) == spec]
    how, _, lay = spec.partition(":")
    if lay not in {layer_of(p) for p in pops}:
        raise ValueError(f"unknown layer {lay!r}")
    if how == "layer":
        return [p for p in pops if layer_of(p) == lay]
    if how == "isolate":
        return [p for p in pops if layer_of(p) != lay]
    raise ValueError(f"unknown silence spec {spec!r}")


def construct(seed: int, pl: dict, silenced: list[str] = ()):
    from jaxfne._construct_adex import construct_adex_model

    rows = neuron_rows(pl)
    over = {(AREA, tag(p), label(p)): dict(SILENCE) for p in silenced} or None
    model = construct_adex_model(build_config(seed, pl), neuron_rows=rows, positions=np.zeros((len(rows), 3)),
                                 param_overrides=over)
    if tuple(model.params["emitter"].labels) != tuple(r["cell_type"] for r in rows):
        raise RuntimeError("emitter labels do not follow the row order")
    return model, rows


def pop_slices(pl: dict) -> dict[str, slice]:
    out, i = {}, 0
    for p in pl["pops"]:
        out[p] = slice(i, i + pl["sizes"][p])
        i += pl["sizes"][p]
    return out


def edge_check(model, pl: dict, z_max: float = 5.0) -> dict:
    """Realized edges per block vs p * N_pre * N_post (Bernoulli); also vs p * N (N - 1) for a block onto itself
    (in case autapses are excluded). A block fails when both |z| exceed z_max. Examined = blocks with p > 0."""
    got = cs.edge_table(model)
    rows, bad = [], []
    for (pre, post), p in pl["prob"].items():
        if p <= 0:
            continue
        name = rule_name(pre, post)
        npre, npost = pl["sizes"][pre], pl["sizes"][post]
        pairs = [npre * npost] + ([npre * (npre - 1)] if pre == post else [])
        obs = got.get(name, 0)
        zs = [(obs - p * m) / np.sqrt(max(p * (1 - p) * m, 1e-12)) for m in pairs]
        z = min(zs, key=abs)
        rows.append({"block": name, "p": p, "expected": p * pairs[0], "observed": obs, "z": float(z)})
        if abs(z) > z_max:
            bad.append(name)
    total = int(sum(got.values()))
    return {"examined": len(rows), "failed": bad, "edges_total": total,
            "edges_expected": float(sum(r["expected"] for r in rows)), "blocks": rows}


def lfp_weights(model, rows, pl: dict) -> tuple[np.ndarray, list[str]]:
    """[N, 2 * n_E_layers] summed outgoing weight onto each layer's E cells (AMPA, GABA); L5 = ET + IT + NP."""
    e_pops = [p for p in pl["pops"] if kind(p) == "E"]
    W = cs.lfp_weights(model, rows, [tag(p) for p in e_pops])
    out = np.zeros((W.shape[0], 2 * len(E_LAYERS)))
    for k, p in enumerate(e_pops):
        j = E_LAYERS.index(layer_of(p))
        out[:, 2 * j:2 * j + 2] += W[:, 2 * k:2 * k + 2]
    return out, list(E_LAYERS)
