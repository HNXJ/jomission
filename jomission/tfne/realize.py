"""Realize NF(A) in JaxFNE and reconcile the realized graph against it.

    NF(A) -> one Configuration.connections() declaration per projection -> construct() -> receipt

`connect_columns` is not used: docs/tfne/07_JAXFNE_REALIZATION_BOUNDARY.md records that one
directional call emits a compound motif and realizes no reciprocal edges, so it cannot be
reconciled one-to-one with a declared projection set. `build_multi_area_columns` appears only as
a zero-connectivity population scaffold, which the same document shows it is at p = 0.

Reconciliation compares projection *identities* first and edge counts second, as the review
required: a matching edge count over a mismatched identity set is not a realization.
"""

from __future__ import annotations

import collections

import jaxfne as jtfne
import numpy as np

from jomission.tfne import ctx
from jomission.tfne.nf import natural_key

AREA_SEP = "_"          # jaxfne area names cannot carry the TFNE '.' separator


def area_name(path):
    """TFNE object path -> JaxFNE area name, injectively. V1.1 -> V1_1."""
    return AREA_SEP.join(path)


def scaffold(nf, n_per_instance=ctx.N_PER_INSTANCE, seed=0):
    """Populations only. p_feedforward = p_feedback = 0 realizes no cross-area topology."""
    areas = tuple(area_name(tuple(o["path"])) for o in nf["objects"])
    return jtfne.build_multi_area_columns(
        areas=areas, n_per_area=int(n_per_instance), layers=tuple(ctx.LAYERS),
        connectivity_mode="sparse", ei_profile="canonical",
        p_feedforward=0.0, p_feedback=0.0), areas


#: tau_ms mirrors the compiler's legacy hardcoded excitatory default, so declaring the mechanism
#: changes which compiler runs without changing the realized synaptic time constant.
MECHANISM_TAU_MS = {"AMPA": 2.0}


def declare_mechanisms(cfg, nf):
    """Declare every mechanism NF(A) names.

    Required for realization fidelity, not convenience: with no resolvable mechanism declaration
    the compiler falls back to inferring the receptor from the weight sign, so a declared AMPA
    projection would be configured but not realized as AMPA. The declaration also selects
    connectivity.compile_connection_rules, whose selector matches layers by strict equality --
    _construct_connectivity._interarea_layer_set expands any superficial layer name to
    {L2, L3, L2/3, L23}, which cannot address CTX's separate L2.E and L3.E populations.
    """
    for mech in sorted({p.mechanism for p in nf["_projection_objects"]}, key=natural_key):
        if mech not in MECHANISM_TAU_MS:
            raise ValueError(f"E_MECHANISM_UNRESOLVED: no tau declared for {mech!r}")
        cfg = cfg.mechanisms(name=mech, kind=mech.lower(), params={"tau_ms": MECHANISM_TAU_MS[mech]})
    return cfg


def declare_projections(cfg, nf, weight=0.5, probability=1.0):
    """One connections() declaration per generated projection. Nothing else is added."""
    for i, p in enumerate(nf["_projection_objects"]):
        if p.source.path == ("Retina",):
            continue                      # x is declared structurally, not constructed here
        cfg = cfg.connections(
            name=f"tfne_{i:03d}_{p.relation.replace('[', '_').replace(']', '')}",
            source={"area": area_name(p.source.path), "layer": p.source.layer,
                    "cell_type": p.source.cls},
            target={"area": area_name(p.target.path), "layer": p.target.layer,
                    "cell_type": p.target.cls},
            probability=probability, weight=weight, mechanism=p.mechanism)
    return cfg


def construct(cfg, seed=0, dt_ms=0.1):
    """Full staging chain; construct() fails loudly without emitter and field declarations."""
    cfg = (cfg.runtime(seed=seed, duration_ms=1000.0, dt_ms=dt_ms, dtype="float32")
              .set_emitter("izhikevich", "cortical_eig")
              .probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=16)
              .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
    return jtfne.construct(cfg)


def index_map(model):
    """I: TFNE address -> realized neuron indices, and back. Derived, never authored."""
    table = model.neuron_table()
    area = np.array([str(r["area"]) for r in table])
    layer = np.array([str(r["layer"]) for r in table])
    cls = np.array([str(r["cell_type"]) for r in table])
    index = {}
    for a in sorted(set(area.tolist()), key=natural_key):
        for lay in sorted(set(layer.tolist()), key=natural_key):
            for c in sorted(set(cls.tolist()), key=natural_key):
                idx = np.flatnonzero((area == a) & (layer == lay) & (cls == c))
                if idx.size:
                    index[f"{a}.{lay}.{c}"] = (int(idx.min()), int(idx.max()) + 1, int(idx.size))
    return index, (area, layer, cls)


def realized_projections(model, area, layer, cls):
    """Identity set actually present in the constructed edge list, with edge counts."""
    el = model.params["edge_list"]
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    counts = collections.Counter(
        (f"{area[a]}.{layer[a]}.{cls[a]}", f"{area[b]}.{layer[b]}.{cls[b]}")
        for a, b in zip(pre, post) if area[a] != area[b])
    return counts, int(pre.size)


def expected_projections(nf):
    """Identity set NF(A) requires, in JaxFNE address form. Retina is structural, not built."""
    out = {}
    for p in nf["_projection_objects"]:
        if p.source.path == ("Retina",):
            continue
        key = (f"{area_name(p.source.path)}.{p.source.layer}.{p.source.cls}",
               f"{area_name(p.target.path)}.{p.target.layer}.{p.target.cls}")
        out[key] = p.relation
    return out


def mechanism_receipt(model, area, layer, cls, nf):
    """Per-projection realized receptor indices and tau values.

    The compiled-rule table is not carried on the Model, so mechanism realization is read from
    the edge list itself: what a declared mechanism became, not what was requested.
    """
    el = model.params["edge_list"]
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    rec = np.asarray(el.receptor_index)
    tau = np.asarray(el.tau_ms)
    # The scaffold's own edges carry no tau, and concatenation leaves the field empty for the
    # whole list. Report that rather than inventing a per-edge value.
    tau_carried = tau.size == rec.size
    want = expected_projections(nf)
    out = {}
    for (s, t) in want:
        sa, sl, sc = s.split(".")
        ta, tl, tc = t.split(".")
        m = ((area[pre] == sa) & (layer[pre] == sl) & (cls[pre] == sc)
             & (area[post] == ta) & (layer[post] == tl) & (cls[post] == tc))
        idx = np.flatnonzero(m)
        out[f"{s} -> {t}"] = {
         "n_edges": int(idx.size),
         "receptor_index": sorted({int(v) for v in rec[idx]}),
         "tau_ms": sorted({round(float(v), 6) for v in tau[idx]}) if tau_carried else None}
    return out, {"tau_carried_per_edge": bool(tau_carried),
                 "tau_ms_field_size": int(tau.size), "n_edges_total": int(rec.size),
                 "declared": dict(MECHANISM_TAU_MS)}
