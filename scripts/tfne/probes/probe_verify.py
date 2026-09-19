"""Verify four JaxFNE 0.4.24 realization findings by execution.

construct() only; no simulate(). Tiny synthetic tensors (labelled synthetic) plus the
shipped canonical multiarea config. Prints a JSON receipt to stdout and writes nothing.
See docs/tfne/02_JAXFNE_COMPATIBILITY.md section 3.

    python scripts/tfne/probes/probe_verify.py [--check]

--check compares against the values recorded for jaxfne 0.4.24 and exits 1 on any change.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import jaxfne as j
from jaxfne.jdna import develop, genome_rules_hash, phenotype_sha256, pseudogenome_from_dict, validate_genome
from jaxfne.neuronal_tensor import (Area, AreaConnection, InterConnection, Layer, NeuronalTensor, NeuronType,
                                    PlasticParams, StaticParams, neuronal_tensor_to_configuration)

RECORDED_0_4_24 = {
    ("A_normalization_and_D_density", "U_alone", "configured", "weight"): 0.15811388300841897,
    ("A_normalization_and_D_density", "U_with_replica_U2", "configured", "weight"): 0.11180339887498948,
    ("A_normalization_and_D_density", "U_alone", "configured", "probability"): 1.0,
    ("A_normalization_and_D_density", "U_alone", "realized_U_L2E_to_U_L4E", "n_edges"): 256,
    ("A_normalization_and_D_density", "U_with_replica_U2", "realized_U_L2E_to_U_L4E", "n_edges"): 256,
    ("B_jdna_develop_parameters", "genome_w0.5_dT3", "developed_area_connection", "w_mech"): 1.0,
    ("B_jdna_develop_parameters", "genome_w0.5_dT3", "developed_area_connection", "H"): 0.0,
    ("B_jdna_develop_parameters", "genome_w0.5_dT3", "developed_area_connection", "dT_ms"): 0.1,
    ("B_jdna_develop_parameters", "genome_w0.5_dT3", "configured", "weight"): 0.22360679774997896,
    ("B_jdna_develop_parameters", "genome_w0.5_dT3", "realized_A_to_B", "tau_ms_unique"): [0.1],
    ("B_jdna_develop_parameters", "direct_tensor_w0.5_dT3", "configured", "weight"): 0.11180339887498948,
    ("B_jdna_develop_parameters", "direct_tensor_w0.5_dT3", "realized_A_to_B", "tau_ms_unique"): [3.0],
    ("B_jdna_develop_parameters", "direct_tensor_w0.5_dT3", "realized_A_to_B", "delay_steps_unique"): [0],
    ("B_jdna_develop_parameters", "genome_hashes_differ"): True,
    ("B_jdna_develop_parameters", "phenotype_hashes_equal"): True,
    ("C_multiarea", "realized_edges"): 24750,
    ("C_multiarea", "realized_within_area_edges"): 0,
    ("C_multiarea", "config_sha256"): "97242c2f176893ac9c3a38e519543a260d0a6cc4bdb43289d69dcc9b536e6ff0",
}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--check", action="store_true", help="compare against values recorded for jaxfne 0.4.24")
args = parser.parse_args()
rec = {"jaxfne": j.__version__, "python": sys.version.split()[0], "data": "synthetic probe tensors + shipped config"}


def rule(cfg, name):
    return next(r for r in cfg.metadata["circuit"]["connections"] if r["name"] == name)


def edges(model):
    """Per-edge values as the kernel consumes them (emitters.resolve_* materializers)."""
    import jax.numpy as jnp
    from jaxfne.emitters import _resolved_edge_weight, resolve_edge_delay_steps, resolve_edge_tau_ms
    el = model.params["edge_list"]
    return {"pre": np.asarray(el.pre), "post": np.asarray(el.post),
            "weight": np.asarray(_resolved_edge_weight(el, jnp.float32, model.params["emitter"])),
            "tau_ms": np.asarray(resolve_edge_tau_ms(el, jnp.float32)),
            "delay_steps": np.asarray(resolve_edge_delay_steps(el)),
            "storage": {"weight": el.weight_storage, "tau": el.tau_storage, "delay": el.delay_storage}}


def ids(model, area, layer, ct):
    return np.array(sorted(r["neuron_id"] for r in model.neuron_table()
                           if (r["area"], r["layer"], r["cell_type"]) == (area, layer, ct)))


def block(model, src, dst):
    e = edges(model)
    s, t = ids(model, *src), ids(model, *dst)
    m = np.isin(e["pre"], s) & np.isin(e["post"], t)
    pairs = set(zip(e["pre"][m].tolist(), e["post"][m].tolist()))
    return {"n_src": int(s.size), "n_dst": int(t.size), "n_edges": int(m.sum()), "n_unique_pairs": len(pairs),
            "weight_unique": sorted({round(float(x), 9) for x in e["weight"][m]}),
            "tau_ms_unique": sorted({round(float(x), 6) for x in e["tau_ms"][m]}),
            "delay_steps_unique": sorted({int(x) for x in e["delay_steps"][m]}),
            "storage": e["storage"]}


def column_layers():
    return [Layer(name=ln, neuron_types=[NeuronType.make("E", fraction=0.8), NeuronType.make("PV", fraction=0.2)],
                  n_neurons=20) for ln in ("L2", "L4")]


def unit(name):
    return Area(name=name, layers=column_layers(),
                inter_connections=[InterConnection("L2", "E", "L4", "E", "AMPA", plastic=PlasticParams(w_mech=1.0))])


# ---- A: global 1/sqrt(N_total) normalization; D: probability 1.0 ----
A = {}
for label, areas in (("U_alone", [unit("U")]), ("U_with_replica_U2", [unit("U"), unit("U2")])):
    t = NeuronalTensor(areas=areas, area_connections=[], name=f"probe-{label}")
    cfg = neuronal_tensor_to_configuration(t, seed=0)
    r = rule(cfg, "interconn_U_0")
    n_total = sum(ly.n_neurons for a in areas for ly in a.layers)
    model = j.construct(cfg)
    A[label] = {"n_total": n_total, "configured": {"weight": r["weight"], "probability": r["probability"]},
                "expected_weight_w_over_sqrt_ntotal": 1.0 / math.sqrt(n_total),
                "realized_U_L2E_to_U_L4E": block(model, ("U", "L2", "E"), ("U", "L4", "E"))}
A["configured_weight_ratio"] = A["U_with_replica_U2"]["configured"]["weight"] / A["U_alone"]["configured"]["weight"]
wa = A["U_alone"]["realized_U_L2E_to_U_L4E"]["weight_unique"]
wb = A["U_with_replica_U2"]["realized_U_L2E_to_U_L4E"]["weight_unique"]
A["realized_weight_ratio"] = (wb[0] / wa[0]) if len(wa) == 1 and len(wb) == 1 else None
A["expected_ratio_sqrt_40_over_80"] = math.sqrt(40 / 80)
rec["A_normalization_and_D_density"] = A

# ---- B: JDNA develop drops area-connection static/plastic ----
def genome(w, dT):
    layer = {"name": "L4", "n_neurons": 10, "depth_band": [0.5, 0.6], "cell_type_fractions": {"E": 1.0}}
    return {"schema_version": "pseudogenome_v1", "name": "probe-jdna",
            "areas": [{"name": "A", "layers": [layer]}, {"name": "B", "layers": [layer]}],
            "area_connections": [{"source_area": "A", "source_layer": "L4", "source_neuron_type": "E",
                                  "target_area": "B", "target_layer": "L4", "target_neuron_type": "E",
                                  "mechanism": "monotonic_cable_synapse",
                                  "plastic": {"w_mech": w, "H": 1.0}, "static": {"dT_ms": dT}}]}


B = {}
for label, (w, dT) in (("genome_w0.5_dT3", (0.5, 3.0)), ("genome_w1.0_dT0.1", (1.0, 0.1))):
    g = pseudogenome_from_dict(genome(w, dT))
    validate_genome(g)
    t = develop(g, seed=0)
    ac = t.area_connections[0]
    cfg = neuronal_tensor_to_configuration(t, seed=0)
    r = rule(cfg, "areaconn_0")
    model = j.construct(cfg)
    B[label] = {"declared": {"w_mech": w, "dT_ms": dT}, "genome_rules_sha256": genome_rules_hash(g),
                "developed_area_connection": {"w_mech": ac.plastic.w_mech, "dT_ms": ac.static.dT_ms, "H": ac.plastic.H},
                "phenotype_sha256": phenotype_sha256(t),
                "configured": {"weight": r["weight"], "mechanism": r["mechanism"]},
                "realized_A_to_B": block(model, ("A", "L4", "E"), ("B", "L4", "E"))}
# direct NeuronalTensor path with the same declared values
direct = NeuronalTensor(
    areas=[Area(name=n, layers=[Layer(name="L4", neuron_types=[NeuronType.make("E", fraction=1.0)], n_neurons=10)],
                inter_connections=[]) for n in ("A", "B")],
    area_connections=[AreaConnection("A", "L4", "E", "B", "L4", "E", "monotonic_cable_synapse",
                                     static=StaticParams(dT_ms=3.0), plastic=PlasticParams(w_mech=0.5, H=1.0))],
    name="probe-direct")
cfg = neuronal_tensor_to_configuration(direct, seed=0)
r = rule(cfg, "areaconn_0")
B["direct_tensor_w0.5_dT3"] = {"configured": {"weight": r["weight"], "mechanism": r["mechanism"]},
                               "realized_A_to_B": block(j.construct(cfg), ("A", "L4", "E"), ("B", "L4", "E"))}
B["genome_hashes_differ"] = B["genome_w0.5_dT3"]["genome_rules_sha256"] != B["genome_w1.0_dT0.1"]["genome_rules_sha256"]
B["phenotype_hashes_equal"] = B["genome_w0.5_dT3"]["phenotype_sha256"] == B["genome_w1.0_dT0.1"]["phenotype_sha256"]
rec["B_jdna_develop_parameters"] = B

# ---- C: shipped multiarea config ----
t = j.load_canonical_neuronal_tensor("canonical-v1-v4-pfc-multiarea")
cfg = neuronal_tensor_to_configuration(t, seed=0)
rules = cfg.metadata["circuit"]["connections"]
model = j.construct(cfg)
e = edges(model)
area_of = {r["neuron_id"]: r["area"] for r in model.neuron_table()}
same_area = sum(1 for p, q in zip(e["pre"].tolist(), e["post"].tolist()) if area_of[p] == area_of[q])
rec["C_multiarea"] = {
    "areas": [(a.name, a.connectivity_mode, len(a.inter_connections)) for a in t.areas],
    "tensor_connectivity_mode": t.connectivity_mode,
    "configured_rules": [{k: rr[k] for k in ("name", "source", "target", "probability", "weight", "mechanism")} for rr in rules],
    "realized_edges": int(e["pre"].size), "realized_within_area_edges": same_area,
    "config_sha256": hashlib.sha256((Path(j.__file__).parent / "configs" / "canonical-v1-v4-pfc-multiarea.json").read_bytes()).hexdigest(),
}

print(json.dumps(rec, indent=1))

if args.check:
    changed = []
    for path, expected in RECORDED_0_4_24.items():
        got = rec
        for key in path:
            got = got[key]
        ok = math.isclose(got, expected, rel_tol=1e-9) if isinstance(expected, float) else got == expected
        if not ok:
            changed.append((".".join(path), expected, got))
    for name, expected, got in changed:
        print(f"CHANGED {name}: recorded {expected!r}, got {got!r}", file=sys.stderr)
    print(f"PROBE_CHECK {'FAIL' if changed else 'PASS'} ({len(RECORDED_0_4_24)} recorded values, jaxfne {j.__version__})",
          file=sys.stderr)
    sys.exit(1 if changed else 0)
