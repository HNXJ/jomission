"""Can Configuration.connections realize probability < 1 below the tensor bridge?

construct() only; prints JSON to stdout and writes nothing. See docs/tfne/02_JAXFNE_COMPATIBILITY.md finding D.
"""
import json

import numpy as np

import jaxfne as j
from jaxfne.neuronal_tensor import Area, InterConnection, Layer, NeuronalTensor, NeuronType, PlasticParams, neuronal_tensor_to_configuration

layers = [Layer(name=ln, neuron_types=[NeuronType.make("E", fraction=0.8), NeuronType.make("PV", fraction=0.2)], n_neurons=20)
          for ln in ("L2", "L4")]
t = NeuronalTensor(areas=[Area(name="U", layers=layers,
                               inter_connections=[InterConnection("L2", "E", "L4", "E", "AMPA", plastic=PlasticParams(w_mech=1.0))])],
                   area_connections=[], name="probe-density")
cfg = neuronal_tensor_to_configuration(t, seed=0)
mech = cfg.metadata["circuit"]["connections"][0]["mechanism"]
cfg = cfg.connections(name="p025", source={"area": "U", "layer": "L4", "cell_type": "E"},
                      target={"area": "U", "layer": "L2", "cell_type": "E"},
                      probability=0.25, weight=0.1, sign="excitatory", mechanism=mech)
out = {}
for seed in (0, 1):
    m = j.construct(cfg.runtime(seed=seed))
    rows = m.neuron_table()
    ids = lambda layer: {r["neuron_id"] for r in rows if (r["area"], r["layer"], r["cell_type"]) == ("U", layer, "E")}  # noqa: E731
    el = m.params["edge_list"]
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    s4, s2 = np.array(sorted(ids("L4"))), np.array(sorted(ids("L2")))
    out[f"seed{seed}"] = {"L4E_to_L2E_edges_p0.25": int((np.isin(pre, s4) & np.isin(post, s2)).sum()),
                          "L2E_to_L4E_edges_p1.0": int((np.isin(pre, s2) & np.isin(post, s4)).sum()),
                          "block_size": int(s4.size * s2.size)}
print(json.dumps(out, indent=1))
