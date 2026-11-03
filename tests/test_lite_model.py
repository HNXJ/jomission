"""Lite model: canonical builder restricted to V1 and V4 (200 neurons). Builds and simulates finitely."""

import numpy as np

import jaxfne as jtfne

from jomission.network.builder import build_jomission_lite_model


def test_lite_model_builds_with_200_neurons_and_simulates_finitely():
    model = build_jomission_lite_model(seed=0)
    el = model.params["edge_list"]
    assert int(el.pre.shape[0]) > 0
    assert int(np.max(np.asarray(el.pre))) < 200 and int(np.max(np.asarray(el.post))) < 200
    sim = jtfne.simulate(
        model,
        jtfne.Simulation(duration_ms=50.0, dt_ms=0.1, seed=0,
                         runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
    )
    vm = np.asarray(sim.V_m)
    assert vm.size > 0 and np.all(np.isfinite(vm))
    assert vm.shape[-1] == 200 or vm.shape[0] == 200 or 200 in vm.shape, vm.shape
