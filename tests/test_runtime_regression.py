"""Regression: delayed models must not resolve to dense backend.

Mechanically preventable semantics error revealed by 0.4.18 migration:
delay_steps [20,80,120] (builder.py:908) via edge_list_with_delay_ms must not
be simulated with recurrent_backend='dense' (silent loss). JaxFNE now raises
ValueError (_model_simulate.py:162); this test ensures the Jomission authority
fails loudly and the dense path is not accidentally used.
"""
import pytest
import jaxfne as jtfne
from jaxfne import Simulation, RuntimeConfig
from jomission.network.builder import build_jomission_model
from jomission.simulation.runtime import simulation_for_model, model_requires_edge_list, resolve_runtime_for_model


def test_delayed_model_requires_edge_list():
    model = build_jomission_model(n_per_area=100, seed=0)
    assert model_requires_edge_list(model) is True
    el = model.params["edge_list"]
    import numpy as np
    assert set(np.unique(np.asarray(el.delay_steps)).tolist()) == {20, 80, 120}


def test_dense_with_delayed_model_fails_loudly():
    model = build_jomission_model(n_per_area=100, seed=0)
    # Explicit dense request on delayed model must raise via authority
    dense = RuntimeConfig(recurrent_backend="dense")
    with pytest.raises(ValueError, match="requires.*edge_list"):
        resolve_runtime_for_model(model, dense)
    # Also direct JaxFNE guard
    with pytest.raises(ValueError, match="has no finite-delay path"):
        jtfne.simulate(model, Simulation(duration_ms=20.0, dt_ms=0.1, seed=0, runtime=dense))


def test_simulation_for_model_supplies_edge_list():
    model = build_jomission_model(n_per_area=100, seed=0)
    sim = simulation_for_model(model, duration_ms=20.0, dt_ms=0.1, seed=0)
    assert sim.runtime is not None
    assert sim.runtime.recurrent_backend == "edge_list"
    # Simulate succeeds via authority
    sig = jtfne.simulate(model, sim)
    assert sig.V_m.shape == (200, 400)


def test_non_delayed_model_allows_dense():
    # Tiny non-delayed model should allow dense
    from jaxfne import Configuration
    from jaxfne.core import construct
    cfg = Configuration()
    cfg = cfg.column("V1", layers=["L2/3"], n=4)
    cfg = cfg.cell_types({"E": 1.0})
    cfg = cfg.connectivity(within_area="sparse", within_gain=0.5)
    cfg = cfg.field(source_mode="proxy_no_field_solve")
    cfg = cfg.probe(n_contacts=4)
    cfg = cfg.runtime(recurrent_backend="dense")
    cfg = cfg.emitter(family="izhikevich")
    from jaxfne.core import construct as cons2
    m2 = cons2(cfg)
    assert model_requires_edge_list(m2) is False
    # Dense should be allowed
    dense = RuntimeConfig(recurrent_backend="dense")
    resolved = resolve_runtime_for_model(m2, dense)
    assert resolved.recurrent_backend == "dense"
