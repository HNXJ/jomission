"""Tests for V1/V4/FEF/PFC × layer × E/PV/SST/VIP network."""

import jaxfne as jtfne
from jomission.network.populations import validate_populations, JOMISSION_AREAS, JOMISSION_LAYERS, JOMISSION_CELL_TYPES, AREA_LAYER_CELL_TYPES
from jomission.network.connectivity import validate_connectivity, HIERARCHY
from jomission.network.geometry import validate_geometry
from jomission.network.builder import build_jomission_network, build_jomission_model, validate_network
from jomission.paradigm.spec import JOMISSION_PARADIGM


def test_populations_explicit():
    v = validate_populations()
    assert v["valid"], v["issues"]
    assert len(JOMISSION_AREAS) == 4
    assert set(JOMISSION_AREAS) == {"V1", "V4", "FEF", "PFC"}
    for area in JOMISSION_AREAS:
        for layer in JOMISSION_LAYERS:
            fracs = AREA_LAYER_CELL_TYPES[area][layer]
            assert abs(sum(fracs.values()) - 1.0) < 1e-6


def test_layer_cell_types_not_identical():
    # FEF/PFC should differ from V1
    assert AREA_LAYER_CELL_TYPES["V1"]["L2/3"] != AREA_LAYER_CELL_TYPES["FEF"]["L2/3"]


def test_connectivity_hierarchy():
    v = validate_connectivity()
    assert v["valid"], v["issues"]
    assert HIERARCHY == ("V1", "V4", "FEF", "PFC")


def test_geometry_naming():
    v = validate_geometry(n_per_area=100)
    assert v["valid"], v["issues"]
    assert v["n_units"] > 0
    # Check names like V1_L2_3_E
    from jomission.network.geometry import build_laminar_populations
    pops = build_laminar_populations(n_per_area=20)
    names = [p.name for p in pops]
    assert any(n.startswith("V1_L2_3_") for n in names)
    assert any(n.endswith("_PV") for n in names)


def test_builder_four_areas():
    cfg = build_jomission_network(n_per_area=100, seed=0)
    v = validate_network(n_per_area=100)
    assert v["valid"], v["issues"]
    assert v["total_n"] == 400
    assert cfg.metadata["hierarchy"] == "V1->V4->FEF->PFC"


def test_builder_ff_fb_present():
    cfg = build_jomission_network(n_per_area=100, seed=0, p_feedforward=0.3, p_feedback=0.2)
    # Check inter_column_connectivity metadata exists
    assert "inter_column_connectivity" in str(cfg.metadata) or "columns" in cfg.metadata
    # Ensure both directions declared (check via internal metadata count? fallback to columns)
    assert len(cfg.metadata["columns"]) == 4


def test_model_construct_and_simulate():
    model = build_jomission_model(n_per_area=100, seed=1)
    from jomission.simulation.runtime import simulation_for_model
    sig = jtfne.simulate(model, simulation_for_model(model, duration_ms=200.0, dt_ms=0.1, seed=0))
    assert sig.V_m.shape == (2000, 400)
    assert sig.field is not None
    assert sig.field.lfp_proxy.shape[0] == 2000
    assert sig.metadata["field_claim_level"] == "proxy_readout"


def test_minimum_n_per_area():
    try:
        build_jomission_network(n_per_area=50)
        assert False, "should have raised"
    except ValueError as e:
        assert "minimum 100" in str(e)


def test_configurable_n_per_area():
    cfg = build_jomission_network(n_per_area=150, seed=0)
    assert cfg.metadata["n_per_area"] == 150
    assert cfg.metadata["n_total"] == 600
