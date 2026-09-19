"""Tier 0: the contract renderers, on a synthetic circuit with a known answer.

No construction and no simulation. A tiny hand-made index and edge list exercise the same
code paths the real model uses, so a regression shows up here rather than in a figure nobody
reads carefully.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from jomission.visualization import contract as VC

# Two areas, two layers each, E and PV. Neuron ids are contiguous per group.
INDEX = {
    "A1.L2.E": (0, 4, 4), "A1.L2.PV": (4, 6, 2),
    "A1.L3.E": (6, 10, 4), "A1.L3.PV": (10, 11, 1),
    "A2.L3.E": (11, 15, 4), "A2.L4.E": (15, 19, 4), "A2.L4.PV": (19, 20, 1),
}
N = 20


def model_with(pre, post, weight, receptor):
    edges = types.SimpleNamespace(pre=np.array(pre), post=np.array(post),
                                  weight=np.array(weight, dtype=float),
                                  receptor_index=np.array(receptor))
    return types.SimpleNamespace(params={"edge_list": edges})


def test_describe_counts_populations_without_naming_neurons():
    m = model_with([0], [15], [0.5], [0])
    d = VC.describe(m, INDEX)
    assert d["populations"]["A1"]["L2"] == {"E": 4, "PV": 2}
    assert d["populations"]["A2"]["L4"] == {"E": 4, "PV": 1}
    # The whole point of V0: blocks only.
    assert "neurons" not in d and all("ids" not in p for p in d["projections"])


def test_local_and_long_range_edges_are_separated():
    # A1.L2.E -> A1.L3.E is local; A1.L2.E -> A2.L4.E crosses areas.
    m = model_with([0, 1], [6, 15], [0.1, 0.5], [0, 0])
    d = VC.describe(m, INDEX)
    assert d["n_edges_total"] == 2
    assert d["n_edges_local"] == 1
    assert d["n_edges_long_range"] == 1
    assert len(d["projections"]) == 1


def test_channel_is_read_from_the_layer_pair():
    """ff, fb and lat are layer identities, not guesses from area order."""
    m = model_with([0, 6, 11], [15, 15, 6], [0.5, 0.5, 0.5], [0, 0, 0])
    by = {(p["source"], p["target"]): p for p in VC.describe(m, INDEX)["projections"]}
    assert by[("A1.L2.E", "A2.L4.E")]["channel"] == "ff"
    assert by[("A1.L3.E", "A2.L4.E")]["channel"] == "ff"
    assert by[("A2.L3.E", "A1.L3.E")]["channel"] == "lat"


def test_mean_weight_is_the_mean_over_the_projection():
    m = model_with([0, 1, 2], [15, 15, 15], [0.2, 0.4, 0.6], [0, 0, 0])
    p = VC.describe(m, INDEX)["projections"][0]
    assert p["n_edges"] == 3
    assert p["mean_weight"] == pytest.approx(0.4)


def test_receptor_sets_the_drawn_sign():
    m = model_with([0, 1], [15, 16], [0.5, 0.5], [0, 1])
    signs = {p["sign"] for p in VC.describe(m, INDEX)["projections"]}
    assert signs == {"excitatory", "inhibitory"}


def test_unit_order_is_area_then_layer_then_cell_type():
    rows, groups = VC.unit_order(INDEX, areas=["A2", "A1"])
    assert [(g[0], g[1], g[2]) for g in groups] == [
        ("A2", "L3", "E"), ("A2", "L4", "E"), ("A2", "L4", "PV"),
        ("A1", "L2", "E"), ("A1", "L2", "PV"), ("A1", "L3", "E"), ("A1", "L3", "PV")]
    assert rows.size == N
    assert sorted(rows.tolist()) == list(range(N))


def test_v1_and_v2_order_identically():
    """A before/after comparison is only readable if it is the same figure twice."""
    a, _ = VC.unit_order(INDEX, areas=["A1", "A2"])
    b, _ = VC.unit_order(INDEX, areas=["A1", "A2"])
    assert np.array_equal(a, b)


def test_raster_reports_the_spikes_it_drew(tmp_path):
    spikes = np.zeros((100, N), dtype=np.int8)
    spikes[10, 0] = spikes[20, 5] = spikes[30, 19] = 1
    out = tmp_path / "raster.png"
    info = VC.raster(spikes, INDEX, out, title="t", dt_ms=1.0, window_ms=100.0,
                     areas=["A1", "A2"])
    assert out.is_file() and out.stat().st_size > 0
    assert info["n_spikes"] == 3
    assert info["n_units"] == N
    assert info["thinned"] is False
    # 3 spikes over 20 units in 100 ms = 1.5 Hz mean.
    assert info["mean_rate_hz"] == pytest.approx(1.5)


def test_raster_refuses_a_window_outside_the_recording(tmp_path):
    spikes = np.zeros((50, N), dtype=np.int8)
    with pytest.raises(ValueError, match="outside recorded"):
        VC.raster(spikes, INDEX, tmp_path / "r.png", title="t", dt_ms=1.0, window_ms=1000.0)


def test_raster_announces_thinning_rather_than_dropping_silently(tmp_path):
    spikes = np.ones((100, N), dtype=np.int8)
    info = VC.raster(spikes, INDEX, tmp_path / "r.png", title="t", dt_ms=1.0, window_ms=100.0,
                     max_points=100)
    assert info["thinned"] is True
    assert info["n_spikes"] == 100 * N  # the count reported is what was found, not what was drawn


def test_schematic_writes_a_png_and_reports_its_edge_accounting(tmp_path):
    m = model_with([0, 6, 11, 1], [15, 15, 6, 6], [0.5, 0.5, 0.5, 0.1], [0, 0, 0, 0])
    out = tmp_path / "hspice.png"
    info = VC.hspice_schematic(m, INDEX, out, title="x : A1 O A2 : y", stages=[["A1"], ["A2"]])
    assert out.is_file() and out.stat().st_size > 0
    assert info["n_edges_total"] == 4
    assert info["n_edges_local"] == 1
    assert info["n_edges_long_range"] == 3


def test_stage_paths_follow_the_manifest_convention():
    from jomission.harness import validate as V
    c = V.visualization_contract()
    paths = VC.stage_paths("SOME-LINEAGE", c)
    assert set(paths) == set(c["order"])
    assert paths["V0"] == "results/viz/SOME-LINEAGE/jaxfne_network_hspice.png"
    assert paths["V3"] == "results/viz/SOME-LINEAGE/atlas/index.html"
