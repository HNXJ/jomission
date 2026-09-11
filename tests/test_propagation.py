"""Propagation battery tests: grammar, cuts, structural-only proof."""

import numpy as np
import pytest

from jomission.network.builder import build_jomission_model
from jomission.qualification import propagation as P


@pytest.fixture(scope="module")
def model():
    return build_jomission_model(n_per_area=100, seed=0)


def test_area_grammar(model):
    areas, tbl = P.area_index(model)
    assert set(areas.keys()) == {"V1", "V4", "FEF", "PFC"}
    assert all(len(v) == 100 for v in areas.values())


def test_cut_counts(model):
    for src, tgt in P.FF_PATH + P.FB_PATH:
        _, n = P.cut_pathway(model, src, tgt)
        assert n > 50, (src, tgt, n)


def test_structural_only_ff(model):
    r = P.pathway_response(model, "V1", "V4", amp=8.0)
    r0 = P.pathway_response(model, "V1", "V4", amp=0.0)
    for a in ("V4", "FEF", "PFC"):
        stim = r["delta"][a] - r0["delta"][a]
        assert abs(stim) < 0.05, (a, stim)  # no stimulus-driven downstream
        assert abs(r["delta"][a] - r["delta_cut"][a]) < 1e-9  # cut-invariant


def test_structural_only_fb(model):
    r = P.pathway_response(model, "PFC", "FEF", drive_layer="L6", amp=20.0)
    r0 = P.pathway_response(model, "PFC", "FEF", drive_layer="L6", amp=0.0)
    stim = r["delta"]["FEF"] - r0["delta"]["FEF"]
    assert abs(stim) < 0.05
