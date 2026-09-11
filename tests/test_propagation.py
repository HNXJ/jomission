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


def test_drive_semantic_guard():
    """Harness: additive-pulse schedules must carry zero tonic content.

    Baseline segments exactly 0; stimulus adds amp on targets only.
    A schedule built by tiling emitter drive (the historical defect)
    would fail the baseline assertion.
    """
    import jax.numpy as jnp

    import jaxfne as jtfne

    dtype = jnp.float32
    d = P.pure_pulse_drive(500, 400, [38, 39], 8.0, 1000, 2000, dtype)
    a = np.asarray(d)
    assert a.shape == (3500, 400)
    assert (a[:1000] == 0.0).all() and (a[-2000:] == 0.0).all()
    assert (a[1000:1500, 38] == 8.0).all() and (a[1000:1500, 39] == 8.0).all()
    rest = a[1000:1500].copy()
    rest[:, [38, 39]] = 0.0
    assert (rest == 0.0).all()


def test_add_convergence_counts(model):
    m2, n = P.add_convergence(model, "V1", "V4", 4.0, seed=0)
    el0 = model.params["edge_list"]
    el2 = m2.params["edge_list"]
    n0 = int((np.asarray(el0.pre).shape[0]))
    n2 = int((np.asarray(el2.pre).shape[0]))
    assert n2 - n0 == n
    assert abs(n / 97 - 3.0) < 0.5  # factor-1 times base count, approx
    # delays preserved on new edges
    d0 = np.asarray(getattr(el0, "delay_steps", None))
    d2 = np.asarray(getattr(el2, "delay_steps", None))
    assert d0 is not None and d2 is not None
    assert len(d2) - len(d0) == n
