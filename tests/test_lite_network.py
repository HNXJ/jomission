"""Omission-Model-lite: construction and a short baseline run (deck Chapter 3, stage 6)."""

import collections

import jax.numpy as jnp
import jaxfne as jtfne
import numpy as np
import pytest
from jaxfne import RuntimeConfig, Simulation

from jomission.network.lite import LITE_AREAS, LITE_CELL_COUNTS, build_lite_model


@pytest.fixture(scope="module")
def lite():
    model = build_lite_model(seed=0)
    meta = model.static["neuron_metadata"]
    area_of = {r["neuron_id"]: r["area"] for r in meta}
    el = model.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    assert pre.size > 0, "fixture must build edges"
    return model, meta, area_of, pre, post


def test_realized_counts_match_deck(lite):
    _, meta, _, _, _ = lite
    realized = collections.Counter((r["area"], r["cell_type"]) for r in meta)
    expected = {
        (area, ct): n for area, counts in LITE_CELL_COUNTS.items() for ct, n in counts.items()
    }
    assert realized == expected
    assert len(meta) == sum(sum(c.values()) for c in LITE_CELL_COUNTS.values()) == 50


def test_two_areas_only(lite):
    _, meta, _, _, _ = lite
    assert {r["area"] for r in meta} == set(LITE_AREAS) == {"V1", "FEF_PFC"}


def test_feedforward_and_feedback_edges_present(lite):
    _, _, area_of, pre, post = lite
    pairs = collections.Counter((area_of[int(p)], area_of[int(q)]) for p, q in zip(pre, post))
    assert pairs[("V1", "FEF_PFC")] > 0, "no feedforward V1 -> FEF_PFC edges"
    assert pairs[("FEF_PFC", "V1")] > 0, "no feedback FEF_PFC -> V1 edges"
    assert pairs[("V1", "V1")] > 0 and pairs[("FEF_PFC", "FEF_PFC")] > 0


def test_baseline_run_is_finite_and_shaped(lite):
    model, _, _, _, _ = lite
    sim = Simulation(duration_ms=20.0, dt_ms=0.1, seed=0,
                     runtime=RuntimeConfig(recurrent_backend="edge_list"))
    sig = jtfne.simulate(model, sim)
    assert sig.V_m.shape == (200, 50)
    assert bool(jnp.all(jnp.isfinite(sig.V_m)))
