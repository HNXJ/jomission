"""Minimal typed column builder: construction only, no simulation (tier 1).

Covers scoreboard items S-01 to S-06 (docs/PLAN_MINIMAL_MODELS.md section 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxfne.emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS, IzhikevichParams
from jaxfne.hdp_network import DEFAULT_HDP
from jomission.minimal.builder import (
    CELL_TYPES,
    SST_ADAPTATION_A,
    build_minimal_model,
    cell_counts,
    load_connectivity,
)

FRACTIONS = {ct: load_connectivity()["cell_types"][ct]["fraction"] for ct in CELL_TYPES}


def _edge_bytes(model) -> bytes:
    edges = model.params["edge_list"]
    items = edges.items() if isinstance(edges, dict) else vars(edges).items()
    return b"".join(
        k.encode() + np.ascontiguousarray(np.asarray(v)).tobytes() for k, v in sorted(items)
    )


def test_n100_builds_with_seed_and_counts():
    """S-01: one call returns a model of 100 neurons with the declared counts."""
    model = build_minimal_model(n=100, seed=0)
    labels = np.asarray(model.params["emitter"].labels).tolist()
    assert len(labels) == 100
    assert {ct: labels.count(ct) for ct in CELL_TYPES} == cell_counts(100, FRACTIONS)
    assert model.cfg.metadata["n_neurons"] == 100
    assert model.cfg.metadata["seed"] == 0


@pytest.mark.parametrize("n", [50, 100, 200])
def test_sizes_build_without_code_change(n):
    """S-02: the same call builds N in {50, 100, 200}; counts sum to N, none empty."""
    model = build_minimal_model(n=n, seed=0)
    labels = np.asarray(model.params["emitter"].labels).tolist()
    counts = {ct: labels.count(ct) for ct in CELL_TYPES}
    assert len(labels) == n
    assert sum(counts.values()) == n
    assert all(c >= 1 for c in counts.values())


def test_same_seed_gives_identical_wiring_bytes():
    """S-03: two builds with the same seed carry identical edge bytes; the edge set is not empty."""
    first = build_minimal_model(n=100, seed=0)
    second = build_minimal_model(n=100, seed=0)
    assert len(_edge_bytes(first)) > 0
    assert _edge_bytes(first) == _edge_bytes(second)


def test_every_neuron_is_izhikevich_with_jaxfne_presets():
    """S-04: the emitter is Izhikevich; each neuron's b, c, d, drive equal the JaxFNE preset, and a equals the preset except SST (checklist A: 0.02)."""
    model = build_minimal_model(n=100, seed=0)
    emitter = model.params["emitter"]
    assert isinstance(emitter, IzhikevichParams)
    labels = np.asarray(emitter.labels).tolist()
    for ct in CELL_TYPES:
        idx = [i for i, lb in enumerate(labels) if lb == ct]
        assert idx, f"fixture must contain {ct}"
        preset = IZHIKEVICH_CELL_TYPE_DEFAULTS[ct]
        for name in ("a", "b", "c", "d", "drive"):
            expected = SST_ADAPTATION_A if (name == "a" and ct == "SST") else preset[name]
            realized = np.asarray(getattr(emitter, name))[idx]
            assert np.allclose(realized, expected, rtol=1e-6, atol=1e-6), (ct, name)


def test_sst_adaptation_is_realized_not_only_declared():
    """The SST a patch changes the built value from the JaxFNE table (0.05) to 0.02: a realized change, not a label."""
    model = build_minimal_model(n=100, seed=0)
    emitter = model.params["emitter"]
    labels = np.asarray(emitter.labels).tolist()
    sst = [i for i, lb in enumerate(labels) if lb == "SST"]
    assert sst, "fixture must contain SST"
    realized = np.asarray(emitter.a)[sst]
    assert not np.allclose(realized, IZHIKEVICH_CELL_TYPE_DEFAULTS["SST"]["a"]), "patch changed nothing"
    assert np.allclose(realized, SST_ADAPTATION_A, rtol=1e-6, atol=1e-6)


def test_hdp_enabled_with_declared_default_parameters():
    """S-05 (declaration): HDP is on for the whole model and carries the JaxFNE default parameters."""
    model = build_minimal_model(n=100, seed=0)
    meta = model.cfg.metadata
    assert meta["enable_hdp"] is True
    assert meta["hdp_source"] == "jaxfne.hdp_network.DEFAULT_HDP"
    for key, value in DEFAULT_HDP.items():
        assert meta["hdp_params"][key] == value, key


def test_every_declared_rule_compiles_edges():
    """Fixture check: each rule in the connectivity table compiles at least one edge at N = 100."""
    model = build_minimal_model(n=100, seed=0)
    rules = model.cfg.metadata["circuit"]["connections"]
    table = load_connectivity()["rules"]
    assert [r["name"] for r in rules] == [r["name"] for r in table]
    for rule in rules:
        assert int(rule.get("compiled_n_edges", 0)) > 0, rule["name"]


def test_counts_reject_empty_cell_type():
    with pytest.raises(ValueError):
        cell_counts(1, FRACTIONS)
