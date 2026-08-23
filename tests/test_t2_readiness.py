"""T2 readiness gates — before authorizing ≥1000s exposure."""

import jax.numpy as jnp
from jomission.simulation.exposure import run_shortened_exposure
from jomission.simulation.stability import check_stability, STABILITY_CRITERIA
from jomission.simulation.ledger import PRODUCTION_SEED_LEDGER, CHECKPOINT_CADENCE, PRODUCTION_SCHEDULE
from jomission.analysis.comparison_matrix import COMPARISON_MATRIX


def test_shortened_exposure_stability():
    # 30s is fast; 120s would be heavier but exercises 100s tau partially
    res = run_shortened_exposure(duration_s=30, dt_ms=0.5, seed=1, enable_hdp=False)
    assert res["finite"] is True
    assert res["field_present"] is True
    chk = check_stability(res)
    assert chk["pass"], chk["issues"]
    # Area rates finite
    for ar in res["area_rates_hz"].values():
        assert 0.5 <= ar <= 80


def test_stability_criteria_frozen():
    assert "finite_state" in STABILITY_CRITERIA
    assert "exclusion_rules" in STABILITY_CRITERIA
    assert STABILITY_CRITERIA["finite_state"]["must_be_finite"] is True


def test_seed_ledger_frozen():
    assert len(PRODUCTION_SEED_LEDGER["seeds"]) >= 8
    assert PRODUCTION_SEED_LEDGER["pairing_scheme"] == "paired_by_replicate"
    assert PRODUCTION_SEED_LEDGER["replicates"] == 8


def test_checkpoint_cadence():
    assert CHECKPOINT_CADENCE["every_n_trials"] == 10
    assert "verification" in CHECKPOINT_CADENCE
    assert "ALWAYS-31" in CHECKPOINT_CADENCE["verification"]


def test_production_schedule():
    assert PRODUCTION_SCHEDULE["exposure"]["duration_s"] >= 1000
    assert PRODUCTION_SCHEDULE["testing"]["total_trials"] == 96


def test_comparison_matrix_frozen():
    assert COMPARISON_MATRIX["matrix_version"] == "jomission_comparison_matrix.v0.1.0"
    assert len(COMPARISON_MATRIX["targets"]) == 7
    ids = [t["id"] for t in COMPARISON_MATRIX["targets"]]
    assert ids == ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    assert COMPARISON_MATRIX["pooling_rule"].startswith("DO NOT pool")
    assert COMPARISON_MATRIX["language_rule"].startswith("lfp_proxy")
