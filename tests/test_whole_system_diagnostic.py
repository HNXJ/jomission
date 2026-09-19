"""Tier 0: WS-DIAG-1 result consistency. No simulation; the driver owns execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jomission.tfne import retina

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "whole_system_diagnostic.json"


def test_retinal_tiling_is_exhaustive_and_non_overlapping():
    for side in ("left", "right"):
        rfs = retina.receptive_fields(side, 14)
        flat = [u for rf in rfs for u in rf]
        assert len(flat) == len(set(flat)) == 512
        assert set(flat) == set(retina.hemifield_units(side))


def test_spot_lies_entirely_in_one_hemifield():
    assert {retina.hemifield(u) for u in retina.spot_units()} == {"left"}
    assert len(retina.spot_units()) == 64


def test_summed_afferent_weight_matches_the_declared_peak():
    for rf in retina.receptive_fields("left", 14):
        assert abs(retina.edge_weight(len(rf), 5.0) * len(rf) - 5.0) < 1e-9


@pytest.mark.skipif(not RESULT.exists(), reason="diagnostic has not been run")
def test_input_interface_was_enforced_not_assumed():
    rec = json.loads(RESULT.read_text(encoding="utf-8"))
    e = rec["enforcement"]
    assert e["intra_retinal_edges_removed"] == 1024 * 1023
    assert e["retinal_tonic_after"] == 0.0
    assert rec["realization_recheck"]["missing"] == [] and rec["realization_recheck"]["extra"] == []
    assert rec["realization_recheck"]["retinal_edges"] == retina.N


@pytest.mark.skipif(not RESULT.exists(), reason="diagnostic has not been run")
def test_verdict_is_the_earliest_failing_gate_in_the_sealed_order():
    rec = json.loads(RESULT.read_text(encoding="utf-8"))
    label = {"execution": "WHOLE_SYSTEM_EXECUTION_FAIL", "collapse": "WHOLE_SYSTEM_COLLAPSE",
             "runaway": "WHOLE_SYSTEM_RUNAWAY", "synchrony": "WHOLE_SYSTEM_SYNCHRONY_FAIL",
             "drift": "WHOLE_SYSTEM_DRIFT", "propagation": "WHOLE_SYSTEM_PROPAGATION_FAIL"}
    first = next((g for g in rec["gate_order"] if not rec["gates"][g]["pass"]), None)
    assert rec["verdict"] == (label[first] if first else "WHOLE_SYSTEM_DIAGNOSTIC_PASS")


@pytest.mark.skipif(not RESULT.exists(), reason="diagnostic has not been run")
def test_baseline_kernel_left_the_plastic_state_static():
    rec = json.loads(RESULT.read_text(encoding="utf-8"))
    assert rec["kernel"] == "baseline" and rec["hdp"] is False
    assert rec["plastic_state"]["w_changed"] is False
    assert rec["plastic_state"]["theta_changed"] is False
