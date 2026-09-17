"""Tier 0 checks for SCI-V21-PV-MICROSTATE (read-only reanalysis).

The script itself writes the result; this test pins its constants to the sealed spec, checks the
two distance functions against cases with known answers, and — when the result exists — checks
that the recorded verdict follows from the recorded numbers.
"""

import json
from pathlib import Path

import numpy as np

from scripts import v21_pv_microstate as m

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "results" / "v21_pv_microstate_spec.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "v21_pv_microstate.json"
LABELS = ("PV_MICROSTATE_PREDECESSOR", "RECORDED_MICROSTATE_INSUFFICIENT", "UNRESOLVED", "TRACE_IDENTITY_FAILED")


def test_constants_match_spec():
    c = SPEC["procedure_constants"]
    assert (m.EPOCH_MS, m.LEAD_EPOCHS, m.PV_RATE_DIVERGENCE_MS, m.PRECEDE_MS, m.SEP_FACTOR,
            m.SUSTAIN_EPOCHS, m.MIN_CELLS_ISI, m.MIN_SPIKES_ISI, m.WARMUP_MS) == (
        c["epoch_ms"], c["lead_window_epochs"], c["pv_rate_divergence_ms"], c["precede_ms"],
        c["separation_factor"], c["sustain_epochs"], c["min_cells_for_isi"],
        c["min_spikes_per_cell_for_isi"], c["warmup_ms"])
    assert set(SPEC["classification"]) == set(LABELS)


def test_wasserstein1_known_cases():
    x = np.arange(40.0)
    assert m.wasserstein1(x, x) == 0.0
    assert m.wasserstein1(x, x + 3.0) == 3.0
    assert m.wasserstein1(x, x[::-1]) == 0.0  # same distribution, permuted


def test_energy_distance_known_cases():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 2))
    assert abs(m.energy_distance(a, a)) < 1e-9
    assert m.energy_distance(a, a + 10.0) > m.energy_distance(a, a + 1.0) > 0.0


def test_sustained_start_requires_consecutive_epochs():
    assert m.sustained_start([1, 0, 1, 0, 1]) is None
    assert m.sustained_start([0, 0, 1, 1, 1]) == 2


def test_evaluate_level_threshold_and_lead():
    zero = [0.0] * 15
    big = [10.0] * 15
    small = [0.1] * 15
    res = m.evaluate_level(big, big, zero, 15)
    assert res["qualifying_epoch"] == 0 and res["leads_macroscopic"]
    assert m.evaluate_level(small, small, small, 15)["qualifying_epoch"] is None


def test_epoch_zero_difference_is_not_treated_as_evidence_by_itself():
    """The lead rule, not the epoch-0 difference, is what a level must satisfy."""
    assert "constitutive" in SPEC["interpretation_limit"]
    late = [0.0] * 14 + [10.0]
    assert m.evaluate_level(late, late, [0.0] * 15, 15)["qualifying_epoch"] is None


def test_result_verdict_follows_from_recorded_numbers():
    if not OUT.exists():
        return
    rec = json.loads(OUT.read_text(encoding="utf-8"))
    assert rec["verdict"] in LABELS
    if rec["verdict"] == "TRACE_IDENTITY_FAILED":
        return
    ident = rec["trace_identity"]
    assert ident["within_print_precision"] and ident["sha256_match"]
    lead = [lv for lv in rec["levels"].values() if lv.get("evaluated") and lv["leads_macroscopic"]]
    if rec["verdict"] == "RECORDED_MICROSTATE_INSUFFICIENT":
        assert not lead
    else:
        assert lead
        first = min(lead, key=lambda lv: lv["level"])
        pop = first["population_mean_test"]
        specific = pop["qualifying_epoch"] is None or pop["qualifying_epoch"] > first["qualifying_epoch"]
        assert (rec["verdict"] == "PV_MICROSTATE_PREDECESSOR") == specific
