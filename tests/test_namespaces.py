"""Tests for pilot/canonical namespace separation and numerical-validity annotation."""

import pytest

from jomission.analysis.namespaces import (
    seal_pilot,
    seal_canonical,
    promote_pilot_to_canonical,
    annotate_numerical_validity,
    NamespaceError,
    P1_TO_D4_MS,
    SCHEDULER_TRIAL_MS,
)


def test_clock_semantics_explicit():
    assert P1_TO_D4_MS == 4124.0
    assert SCHEDULER_TRIAL_MS == 4624.0
    p = seal_pilot({"x": 1})
    assert p["clock_note"]["p1_to_d4_ms"] == 4124.0
    assert p["clock_note"]["scheduler_trial_ms"] == 4624.0


def test_pilot_cannot_become_canonical():
    p = seal_pilot({"T1": "NEGATIVE"})
    assert p["namespace"] == "pilot_developmental"
    assert p["promotable_to_canonical"] is False
    with pytest.raises(NamespaceError):
        seal_canonical(p, computational_valid=True)
    with pytest.raises(NamespaceError):
        promote_pilot_to_canonical()


def test_canonical_requires_computational_validity():
    c = seal_canonical({"delta_exposure": None}, computational_valid=False)
    assert c["namespace"] == "canonical_confirmatory"
    assert c["delta_exposure"] == "UNRESOLVED"
    assert c["T1"] == "UNRESOLVED"
    c2 = seal_canonical({"T1": "NEGATIVE"}, computational_valid=True)
    assert c2["T1"] == "NEGATIVE"


def test_numerical_violations_recorded_not_hidden():
    # Θ violation and rate violation must be logged; stable_claim False
    r = annotate_numerical_validity({}, theta_min=-0.129, theta_lo_bound=-0.1,
                                    rate_hz_mean=73.0, rate_range=(1.0, 50.0))
    nv = r["numerical_validity"]
    assert not nv["stable_claim"]
    assert any("theta_boundary_violation" in i for i in nv["threshold_violations"])
    assert any("rate_outside_frozen_criterion" in i for i in nv["threshold_violations"])
    # Clean values remain stable
    r2 = annotate_numerical_validity({}, theta_min=1.02, theta_lo_bound=-0.1,
                                     rate_hz_mean=21.9, rate_range=(1.0, 50.0))
    assert r2["numerical_validity"]["stable_claim"]
