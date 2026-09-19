"""Tier 0: the causal propagation estimator, on known-answer synthetic rates.

No construction and no simulation. Each case plants a known mixture of true transmission and
the confound WS-AUTH-2 exposed, then checks the estimator separates them.
"""

from __future__ import annotations

import pytest

from jomission.harness import propagation as P

AREAS = ("V1_1", "V1_2", "V4_1", "V4_2", "FEF", "PFC")


def rates(**over):
    """A flat 10 Hz baseline in every area, overridden per area."""
    return {a: over.get(a, 10.0) for a in AREAS}


def test_common_drift_is_removed_exactly():
    """The WS-AUTH-2 failure: every area rises together, including unreachable ones."""
    drift = 0.8
    on = rates(**{a: 10.0 + drift for a in AREAS})
    off = rates()
    # The disconnected control shows the same rise, because it is not transmission.
    control_on = rates(**{a: 10.0 + drift for a in AREAS})
    control_off = rates()
    r = P.causal_response(on, off, control_on, control_off)
    for a in AREAS:
        assert r[a] == pytest.approx(0.0), a
    # The old estimator would have called this a response everywhere.
    old = {a: on[a] - off[a] for a in AREAS}
    assert all(v == pytest.approx(drift) for v in old.values())


def test_true_transmission_is_recovered_on_top_of_drift():
    drift, signal = 0.8, 0.35
    reached = ("V1_1", "V4_1", "FEF")
    on = rates(**{a: 10.0 + drift + (signal if a in reached else 0.0) for a in AREAS})
    off = rates()
    control_on = rates(**{a: 10.0 + drift for a in AREAS})
    control_off = rates()
    r = P.causal_response(on, off, control_on, control_off)
    for a in AREAS:
        assert r[a] == pytest.approx(signal if a in reached else 0.0), a


def test_evaluate_uses_the_tolerance_and_reports_levels():
    signal = 0.5
    on = rates(**{a: 10.0 + signal for a in ("V1_1", "V4_1", "FEF")})
    off, control_on, control_off = rates(), rates(), rates()
    # Known-answer on synthetic rates: no simulation and no result verdict is read.
    good = P.evaluate(on, off, control_on, control_off, tolerance_hz=0.1)
    detected = good["pass"]
    assert detected is True
    assert good["levels"] == {"V1_driven": True, "V4": True, "frontal": True}
    # A tolerance above the signal must reject it rather than round it up.
    strict = P.evaluate(on, off, control_on, control_off, tolerance_hz=1.0)
    rejected = strict["pass"]
    assert rejected is False
    assert not any(strict["responded"].values())


def test_evaluate_refuses_a_non_measured_tolerance():
    r = rates()
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="positive"):
            P.evaluate(r, r, r, r, tolerance_hz=bad)


def test_mismatched_area_sets_raise_rather_than_silently_intersect():
    full = rates()
    partial = {a: 10.0 for a in AREAS[:-1]}
    with pytest.raises(ValueError, match="disagree on areas"):
        P.causal_response(full, partial, full, full)


def test_self_control_is_degenerate_and_noise_floor_says_so():
    """Evaluating R on the control pair itself is X - X: plumbing, not evidence."""
    on = rates(**{a: 10.0 + 0.8 for a in AREAS})
    off = rates()
    r = P.causal_response(on, off, on, off)
    assert all(v == 0.0 for v in r.values())
    floor = P.noise_floor(on, off, on, off)
    assert floor["is_degenerate"] is True, "a self-control floor must announce that it measured nothing"
    assert floor["max_abs"] == 0.0


def test_noise_floor_on_a_held_out_replicate_is_not_degenerate():
    """A real null replicate differs from the control by chance, and that spread is the floor."""
    control_on = rates(**{a: 10.0 + 0.80 for a in AREAS})
    control_off = rates()
    # Same null condition, different seed: the common drift lands slightly differently.
    rep_on = rates(V1_1=10.83, V1_2=10.74, V4_1=10.86, V4_2=10.77, FEF=10.91, PFC=10.72)
    rep_off = rates()
    floor = P.noise_floor(rep_on, rep_off, control_on, control_off,
                          areas=P.DISCONNECTED_AT_ZERO_G)
    assert floor["is_degenerate"] is False
    assert floor["n_areas"] == 4
    assert floor["max_abs"] == pytest.approx(0.11, abs=1e-9)
    assert 0.0 < floor["rms"] <= floor["max_abs"]
    # A tolerance drawn from this floor rejects the replicate itself, which is the point.
    verdict = P.evaluate(rep_on, rep_off, control_on, control_off,
                         tolerance_hz=floor["max_abs"] * 1.5)
    assert not any(verdict["responded"][a] for a in P.DISCONNECTED_AT_ZERO_G)


def test_noise_floor_rejects_a_replicate_missing_areas():
    r = rates()
    with pytest.raises(ValueError, match="disagree on areas|missing areas"):
        P.noise_floor({a: 10.0 for a in AREAS[:3]}, r, r, r, areas=AREAS)
