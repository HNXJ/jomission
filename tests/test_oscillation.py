"""Tier 0: the periodicity estimator, on synthetic signals with a known answer.

No construction and no simulation. Each case plants a period, or plants none, and checks the
estimator recovers it and that its null rejects what it should.
"""

from __future__ import annotations

import numpy as np
import pytest

from jomission.harness import oscillation as O

DT = 1.0
N = 3000


def periodic(period_ms, rate=10.0, depth=1.0, seed=0):
    """A population rate bursting at a known period, on a Poisson background."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT
    wave = 0.5 * (1.0 + np.cos(2.0 * np.pi * t / period_ms))
    return rate * (1.0 - depth + depth * 2.0 * wave) + rng.normal(0.0, 0.2, N)


def flat(rate=10.0, seed=0):
    return np.random.default_rng(seed).normal(rate, 1.0, N)


def test_recovers_a_planted_period():
    for period in (95.0, 190.0, 250.0):
        m = O.measure(periodic(period), dt_ms=DT, n_cells=828, n_surrogate=40)
        assert m["periodic_mode_present"] is True, period
        assert m["period_ms"] == pytest.approx(period, abs=2.0), (period, m["period_ms"])
        assert m["frequency_hz"] == pytest.approx(1000.0 / period, rel=0.03)


def test_an_aperiodic_signal_has_no_mode():
    m = O.measure(flat(), dt_ms=DT, n_cells=828, n_surrogate=40)
    assert m["periodic_mode_present"] is False
    assert m["peak_autocorr"] <= m["null"]["threshold"]


def test_lag_zero_is_never_reported_as_the_period():
    """The autocorrelation is 1.0 at lag 0; a naive argmax would return it every time."""
    for sig in (periodic(190.0), flat(), np.ones(N) * 5.0 + np.arange(N) * 1e-3):
        ac = O.autocorrelation(sig, 500)
        assert ac[0] == pytest.approx(1.0) or ac[0] == 0.0
        lag, _v = O.first_peak_after_zero_crossing(ac)
        assert lag is None or lag > 0


def test_a_monotone_decay_reports_no_peak():
    ac = np.exp(-np.arange(200) / 30.0)          # never crosses zero
    assert O.first_peak_after_zero_crossing(ac) == (None, 0.0)


def test_null_scales_with_the_counting_noise():
    """Fewer cells means a noisier rate, so chance periodicity is easier; the floor rises."""
    few = O.poisson_null(10.0, 10, N, DT, max_lag=500, n_surrogate=60, seed=1)
    many = O.poisson_null(10.0, 1000, N, DT, max_lag=500, n_surrogate=60, seed=1)
    assert few["threshold"] > many["threshold"]


def test_measure_refuses_a_window_shorter_than_the_lag():
    with pytest.raises(ValueError, match="exceeds"):
        O.measure(np.zeros(100), dt_ms=DT, n_cells=10, max_lag_ms=500.0, n_surrogate=5)


def test_measure_refuses_a_degenerate_input():
    with pytest.raises(ValueError, match="at least 8 samples"):
        O.measure(np.zeros(4), dt_ms=DT, n_cells=10, n_surrogate=5)
    with pytest.raises(ValueError, match="positive"):
        O.measure(np.zeros(100), dt_ms=DT, n_cells=0, max_lag_ms=10.0, n_surrogate=5)


def test_a_constant_signal_has_no_correlation_structure():
    ac = O.autocorrelation(np.full(500, 7.0), 100)
    assert np.all(ac == 0.0)


# ------------------------------------------------------------------ the comparison


def MEASURE(**over):
    base = {"periodic_mode_present": True, "period_ms": 190.0, "peak_autocorr": 0.9}
    base.update(over)
    return base


def test_mode_lost_is_attributed_to_the_ablated_connection():
    c = O.compare(MEASURE(), MEASURE(periodic_mode_present=False, period_ms=None))
    assert c["outcome"] == "MODE_REQUIRES_ABLATED_CONNECTION"


def test_mode_surviving_at_the_same_period_exonerates_the_connection():
    c = O.compare(MEASURE(), MEASURE(period_ms=195.0))
    assert c["outcome"] == "MODE_INDEPENDENT_OF_ABLATED_CONNECTION"
    assert c["period_change_fraction"] == pytest.approx(5.0 / 190.0, abs=1e-6)


def test_a_moved_period_is_a_quantitative_dependence_not_an_exoneration():
    c = O.compare(MEASURE(), MEASURE(period_ms=320.0))
    assert c["outcome"] == "MODE_DEPENDS_QUANTITATIVELY"


def test_a_control_without_a_mode_cannot_be_read_as_evidence():
    """If the intact arm shows nothing, the ablation says nothing about recurrence."""
    c = O.compare(MEASURE(periodic_mode_present=False),
                  MEASURE(periodic_mode_present=False))
    assert c["outcome"] == "CONTROL_HAS_NO_MODE"
    assert "not about recurrence" in c["reading"]
