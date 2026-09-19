"""Regression for jomission.harness.operation.deviation_time (tier 0).

The estimator used in SCI-V21-PV-PREDECESSOR took its baseline mean and standard deviation over
a window whose leading samples were NaN (reconstruction warm-up), so the band was NaN and no
sample could exceed it: four input channels were structurally undetectable and reported as
"no deviation". The repair is general — the baseline holds only finite eligible samples — not a
special case for the 50 ms warm-up that exposed it.
"""

import numpy as np
import pytest

from jomission.harness.operation import deviation_time

DT = 1.0
BASELINE_MS, K, SUSTAIN_MS = 5000.0, 5.0, 1000.0


def _trace(n=60000, seed=0, warmup=0, step_at=None, amp=0.0):
    rng = np.random.default_rng(seed)
    t = rng.normal(0.0, 1.0, n)
    if step_at is not None:
        t[step_at:] += amp
    if warmup:
        t[:warmup] = np.nan
    return t


def test_leading_nans_do_not_blind_the_detector():
    """The defect: a NaN warm-up inside the baseline window hid every later deviation."""
    clean = _trace(step_at=20000, amp=30.0)
    warm = clean.copy()
    warm[:50] = np.nan
    assert deviation_time(clean, DT, BASELINE_MS, K, SUSTAIN_MS) == 20000.0
    assert deviation_time(warm, DT, BASELINE_MS, K, SUSTAIN_MS) == 20000.0


def test_naive_baseline_would_have_missed_it():
    """Receipt for the defect being real: the old arithmetic yields a NaN band."""
    warm = _trace(step_at=20000, amp=30.0)
    warm[:50] = np.nan
    mu, sd = warm[:5000].mean(), warm[:5000].std()
    assert np.isnan(mu) and np.isnan(sd)
    assert not (np.abs(warm - mu) > K * sd).any()


def test_no_deviation_returns_none():
    assert deviation_time(_trace(), DT, BASELINE_MS, K, SUSTAIN_MS) is None


def test_excursion_shorter_than_sustain_is_not_reported():
    t = _trace()
    t[20000:20500] += 30.0
    assert deviation_time(t, DT, BASELINE_MS, K, SUSTAIN_MS) is None


def test_non_finite_samples_break_a_sustained_run():
    t = _trace(step_at=20000, amp=30.0)
    t[20400] = np.nan
    assert deviation_time(t, DT, BASELINE_MS, K, SUSTAIN_MS) == 20401.0


def test_insufficient_finite_baseline_raises():
    t = _trace()
    t[:4950] = np.nan
    with pytest.raises(ValueError):
        deviation_time(t, DT, BASELINE_MS, K, SUSTAIN_MS)


def test_trace_shorter_than_baseline_returns_none():
    assert deviation_time(_trace(n=1000), DT, BASELINE_MS, K, SUSTAIN_MS) is None
