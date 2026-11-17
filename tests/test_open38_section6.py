"""OPEN-38 §6 decision-rule pieces against known answers: Holm, the difference / equivalence / INCONCLUSIVE verdicts,
the MDD power solve, the SD interval."""
import importlib.util
import math
from pathlib import Path

import pytest
from scipy import stats

SPEC = importlib.util.spec_from_file_location(
    "s6", Path(__file__).resolve().parents[1] / "scripts" / "open38_section6.py")
s6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s6)


def test_holm_matches_hand_computation():
    # sorted 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone max 0.06
    assert s6.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02], rel=1e-12)
    assert s6.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_difference_needs_holm_significance_and_one_sign():
    d = [1.50, 1.55, 1.52, 1.58, 1.49]
    assert stats.ttest_1samp(d, 0).pvalue < 1e-6, "fixture must be significant"
    assert s6.verdicts(d, 1e-6)["difference"] == "ref higher"
    assert s6.verdicts([-x for x in d], 1e-6)["difference"] == "arm higher"
    assert s6.verdicts(d, 0.06)["difference"] is None, "Holm p above 0.05 gives no difference verdict"
    mixed = [0.30, 0.31, 0.29, 0.32, -0.01]
    assert s6.verdicts(mixed, 1e-6)["difference"] is None, "one opposite-sign seed blocks the verdict"


def test_equivalence_is_the_90_interval_inside_the_margin():
    small = [0.01, -0.01, 0.02, 0.0, -0.02]
    v = s6.verdicts(small, 0.9)
    assert v["equivalent"] and v["verdict"] == "equivalent within ±0.1"
    lo, hi = s6.t_interval(small, 0.90)
    assert -0.1 < lo and hi < 0.1
    wide = [0.2, -0.2, 0.15, -0.1, 0.05]
    v = s6.verdicts(wide, 0.9)
    assert not v["equivalent"] and v["difference"] is None and v["verdict"] == "INCONCLUSIVE"


def test_difference_and_equivalence_can_both_hold_and_the_margin_is_strict():
    tiny = [0.03, 0.031, 0.029, 0.032, 0.030]  # consistent, significant and inside +-0.1
    assert stats.ttest_1samp(tiny, 0).pvalue < 1e-5, "fixture must be significant"
    v = s6.verdicts(tiny, 1e-5)
    assert v["difference"] == "ref higher" and v["equivalent"]
    assert v["verdict"] == "ref higher and equivalent within ±0.1"
    lo, hi = s6.t_interval(tiny, 0.90)
    shift = 0.1 - hi  # move the upper 90 % bound exactly onto the margin: not inside, so not equivalent
    at = [x + shift for x in tiny]
    assert s6.t_interval(at, 0.90)[1] == pytest.approx(0.1, abs=1e-12)
    assert not s6.verdicts([x + shift + 1e-9 for x in tiny], 1e-5)["equivalent"]


def test_mdd_reaches_the_target_power_and_matches_the_normal_limit():
    sigma = 0.0247
    m = s6.mdd(sigma)
    df, tc, nc = 4, stats.t.ppf(1 - 0.05 / 8 / 2, 4), m * math.sqrt(5) / sigma
    lower = stats.nct.sf(tc, df, -nc)
    assert math.isfinite(lower) and lower < 1e-6
    assert stats.nct.sf(tc, df, nc) + lower == pytest.approx(0.80, abs=1e-9)
    # a Monte Carlo check of the same power, independent of nct
    import numpy as np
    rng = np.random.default_rng(1)
    d = rng.normal(m, sigma, size=(200_000, 5))
    t = d.mean(1) / (d.std(1, ddof=1) / math.sqrt(5))
    assert (abs(t) > tc).mean() == pytest.approx(0.80, abs=0.005)
    # large n: delta/sigma -> (z_{1-a/2} + z_0.8) / sqrt(n)
    big = s6.mdd(1.0, n=400, alpha=0.05)
    assert big == pytest.approx((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / 20, rel=0.01)


def test_sd_interval_brackets_the_estimate():
    lo, hi = s6.sd_ci(0.0129, 3)
    assert lo < 0.0129 < hi
    assert hi / lo == pytest.approx(math.sqrt(stats.chi2.ppf(0.975, 2) / stats.chi2.ppf(0.025, 2)), rel=1e-12)
