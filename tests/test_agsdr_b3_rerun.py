"""B3 rerun arithmetic (non-simulating). Spec: docs/designs/SCI-AGSDR-B3-RERUN_spec.md."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import agsdr_b3_rerun as B  # noqa: E402


def test_b3_known_answer_tau_free():
    # ratios: pop0 = 1/2 (in band), pop1 = 3/4 (out), pop2 has no current (excluded)
    out = B.b3_from_sums(np.array([1.0, 3.0, 0.0]), np.array([1.0, 1.0, 0.0]))
    assert out["n_populations_with_current"] == 2
    assert out["frac_in_band"] == pytest.approx(0.5)
    assert out["B3_term"] == pytest.approx((0.8 - 0.5) / 0.8)


def test_b3_tau_weighting_changes_membership():
    # One population, E=1, I=2.5. Tau-free ratio 1/3.5 = 0.286 (in band). Tau-weighted
    # (E x 2, I x 5): 2/14.5 = 0.138 (out of band). Membership changes with tau.
    e, i = np.array([1.0]), np.array([2.5])
    free = B.b3_from_sums(e, i)
    tau = B.b3_from_sums(2.0 * e, 5.0 * i)
    assert free["frac_in_band"] == pytest.approx(1.0)
    assert tau["frac_in_band"] == pytest.approx(0.0)
    assert tau["B3_term"] == pytest.approx(1.0)


def test_b3_band_edges_are_inclusive():
    # runner convention: 0.15 <= ratio <= 0.60
    # exact in floating point: ratios 1.5/10, 6/10, 6.01/10
    e = np.array([1.5, 6.0, 6.01])
    i = np.array([8.5, 4.0, 3.99])
    out = B.b3_from_sums(e, i)
    assert out["frac_in_band"] == pytest.approx(2.0 / 3.0)


def test_b3_no_current_is_nan_not_zero():
    out = B.b3_from_sums(np.zeros(3), np.zeros(3))
    assert out["n_populations_with_current"] == 0
    assert np.isnan(out["frac_in_band"])
    assert np.isnan(out["B3_term"])


def test_spec_and_script_are_committed_files():
    assert os.path.exists(B.SPEC)
    assert os.path.exists(os.path.join(REPO, "scripts", "agsdr_b3_rerun.py"))
