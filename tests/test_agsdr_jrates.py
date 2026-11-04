"""J_rates arithmetic under rulings R1 to R5 (non-simulating). Spec: docs/designs/SCI-AGSDR-JRATES_spec.md."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import agsdr_jrates_rerun as J  # noqa: E402


def test_target_bands_follow_rulings():
    assert J.target_for("V1", "L4", "E") == pytest.approx((3.0, 10.5))   # per-layer 6 to 7
    assert J.target_for("V1", "L2/3", "E") == pytest.approx((2.0, 9.0))  # per-layer 4 to 6
    assert J.target_for("FEF", "L5", "E") == pytest.approx((1.5, 9.0))   # R2: every FEF layer
    assert J.target_for("PFC", "L1", "E") == pytest.approx((1.5, 9.0))   # R2: every PFC layer
    assert J.target_for("V4", "L2/3", "E") == pytest.approx((2.5, 12.0)) # R1: class global
    assert J.target_for("V1", "L1", "E") == pytest.approx((2.5, 12.0))   # R1: no per-layer entry
    assert J.target_for("V1", "L1", "PV") == pytest.approx((6.0, 37.5))


def test_all_in_band_and_ordered_gives_zero():
    labels = [("V1", "L4", "E"), ("V1", "L4", "PV")]
    out = J.j_rates(labels, [6.5, 20.0])
    assert out["coverage"] == pytest.approx(1.0)
    assert out["ordering_violations"] == 0
    assert out["J_rates"] == pytest.approx(0.0)


def test_all_out_and_violated_gives_one():
    labels = [("V1", "L4", "E"), ("V1", "L4", "PV")]
    out = J.j_rates(labels, [2.0, 1.0])  # E out of [3, 10.5]; PV out of [6, 37.5]; E >= PV
    assert out["coverage"] == pytest.approx(0.0)
    assert out["ordering_violations"] == 1
    assert out["J_rates"] == pytest.approx(1.0)


def test_partial_coverage_no_violation_trade_off():
    labels = [("V1", "L4", "E"), ("V1", "L4", "PV"), ("V1", "L2/3", "E")]
    out = J.j_rates(labels, [6.5, 30.0, 10.0])  # 2 of 3 in band; E < PV at L4
    assert out["coverage"] == pytest.approx(2.0 / 3.0)
    assert out["ordering_penalty"] == pytest.approx(0.0)
    assert out["J_rates"] == pytest.approx(0.5 * (1.0 / 3.0))


def test_epsilon_boundary_is_inclusive_on_band_edges():
    # V1 L4 E band is [3, 10.5]; exact edges count as in band
    out = J.j_rates([("V1", "L4", "E")], [3.0])
    assert out["n_inband"] == 1
    out = J.j_rates([("V1", "L4", "E")], [10.5])
    assert out["n_inband"] == 1
    out = J.j_rates([("V1", "L4", "E")], [10.5000001])
    assert out["n_inband"] == 0
