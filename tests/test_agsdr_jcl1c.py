"""J_CL1C arithmetic (non-simulating). Spec: docs/designs/SCI-AGSDR-J_CL1C_spec.md."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import agsdr_jcl1c_rerun as J  # noqa: E402


def test_shortfall_ge_known_values():
    assert J.shortfall_ge(20.0, 20.0) == pytest.approx(0.0)   # meets threshold
    assert J.shortfall_ge(10.0, 20.0) == pytest.approx(0.5)   # half the requirement
    assert J.shortfall_ge(-5.0, 20.0) == pytest.approx(1.0)   # clipped at 1
    assert J.shortfall_ge(30.0, 20.0) == pytest.approx(0.0)   # over-achieved, no violation


def test_shortfall_le_known_values():
    assert J.shortfall_le(-0.05, -0.05) == pytest.approx(0.0)
    assert J.shortfall_le(-0.025, -0.05) == pytest.approx(0.5)  # half-way to failing
    assert J.shortfall_le(0.2, -0.05) == pytest.approx(1.0)     # clipped at 1


def _deltas(**kw):
    base = dict(delta_r_PV=20.0, delta_rho=-0.05, SST_delta_r_SST=8.0, SST_delta_r_E=-0.5,
                VIP_delta_r_VIP=1.5, VIP_delta_r_SST=-1.0)
    base.update(kw)
    return base


def test_all_thresholds_met_gives_zero():
    v = J.violations(_deltas())
    assert v["J_CL1C"] == pytest.approx(0.0)


def test_nothing_happens_gives_one():
    v = J.violations(_deltas(delta_r_PV=0.0, delta_rho=0.0, SST_delta_r_SST=0.0,
                             SST_delta_r_E=0.0, VIP_delta_r_VIP=0.0, VIP_delta_r_SST=0.0))
    assert v["PV"] == pytest.approx(1.0)  # both PV shortfalls are 1 (no recruitment, no decorrelation)
    assert v["J_CL1C"] == pytest.approx(1.0)
    assert v["J_CL1C"] == pytest.approx((v["PV"] + v["SST"] + v["VIP"]) / 3.0)
    assert 0.0 <= v["J_CL1C"] <= 1.0


def test_equal_weight_mean_across_perturbations():
    v = J.violations(_deltas(delta_r_PV=10.0))  # PV shortfall 0.5 -> PV violation 0.25
    assert v["PV"] == pytest.approx(0.25)
    assert v["J_CL1C"] == pytest.approx(0.25 / 3.0)
