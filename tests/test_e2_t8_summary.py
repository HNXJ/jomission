"""SEAL_T8 decision rule and evaluability (scripts/e2_t8_summary.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import e2_t8_summary as t8  # noqa: E402


@pytest.mark.parametrize("d,want", [
    ([0.3, 0.31, 0.29], "ref higher"),            # all > 0, mean 0.30 >> 2 SD/sqrt3 = 0.012
    ([-0.3, -0.31, -0.29], "control higher"),
    ([0.3, 0.2, -0.01], "no consistent difference"),   # one sign flip
    ([0.01, 0.5, 0.02], "no consistent difference"),   # all > 0, mean 0.177 < 2 SD/sqrt3 = 0.32
])
def test_decide(d, want):
    assert t8.decide(np.array(d)) == want


def test_threshold_uses_sample_sd():
    d = np.array([0.1, 0.2, 0.3])  # mean 0.2; sample SD 0.1 -> bound 0.1155; population SD -> 0.0943
    assert t8.decide(d) == "ref higher"
    assert 2 * d.std(ddof=1) / np.sqrt(3) == pytest.approx(0.11547, rel=1e-4)


def test_evaluable():
    ok = {"missing": False, "refused": False, "stop": None}
    assert t8.evaluable([ok, ok, ok])
    assert not t8.evaluable([ok, ok, {**ok, "stop": "deep E rate < 0.1 Hz"}])
    assert not t8.evaluable([ok, {**ok, "refused": True}, ok])
    assert not t8.evaluable([ok, ok, {"missing": True}])
