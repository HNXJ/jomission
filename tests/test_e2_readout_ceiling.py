"""Readout ceiling optimizer: a profile inside the readout set reaches r = 1; one outside it does not."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_readout_ceiling as rc  # noqa: E402

Z = np.array([-0.4, 0.0, 0.3])
C = np.round(np.linspace(-0.6, 0.4, 21), 3)
K = np.exp(-((C[:, None] - Z[None, :]) ** 2) / (2 * 0.15 ** 2))


def test_member_reaches_one():
    y = (K ** 2) @ np.array([1.0, 0.2, 3.0])
    assert rc.best(K, y, False, 16, np.random.default_rng(1)) > 1 - 1e-6


def test_coherent_member_and_outside_profile():
    rng = np.random.default_rng(1)
    y = ((K @ np.array([1.0, -1.0, 0.0])) ** 2)  # anti-phase S and M: a coherent member
    assert rc.best(K, y, True, 16, rng) > 1 - 1e-6
    narrow = np.exp(-((C + 0.2) ** 2) / (2 * 0.03 ** 2))  # bump narrower than the kernel, between units
    coh, inc = rc.best(K, narrow, True, 16, rng), rc.best(K, narrow, False, 16, rng)
    assert inc < 0.9 and coh < 0.95, (coh, inc)
    assert coh >= inc - 1e-6  # the incoherent set is inside the coherent one
