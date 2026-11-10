"""scripts/e2_unit_power.py: band sums land on the right unit and band, at asymmetric non-neutral values."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_unit_power as up  # noqa: E402


def test_band_power_assigns_units_and_bands():
    f = np.arange(151.0)
    S = np.zeros((2, 151, 2, 2), dtype=complex)
    S[:, 20, 0, 0] = 3.0   # unit A: alpha-beta only
    S[:, 80, 1, 1] = 7.0   # unit B: gamma only
    S[:, 5, 1, 1] = 1.0    # unit B: slow
    S[:, 80, 0, 1] = 1j    # off-diagonal must not leak into the power
    r = up.band_power(f, S, ["A", "B"])
    assert (r["A"]["ab"], r["A"]["gm"], r["A"]["tot"]) == (3.0, 0.0, 3.0)
    assert (r["B"]["ab"], r["B"]["gm"], r["B"]["lo"], r["B"]["tot"]) == (0.0, 7.0, 1.0, 8.0)
    assert r["B"]["gm_frac"] == 7.0 / 8.0 and r["A"]["ab_frac"] == 1.0
    assert r["_imag"] == 0.0
