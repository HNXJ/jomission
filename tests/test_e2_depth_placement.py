"""scripts/e2_depth_placement.py on synthetic unit spectra with a known best placement."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_depth_placement as dp  # noqa: E402


def _spectra(gamma_unit: int) -> np.ndarray:
    """Diagonal S[f, u, u]: unit `gamma_unit` carries 50-150 Hz power, the others 10-30 Hz."""
    S = np.zeros((151, 3, 3))
    f = np.arange(151)
    for u in range(3):
        band = (f >= 50) & (f <= 150) if u == gamma_unit else (f >= 10) & (f <= 30)
        S[:, u, u] = 1e-6 + band * (1.0 + 0.1 * u)
    return S


def test_kernel_matches_scorer_at_own_depths():
    z = [-0.4, 0.0, 0.3]
    K = np.exp(-((dp.mf.CHANNELS[:, None] - np.array(z)[None, :]) ** 2) / (2 * dp.mf.SIGMA ** 2))
    np.testing.assert_allclose(dp.kernel(z), K, rtol=1e-12)


def test_gamma_unit_on_top_beats_gamma_unit_at_bottom():
    # Observed map: gamma above alpha-beta. With order kept (zS < zM < zD), gamma spectra in the top unit must fit
    # r_proto better than in the bottom unit. r_gm alone cannot separate them: a one-unit gamma profile can be
    # placed anywhere by the sweep (both reach the same best r_gm), which is why selection uses r_proto.
    top, bottom = (dp.summary(dp.table(_spectra(u)))["best"] for u in (0, 2))
    assert abs(top["r_gm"][0] - bottom["r_gm"][0]) < 1e-3
    assert top["r_proto"][0] > bottom["r_proto"][0] + 0.2, (top["r_proto"], bottom["r_proto"])


def test_select_respects_both_band_thresholds():
    rows = [((-0.4, 0.0, 0.3), 0.9, 0.5, 0.9), ((-0.3, 0.0, 0.3), 0.7, 0.6, 0.6), ((-0.2, 0.0, 0.3), 0.8, 0.7, 0.59)]
    assert dp.select(rows) == (0.7, [-0.3, 0.0, 0.3])
    assert dp.select(rows[:1]) is None
