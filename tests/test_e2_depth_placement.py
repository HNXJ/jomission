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


def test_permutation_moves_cross_terms_with_units():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(151, 3, 3)) + 1j * rng.normal(size=(151, 3, 3))
    S = A @ np.conj(np.transpose(A, (0, 2, 1)))  # Hermitian, off-diagonal cross terms present
    assert np.abs(S[:, 0, 1]).min() > 0, "fixture: cross terms"
    p = [2, 0, 1]
    P = dp.permute(S, p)
    assert np.allclose(P[:, 0, 1], S[:, 2, 0]), "position 0 holds unit 2, position 1 holds unit 0"
    z = [-0.4, 0.0, 0.3]
    # Permuting spectra and permuting depths the same way give the same probe power.
    np.testing.assert_allclose(dp.mf.probe_power(P, dp.kernel(z)),
                               dp.mf.probe_power(S, dp.kernel([z[p.index(i)] for i in range(3)])), rtol=1e-10)


def test_own_depth_mismatch_raises(tmp_path, monkeypatch):
    import json
    import pytest
    (tmp_path / "x_motif.json").write_text(json.dumps({"discard_s": 2.0, "unit_depth_rel_L4": [-0.4, 0.0, 0.3]}))
    (tmp_path / "x_verdict.json").write_text(json.dumps({"r_proto": 0.9, "r_ab": 0.9, "r_gm": 0.9}))
    S = _spectra(0)[None]
    monkeypatch.setattr(dp.sc, "unit_spectra", lambda tag, ds, d: (None, S, None))
    with pytest.raises(ValueError, match="differ from the verdict"):
        dp.spectra("x", tmp_path)


def test_shift_null_keeps_unit_power():
    S = _spectra(0)
    D0 = dp.diag(S)
    assert np.allclose(np.einsum("fuu->fu", D0), np.einsum("fuu->fu", S)) and np.count_nonzero(D0[:, 0, 1]) == 0
    out = dp.shift_null(S, n=1, seed=0)
    assert len(out["draws"]) == 1 and out["diag_real"][0] is not None


def test_select_respects_both_band_thresholds():
    rows = [((-0.4, 0.0, 0.3), 0.9, 0.5, 0.9), ((-0.3, 0.0, 0.3), 0.7, 0.6, 0.6), ((-0.2, 0.0, 0.3), 0.8, 0.7, 0.59)]
    assert dp.select(rows) == (0.7, [-0.3, 0.0, 0.3])
    assert dp.select(rows[:1]) is None
