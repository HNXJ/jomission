"""Known-answer tests for scripts/e2_r2_readout.py (kernel, readout, per-area correlations, sensitivity flags).

Synthetic inputs only; no simulation output is needed. Run from the repository root:
    python -m unittest tests/test_e2_r2_readout.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import e2_r2_readout as rr  # noqa: E402
import e2_motif_figure as mf  # noqa: E402


class KernelAndReadout(unittest.TestCase):
    def test_kernel_known_value(self):
        K = rr.kernel(np.array([0.0, 0.0, 0.0]), 0.15)
        self.assertEqual(K.shape, (21, 3))
        i0 = int(np.argmin(np.abs(mf.CHANNELS - 0.0)))
        self.assertAlmostEqual(K[i0, 0], 1.0)
        # contact at -0.6 relative to a unit at 0: exp(-0.36 / (2 * 0.15^2)) = exp(-8)
        self.assertAlmostEqual(K[0, 1], np.exp(-8.0), places=12)

    def test_flat_unit_spectra_readout(self):
        # identity cross-spectra (unit power, no cross terms) at every frequency: P[c] = sum_u K[c,u]^2, flat in f
        K = rr.kernel(rr.D_ANAT, rr.SIG)
        S = np.zeros((151, 3, 3))
        for u in range(3):
            S[:, u, u] = 1.0
        R = rr.readout(S, K)
        expect = (K ** 2).sum(axis=1)
        expect = expect / expect.max()
        self.assertEqual(R.shape, (21, 150))
        np.testing.assert_allclose(R, np.repeat(expect[:, None], 150, axis=1), rtol=1e-12, atol=0)

    def test_readout_negative_control(self):
        # a single-unit spectrum at the shallowest unit gives a different profile than the flat one
        K = rr.kernel(rr.D_ANAT, rr.SIG)
        S = np.zeros((151, 3, 3))
        S[:, 0, 0] = 1.0
        R = rr.readout(S, K)
        self.assertGreater(np.abs(R[:, 0] - R[:, 0].mean()).max(), 0.05)


class PerAreaCorrelation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.R = rng.random((21, 150))
        self.areas = [f"a{i}" for i in range(11)]

    def test_identical_target_gives_one(self):
        Y = np.repeat(self.R[:, :, None], 11, axis=2)
        pa = rr.per_area(self.R, Y, self.areas)
        for name in self.areas:
            self.assertAlmostEqual(pa[name]["r_gm"], 1.0, places=10)
            self.assertAlmostEqual(pa[name]["r_ab"], 1.0, places=10)

    def test_negated_target_gives_minus_one(self):
        Y = np.repeat(-self.R[:, :, None], 11, axis=2)
        pa = rr.per_area(self.R, Y, self.areas)
        self.assertAlmostEqual(pa["a0"]["r_gm"], -1.0, places=10)
        self.assertAlmostEqual(pa["a0"]["r_ab"], -1.0, places=10)

    def test_gamma_band_only_matches_gamma_columns(self):
        # target equal to R in the 50-150 Hz columns and random elsewhere: r_gm = 1, r_ab < 1
        rng = np.random.default_rng(2)
        Y1 = rng.random((21, 150))
        gm_cols = (np.arange(1, 151) >= 50) & (np.arange(1, 151) <= 150)
        Y1[:, gm_cols] = self.R[:, gm_cols]
        Y = np.repeat(Y1[:, :, None], 11, axis=2)
        pa = rr.per_area(self.R, Y, self.areas)
        self.assertAlmostEqual(pa["a0"]["r_gm"], 1.0, places=10)
        self.assertLess(pa["a0"]["r_ab"], 0.99)

    def test_summary_medians_and_flags(self):
        pa = {f"a{i}": {"r_ab": 0.9, "r_gm": 0.55 if i < 6 else 0.1} for i in range(11)}
        s = rr.summary(pa)
        self.assertAlmostEqual(s["median_r_gm"], 0.55)
        self.assertEqual(s["n_r_gm_ge_0.6"], 0)
        self.assertEqual(s["median_r_gm_ge"], {"0.5": True, "0.6": False, "0.7": False})
        self.assertEqual(s["median_r_ab_ge"], {"0.5": True, "0.6": True, "0.7": True})


if __name__ == "__main__":
    unittest.main()
