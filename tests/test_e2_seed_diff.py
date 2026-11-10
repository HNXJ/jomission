"""SSIM and Pearson helpers of scripts/e2_seed_diff.py on known cases."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_seed_diff as sd  # noqa: E402


def test_ssim_identity_and_degradation():
    rng = np.random.default_rng(0)
    x = rng.random((21, 150))
    assert abs(sd.ssim(x, x) - 1.0) < 1e-12
    noisy = np.clip(x + 0.2 * rng.normal(size=x.shape), 0, 1)
    flipped = x[::-1]
    s_noisy, s_flip = sd.ssim(x, noisy), sd.ssim(x, flipped)
    assert 0 < s_noisy < 1 and s_flip < s_noisy


def test_ssim_symmetric_and_sensitive_to_structure_not_only_mean():
    y = np.tile(np.linspace(0, 1, 21)[:, None], (1, 150))
    assert abs(sd.ssim(y, y[::-1]) - sd.ssim(y[::-1], y)) < 1e-12
    assert sd.ssim(y, y[::-1]) < 0.5  # same mean and variance, inverted structure


def test_pearson():
    y = np.arange(12.0).reshape(3, 4)
    assert abs(sd.pearson(y, 2 * y + 1) - 1) < 1e-12 and abs(sd.pearson(y, -y) + 1) < 1e-12
