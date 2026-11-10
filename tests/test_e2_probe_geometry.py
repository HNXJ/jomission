"""Tests for scripts/e2_probe_geometry.py on synthetic fixtures only. Each test asserts its fixture precondition."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_motif_figure as mf  # noqa: E402
import e2_probe_geometry as pg  # noqa: E402
import e2_scorer as es  # noqa: E402

FQ = pg.FREQ  # 501 bins, frequency index = Hz
BANDS = mf.BANDS[mf.FIG_BANDS]
NW = 6  # windows


def spectra(P: np.ndarray, shared: bool = False, seed: int = 0) -> np.ndarray:
    """P [501, U] unit power -> S [window, f, U, U], diagonal, exp(1) noise per window, frequency and unit
    (shared across units when shared=True, so identical P gives identical units window by window)."""
    rng = np.random.default_rng(seed)
    n_f, U = P.shape
    noise = rng.exponential(1.0, size=(NW, n_f, 1 if shared else U))
    S = np.zeros((NW, n_f, U, U), dtype=complex)
    idx = np.arange(U)
    S[:, :, idx, idx] = P[None] * np.broadcast_to(noise, (NW, n_f, U))
    return S


def two_unit_power() -> np.ndarray:
    """Opposite spectra: unit 0 (superficial) gamma 50-150 Hz, unit 1 (deep) alpha-beta 10-30 Hz. [501, 2]"""
    P = np.full((501, 2), 0.1)
    P[(FQ >= 50) & (FQ <= 150), 0] += 1.0
    P[(FQ >= 10) & (FQ <= 30), 1] += 1.0
    return P


def test_theta0_offset0_large_extent_reproduces_figure_kernel_and_contrast():
    zu = np.array([-0.4, 0.4])
    S = spectra(two_unit_power())
    x, z = pg.contacts(0, 0.0)
    K = pg.kernel(zu, x, z, 1e6)
    K_ref = np.exp(-((mf.CHANNELS[:, None] - zu[None, :]) ** 2) / (2 * mf.SIGMA ** 2))
    assert np.allclose(K, K_ref, rtol=0, atol=1e-12), "kernel must equal the figure's depth Gaussian at theta 0"
    M_ref = mf.score(FQ, S, K_ref, mf.CHANNELS, BANDS)["contrast_M"]
    assert abs(M_ref) > 1e-3, "fixture must give a nonzero contrast"
    row = pg.geometry_row(S, zu, 0, 0.0, 1e6, FQ)
    assert abs(row["M_arc"] - M_ref) <= 1e-12, (row["M_arc"], M_ref)
    assert abs(row["M_z"] - M_ref) <= 1e-12, "at theta 0 arc and projected depth coincide"
    assert row["n_bins_covered"] == 21 and row["reason"] is None
    # identity at theta 0: bin mapping is the contact index, so r_proto equals the scorer's rp_map r_proto
    r_scorer = es.r_proto(es.rp_map(S.mean(axis=0), K_ref))
    assert abs(row["r_proto"] - r_scorer) <= 1e-12, (row["r_proto"], r_scorer)


def test_identical_units_give_zero_contrast_and_zero_delta_where_defined():
    zu = np.array([-0.4, 0.0, 0.4])
    P1 = 0.1 + ((FQ >= 10) & (FQ <= 30)).astype(float)
    P = np.repeat(P1[:, None], 3, axis=1)
    assert P1.std() > 0 and np.array_equal(P[:, 0], P[:, 2]), "fixture: identical, frequency-varying units"
    S = spectra(P, shared=True)
    assert np.array_equal(S[:, :, 0, 0], S[:, :, 2, 2]), "fixture: identical units window by window"
    for th in pg.THETAS_DEG:
        for off in pg.OFFSETS:
            row = pg.geometry_row(S, zu, th, off, 0.5, FQ)
            assert abs(row["M_arc"]) < 1e-10, row  # primary: defined at every geometry
            assert math.isnan(row["M_z"]) or abs(row["M_z"]) < 1e-10, row
            if th == 90:  # depth collapse: r_proto undefined, reported NaN, never filled
                assert math.isnan(row["M_z"]) and math.isnan(row["delta_r"]), row
                assert row["reason"] == "projected depths collapse (z range 0)", row
            else:
                assert abs(row["delta_r"]) < 1e-12, row


def test_opposite_two_unit_arc_contrast_falls_toward_90_and_vanishes():
    zu = np.array([-0.4, 0.4])
    S = spectra(two_unit_power())
    M = {th: pg.geometry_row(S, zu, th, 0.0, 0.5, FQ)["M_arc"] for th in (0, 30, 60, 90)}
    assert M[0] > 0, f"fixture: alpha-beta deep and gamma superficial must give M_arc > 0, got {M[0]:.3f}"
    # Not monotone from 0: at 30 deg the contacts beyond the units' depth span (flat profile) are pulled
    # inside it, which steepens the arc slope (observed 1.342 -> 1.410). From 30 deg on it falls.
    assert abs(M[30]) > abs(M[60]) > abs(M[90]) and abs(M[60]) < abs(M[0]), M
    big = pg.geometry_row(S, zu, 90, 0.0, 1e6, FQ)["M_arc"]
    assert abs(big) < 0.05 * abs(M[0]), f"theta 90, offset 0, large extent: |M_arc| {abs(big):.4g} vs theta 0 {M[0]:.4g}"


def test_sparse_projection_gives_nan_r_with_reason():
    P = np.random.default_rng(1).uniform(0.1, 1.0, size=(21, 150))
    z_sparse = np.linspace(-0.06, 0.06, 21)  # covers the bin centres -0.05, 0, 0.05 only
    r, n, reason = pg.r_proto_bins(P, z_sparse)
    assert n == 3, f"fixture must cover 3 bins, got {n}"
    assert math.isnan(r) and reason == "projected depths cover < 5 bins", (r, n, reason)
    r_dense, n_dense, reason_dense = pg.r_proto_bins(P, np.linspace(-0.3, 0.2, 21))
    assert n_dense == 11 and reason_dense is None and np.isfinite(r_dense), (r_dense, n_dense, reason_dense)
