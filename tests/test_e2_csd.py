"""Tests for scripts/e2_csd.py: source placement, forward model, CSD estimate and rank on synthetic fixtures."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import e2_csd as ec  # noqa: E402

DEPTH = {"S": -0.4, "M": 0.0, "D": 0.3}  # the depth_rel_L4 of the 3-unit E2 layout
COLS3 = [(u, r) for u in ("S", "M", "D") for r in ("ampa", "gaba", "gaba_sst")]


def test_true_sources_sum_to_zero_over_depth():
    rng = np.random.default_rng(0)
    I = rng.standard_normal((400, len(COLS3)))  # 400 ms of independent currents
    F = ec.forward(I, COLS3, DEPTH)
    scale = np.abs(F["csd_true"]).max()
    assert scale > 0, "fixture must carry current"
    assert (F["csd_true"] > 0).any() and (F["csd_true"] < 0).any(), "fixture must have sources and sinks"
    assert np.abs(F["csd_true"].sum(axis=1)).max() <= 1e-9 * scale


def test_planted_dipole_is_recovered():
    rng = np.random.default_rng(1)
    I = rng.standard_normal((400, 1))
    cols = [("U", "gaba_sst")]  # one column: source at 0.00 (z_u - 0.15), return at 0.15 (z_u)
    F = ec.forward(I, cols, {"U": 0.15})
    occupied = np.nonzero(np.abs(F["csd_true"]).sum(axis=0) > 0)[0]
    assert occupied.size == 2, "fixture must be one source and one return"
    assert np.abs(F["csd_est"]).max() > 0, "fixture must give a non-zero estimate"
    r = ec.corr(F["csd_est"], F["csd_true"])
    assert r >= 0.9, f"dipole correlation {r:.4f} < 0.9"


def test_three_unit_mixing_matrix_is_full_rank():
    # csd_est = I M^T for fixed placements, so with T >= K independent currents its rank is rank(M), K = 9 columns
    rng = np.random.default_rng(2)
    I = rng.standard_normal((2000, len(COLS3)))
    c = np.corrcoef(I, rowvar=False)
    off = c[~np.eye(len(COLS3), dtype=bool)]
    assert np.abs(off).max() < 0.1, "fixture currents must be independent"
    F = ec.forward(I, COLS3, DEPTH)
    assert np.abs(F["csd_est"]).max() > 0, "fixture must give a non-zero estimate"
    rank = ec.numerical_rank(F["csd_est"])
    assert rank == len(COLS3), f"mixing rank {rank} != {len(COLS3)} columns"


def test_csd_estimate_end_contacts_follow_vaknin_padding():
    dz = 0.05
    rng = np.random.default_rng(3)
    phi = rng.standard_normal((50, 7))
    csd = ec.csd_estimate(phi, dz)
    assert np.abs(csd).max() > 0, "fixture must give a non-zero estimate"
    # end value duplicated: first contact -(phi1 - phi0) / dz^2, last contact -(phi[-2] - phi[-1]) / dz^2
    np.testing.assert_allclose(csd[:, 0], -(phi[:, 1] - phi[:, 0]) / dz ** 2, rtol=1e-12)
    np.testing.assert_allclose(csd[:, -1], -(phi[:, -2] - phi[:, -1]) / dz ** 2, rtol=1e-12)
    lin = 2.0 + 0.7 * np.arange(7)[None, :] + np.zeros((50, 1))
    lin_csd = ec.csd_estimate(lin, dz)
    assert np.abs(lin_csd[:, 0]).min() > 0, "fixture must give a non-zero end value for a linear phi"
    np.testing.assert_allclose(lin_csd[:, 1:-1], 0.0, atol=1e-9)


def _write_run(tmp_path, cols, g_shape):
    rec = {"g_columns": [list(c) for c in cols], "g_reversal_mV": [0.0] * len(cols), "dt_ms": 0.1,
           "units": ["U"], "depth_rel_L4": {"U": 0.0}}
    (tmp_path / "run.json").write_text(json.dumps(rec), encoding="utf-8")
    np.savez(tmp_path / "run.npz", g_nS=np.ones(g_shape))
    return tmp_path


def test_refuses_run_without_gaba_sst(tmp_path):
    cols = [("U", "ampa"), ("U", "gaba")]
    assert not any(r == "gaba_sst" for _, r in cols), "fixture must lack gaba_sst"
    _write_run(tmp_path, cols, (100, len(cols)))
    with pytest.raises(SystemExit, match="gaba_sst"):
        ec.load_run(tmp_path, "run")


def test_stops_when_g_nS_shape_differs_from_g_columns(tmp_path):
    cols = [("U", "ampa"), ("U", "gaba"), ("U", "gaba_sst")]
    _write_run(tmp_path, cols, (100, len(cols) + 1))
    assert np.load(tmp_path / "run.npz")["g_nS"].shape[1] != len(cols), "fixture must mismatch"
    with pytest.raises(SystemExit, match="does not match"):
        ec.load_run(tmp_path, "run")
