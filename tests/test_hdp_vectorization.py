"""Tier 1: vectorized HDP helpers match their scalar references bit-for-bit.

Each test replicates the pre-vectorization loop inline (the reference) and
compares against the module function (the candidate) across asymmetric,
boundary, inhibitory/excitatory, and empty/small cases. No simulation.
"""

import numpy as np
import pytest

import jomission.qualification.hdp_attractor as ha


def scalar_susceptibility(H, members, h_c, p=2.0, m_min=0.2, m_max=2.0):
    from jomission.qualification.hdp_attractor import I_CLASSES

    H = np.asarray(H, dtype=float)
    m = np.empty_like(H)
    cls_of = {}
    for c, idx in members.items():
        cls_of.update(
            {
                int(i): c
                for c in (["E"] if c == "E" else [c])
                for i in (idx if c in ("E",) + I_CLASSES else [])
            }
        )
    for i in range(len(H)):
        c = cls_of.get(i, "E")
        hc = h_c[c] if isinstance(h_c, dict) else float(h_c)
        f = (H[i] ** p) / (hc**p + H[i] ** p) if H[i] > 0 else 0.0
        m[i] = m_min + (m_max - m_min) * f
    return m


def scalar_tau_eff(theta_hist, theta0, comp=5):
    th = np.asarray(theta_hist, dtype=float)
    d = th[:, comp] - float(np.asarray(theta0, float)[comp])
    taus, mags = [], []
    for k in range(len(d) - 1):
        dd = d[k + 1] - d[k]
        if dd != 0 and d[k] != 0 and np.sign(dd) != np.sign(d[k]):
            taus.append(-d[k] / dd)
            mags.append(abs(d[k]))
    return np.array(taus), np.array(mags)


def test_susceptibility_matches_scalar():
    rng = np.random.default_rng(0)
    cases = [
        (
            np.abs(rng.normal(5, 3, 400)),
            {
                "E": np.arange(300),
                "PV": np.arange(300, 340),
                "SST": np.arange(340, 372),
                "VIP": np.arange(372, 400),
            },
            {"E": 5.0, "PV": 8.0, "SST": 4.0, "VIP": 6.0},
        ),
        (
            np.array([0.0, -1.0, float("nan"), 1e-12, 1e6]),
            {"E": [0, 1], "PV": [2], "SST": [3], "VIP": [4], "XX": [0]},
            {"E": 5.0, "PV": 8.0, "SST": 4.0, "VIP": 6.0, "XX": 1.0},
        ),
        (np.array([3.0, 7.0]), {"E": [0, 1, 99]}, 4.0),
        (np.array([]), {"E": []}, {"E": 5.0}),
        (np.array([2.0]), {}, 4.0),
    ]
    for H, members, hc in cases:
        np.testing.assert_array_equal(
            ha.susceptibility(H, members, hc, 2.0), scalar_susceptibility(H, members, hc)
        )
    with pytest.raises(KeyError):
        ha.susceptibility(np.array([1.0]), {"E": [0]}, {"PV": 1.0})
    with pytest.raises(KeyError):
        scalar_susceptibility(np.array([1.0]), {"E": [0]}, {"PV": 1.0})


def test_tau_eff_series_matches_scalar():
    rng = np.random.default_rng(1)
    cases = [
        np.cumsum(rng.normal(0, 0.05, (60, 6)), axis=0),
        np.zeros((10, 6)),
        np.ones((10, 6)) * 3.0,
        np.array([[1.0], [2.0]]),
        np.zeros((1, 6)),
        np.array([[-1.0, 0.5, -0.5, 0.0, 2.0]]).T,
    ]
    for th in cases:
        for comp in (0, 5):
            if comp >= th.shape[1]:
                continue
            got_t, got_m = ha.tau_eff_series(th, np.zeros(6), comp)
            exp_t, exp_m = scalar_tau_eff(th, np.zeros(6), comp)
            np.testing.assert_array_equal(got_t, exp_t)
            np.testing.assert_array_equal(got_m, exp_m)


def test_caller_median_matches_scalar_columns():
    rng = np.random.default_rng(2)
    series = np.cumsum(rng.normal(0, 0.02, (40, 30)), axis=0)
    series[:, 0] = 1.0
    n = series.shape[1]
    expected = np.full(n, np.nan)
    for i in range(n):
        te, _ = scalar_tau_eff(series[:, i][:, None], np.zeros(1), comp=0)
        if len(te):
            expected[i] = float(np.median(te))
    d = np.asarray(series, dtype=float)
    dd = d[1:] - d[:-1]
    valid = (dd != 0) & (d[:-1] != 0) & (np.sign(dd) != np.sign(d[:-1]))
    ok = valid.any(axis=0)
    te = np.where(valid, -d[:-1] / np.where(valid, dd, 1.0), np.nan)
    got = np.full(n, np.nan)
    got[ok] = np.nanmedian(te[:, ok], axis=0)
    np.testing.assert_array_equal(got, expected)


def test_apply_theta6_matches_scalar():
    from jomission.network.builder import build_jomission_model
    from jomission.qualification.hdp_attractor import theta_to_gains

    model = build_jomission_model(n_per_area=100, seed=0)
    tbl = model.neuron_table()
    is_E = np.array([str(r["cell_type"]) == "E" for r in tbl])
    assert is_E.any() and (~is_E).any(), "mask must see both classes"
    for theta in (np.zeros(6), np.array([0.1, -0.2, 0.05, 0.0, 0.5, -0.3])):
        gm = ha.apply_theta(model, theta[:4])
        e = gm.params["emitter"]
        base = np.asarray(e.drive, dtype=float)
        gdE, gdI = theta_to_gains(theta[4:6])
        expected = np.array(
            [base[i] * (gdE if is_E[i] else gdI) for i in range(len(base))], dtype=e.drive.dtype
        )
        got = np.asarray(ha.apply_theta6(model, theta).params["emitter"].drive)
        np.testing.assert_array_equal(got, expected)
