"""Tests for scripts/allen_v1_units.py and the four-unit scorer path. Synthetic fixtures only; each test asserts its
fixture precondition. Real PROTO is used only where stated; the planted comparator is built from the observed map."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import allen_v1_units as au  # noqa: E402
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

# E-layer depths (thickness units, relative to mid-L4) from the seed-1 20 s replica run, x0_H_s1_T20.
ZU = np.array([-0.21566110397946084, 0.0, 0.19768934531450577, 0.4056482670089859])
BANDS = mf.BANDS[mf.FIG_BANDS]
N_WIN = 12


def kernel(zu) -> np.ndarray:
    return np.exp(-((mf.CHANNELS[:, None] - np.asarray(zu)[None, :]) ** 2) / (2 * mf.SIGMA ** 2))


def diag_fit_to(target: np.ndarray) -> np.ndarray:
    """Best diagonal 4-unit spectra per frequency: NNLS on centred probe columns, index = Hz, [151, U] (1e-6 floor)."""
    K = kernel(ZU)
    Ac = K ** 2 - (K ** 2).mean(axis=0, keepdims=True)
    P = np.full((151, len(ZU)), 1e-6)
    for j in range(150):
        y = target[:, j]
        p, _ = nnls(Ac, y - y.mean())
        P[j + 1] += p
    return P


def diag(P: np.ndarray) -> np.ndarray:
    S = np.zeros((P.shape[0], P.shape[1], P.shape[1]))
    idx = np.arange(P.shape[1])
    S[:, idx, idx] = P
    return S


def windows(S_mean: np.ndarray, seed: int = 0) -> np.ndarray:
    """N_WIN windows around S_mean with 5 % lognormal jitter per unit, so the jackknife SE is finite and nonzero."""
    rng = np.random.default_rng(seed)
    U = S_mean.shape[1]
    jit = np.exp(0.05 * rng.standard_normal((N_WIN, 1, U)))
    d = np.diagonal(S_mean, axis1=1, axis2=2)  # [151, U]
    return np.stack([diag(d * jit[w]) for w in range(N_WIN)])


def test_depth_map_takes_four_e_layers_and_rejects_three():
    fig = {"layers": ["L2/3", "L4", "L5", "L6"], "depth_rel_mid_L4": list(ZU)}
    units, depth = au.depth_map(fig)
    assert units == ["L2/3", "L4", "L5", "L6"] and depth["L4"] == 0.0
    with pytest.raises(ValueError):
        au.depth_map({"layers": ["L2/3", "L4", "L5"], "depth_rel_mid_L4": list(ZU[:3])})


def test_planted_four_unit_model_built_from_observed_map_passes_each_leg(monkeypatch):
    K = kernel(ZU)
    S_star = diag(diag_fit_to(sc.PROTO))  # fitted to the observed map before the comparator is swapped
    R_star = sc.rp_map(S_star, K)
    monkeypatch.setattr(sc, "PROTO", R_star)  # comparator = the planted map (as tests/test_e2_scorer.py)
    S_mean = S_star
    W = windows(S_mean)
    R = sc.rp_map(S_mean, K)
    r = sc.r_proto(R)
    rab, rgm = sc.band_r(R, BANDS)
    rank = sc.permutation_rank(S_mean, K)
    p95, _ = sc.smooth_null(K)
    ctrl = sc.identical_unit_control(S_mean, K)
    hv = sc.halves(W, K)
    assert K.shape == (21, 4) and np.all(np.diagonal(S_mean, axis1=1, axis2=2) > 0), "fixture: four units, positive"
    assert r > 0.9, f"fixture: planted map must match its own comparator, r_proto={r:.3f}"
    assert rab >= 0.6 and rgm >= 0.6, f"legs 2: r_ab={rab:.3f} r_gm={rgm:.3f}"
    assert rank == 1, f"leg 3: identity rank {rank}"
    assert r > max(p95, ctrl), f"leg 4: r={r:.3f} p95={p95:.3f} ctrl={ctrl:.3f}"
    assert hv["agree"] is True and np.isfinite(hv["se_diff"]), hv


def test_identical_units_fail_the_control_leg(monkeypatch):
    K = kernel(ZU)
    S_star = diag(diag_fit_to(sc.PROTO))
    monkeypatch.setattr(sc, "PROTO", sc.rp_map(S_star, K))
    S_id = diag(np.repeat(np.diagonal(S_star, axis1=1, axis2=2)[:, :1], len(ZU), axis=1))  # every unit = unit 0
    R = sc.rp_map(S_id, K)
    r = sc.r_proto(R)
    ctrl = sc.identical_unit_control(S_id, K)
    p95, _ = sc.smooth_null(K)
    assert np.allclose(S_id[:, 0, 0][:, None], np.diagonal(S_id, axis1=1, axis2=2)), "fixture: units identical"
    assert np.isclose(r, ctrl, rtol=0, atol=1e-12), f"identical units: r={r} must equal the control {ctrl}"
    assert not (r > max(p95, ctrl)), "negative: identical units must not pass the control leg"


def test_free_diagonal_four_unit_fit_reaches_gamma_leg_against_observed_map():
    """Attainability of leg 2 at these four depths, with the real observed map (no comparator swap). The best
    free diagonal fit reaches r_gm 0.835 (asserted to 0.01). So the failing r_gm of x0_H_s1_T20 (-0.163) is a
    property of the simulated spectra, not a structural ceiling of four diagonal units."""
    K = kernel(ZU)
    S = diag(diag_fit_to(sc.PROTO))
    rab, rgm = sc.band_r(sc.rp_map(S, K), BANDS)
    assert np.isfinite(rab) and np.isfinite(rgm)
    assert abs(rgm - 0.835) < 0.01, f"free diagonal four-unit fit reaches r_gm {rgm:.4f}, measured 0.835"


def test_end_to_end_write_inputs_then_verdict_puts_planted_gaba_tone_in_its_unit_only(tmp_path):
    """The adapter path, not only the spectral helpers: a synthetic run npz with g_nS [T, 8] (per layer AMPA then GABA,
    unit order), a 10 Hz tone planted on the GABA column of L4 only, then write_inputs and sc.verdict. Asserts the tone
    is in unit L4's spectrum and in no other unit's, and that the verdict reads the four adapter units."""
    dt_ms, T_s = 0.1, 3.0
    n = int(round(T_s * 1000.0 / dt_ms))
    t = np.arange(n) * dt_ms / 1000.0
    units = ["L2/3", "L4", "L5", "L6"]
    rng = np.random.default_rng(0)
    g = 5.0 + 0.01 * rng.standard_normal((n, 2 * len(units)))  # nS, positive, background only
    planted = units.index("L4")
    g[:, 2 * planted + 1] = 5.0 + 2.0 * np.sin(2 * np.pi * 10.0 * t)  # GABA column of L4: 10 Hz on a positive offset
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    np.savez(src / "e2e.npz", g_nS=g.astype(np.float32))
    (src / "e2e.json").write_text(json.dumps({"dt_ms": dt_ms, "T_s": T_s, "seed": 1}), encoding="utf-8")
    (src / "e2e_motif.json").write_text(json.dumps({
        "layers": units, "depth_rel_mid_L4": list(ZU), "discard_s": 0.0}), encoding="utf-8")

    summary = au.write_inputs("e2e", src, dst)
    assert summary["units"] == units and (dst / "e2e.npz").is_symlink(), summary

    f, S, _ = sc.unit_spectra("e2e", 0.0, dst)
    assert f[10] == 10.0, "fixture: frequency index must equal Hz"
    p10 = S.mean(axis=0)[10].real.diagonal()
    assert p10[planted] > 0, "fixture: planted unit must carry 10 Hz power"
    others = np.delete(p10, planted)
    assert p10[planted] > 10 * others.max(), f"10 Hz power by unit {dict(zip(units, p10))}: tone must be L4 only"

    out = sc.verdict("e2e", results_dir=dst)
    assert out["units"] == units and out["n_windows"] >= 2, out
    assert out["pass"] is False, "one tone in one layer is not the motif; the verdict must not pass"

    before = (dst / "e2e.npz").resolve()
    au.write_inputs("e2e", src, dst)  # re-run replaces the link; a dangling link must not raise
    assert (dst / "e2e.npz").resolve() == before
