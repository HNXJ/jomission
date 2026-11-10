"""Tests for scripts/e2_scorer.py on synthetic fixtures only. Each test asserts its fixture precondition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_scorer as es  # noqa: E402

F = es.F
ZU = np.array([-0.5, 0.0, 0.4])  # three units: superficial, mid, deep (rel. mid-L4)
BANDS = es.mf.BANDS[es.mf.FIG_BANDS]


def kernel(zu) -> np.ndarray:
    return np.exp(-((es.mf.CHANNELS[:, None] - np.asarray(zu)[None, :]) ** 2) / (2 * es.mf.SIGMA ** 2))


def diag_spectra(P: np.ndarray) -> np.ndarray:
    """P [f, U] -> S [f, U, U] with the unit powers on the diagonal and zero cross terms."""
    S = np.zeros((P.shape[0], P.shape[1], P.shape[1]))
    idx = np.arange(P.shape[1])
    S[:, idx, idx] = P
    return S


def contrast_slope(R: np.ndarray, bands) -> float:
    """Slope over depth of (alpha-beta minus gamma) band-mean profiles; > 0 means alpha-beta deeper."""
    (a0, a1), (g0, g1) = bands
    ab = R[:, (F >= a0) & (F <= a1)].mean(axis=1)
    gm = R[:, (F >= g0) & (F <= g1)].mean(axis=1)
    return float(es.mf.slope(es.mf.CHANNELS, ab - gm))


def gamma_slope(R: np.ndarray) -> float:
    """Slope over depth of the gamma-band (50-150 Hz) mean profile; < 0 means gamma is superficial."""
    gm = R[:, (F >= 50) & (F <= 150)].mean(axis=1)
    return float(es.mf.slope(es.mf.CHANNELS, gm))


def planted_power() -> np.ndarray:
    """Best diagonal 3-unit shape fit to PROTO, per frequency: p >= 0 maximising Pearson over contacts with PROTO
    (NNLS on centred probe columns), plus a 1e-6 floor so no unit is exactly silent. [f, U], index = Hz.
    A fixed planted pattern (band power only) gave r_proto 0.16, below the null p95 0.27: r_proto averages
    over all 150 Hz, so a planted map must match the observed shape at every frequency."""
    K = kernel(ZU)
    A = K ** 2  # diagonal S: P(c, f) = sum_u K_cu^2 S_uu(f)
    Ac = A - A.mean(axis=0, keepdims=True)
    P = np.full((501, 3), 1e-6)
    for j in range(150):
        y = es.PROTO[:, j]
        p, _ = nnls(Ac, y - y.mean())
        P[j + 1] += p
    return P


def test_planted_alpha_beta_deep_gamma_superficial_ranks_first_and_beats_null():
    K = kernel(ZU)
    P = planted_power()
    ab_u, gm_u = P[10:31].mean(axis=0), P[50:151].mean(axis=0)  # per unit, ZU ascending: superficial, mid, deep
    assert ab_u[2] > ab_u[0] and gm_u[0] > gm_u[2], "fixture must put alpha-beta on the deep unit, gamma on the superficial"
    S = diag_spectra(P)
    R = es.rp_map(S, K)
    assert contrast_slope(R, BANDS) > 0, "fixture must put alpha-beta deep and gamma superficial"
    assert contrast_slope(es.PROTO, BANDS) > 0, "PROTO must have alpha-beta deeper than gamma"
    r = es.r_proto(R)
    p95, _ = es.smooth_null(K, n_draw=1000, seed=0)
    assert r > p95, f"planted r_proto {r:.3f} must exceed the smooth-null p95 {p95:.3f}"
    assert es.permutation_rank(S, K) == 1


def test_identical_units_control_equals_r_proto():
    K = kernel(ZU)
    spec = 0.1 + np.exp(-((np.arange(501) - 20.0) ** 2) / (2 * 4.0 ** 2))  # one non-flat spectrum
    assert spec.std() > 0, "fixture spectrum must vary with frequency"
    S = diag_spectra(np.repeat(spec[:, None], 3, axis=1))
    assert np.array_equal(S[:, 0, 0], S[:, 2, 2]) and np.all(S[:, 0, 1] == 0), "units must be identical and diagonal"
    assert np.allclose(es.identical_unit_control(S, K), es.r_proto(es.rp_map(S, K)), rtol=0, atol=1e-12)


def test_smooth_null_deterministic_at_seed0_and_fresh_draws_below_p95_fraction():
    K = kernel(ZU)
    assert K.shape[1] == 3, "fixture must be three units"
    p95a, draws_a = es.smooth_null(K, n_draw=1000, seed=0)
    p95b, draws_b = es.smooth_null(K, n_draw=1000, seed=0)
    assert len(draws_a) == 1000 and np.array_equal(draws_a, draws_b) and p95a == p95b
    p95_seed0 = p95a
    _, fresh = es.smooth_null(K, n_draw=200, seed=1)
    assert len(fresh) == 200 and fresh.std() > 0, "fresh draws must vary"
    frac = float((fresh > p95_seed0).mean())
    assert 0.0 <= frac <= 0.08, f"fraction of fresh draws above seed-0 p95 is {frac:.3f}"


UNITS = ["S", "M", "D"]
DEPTH = {"S": -0.5, "M": 0.0, "D": 0.4}
N_STEPS_PER_MS = 10  # dt_ms = 0.1
D_STEPS = 60  # RWS delay, 6 ms at dt 0.1 ms


def swap_gamma(P: np.ndarray) -> np.ndarray:
    """Gamma-band (50-150 Hz) power exchanged between the superficial and the deep unit. [f, U]"""
    Q = P.copy()
    gm = (np.arange(P.shape[0]) >= 50) & (np.arange(P.shape[0]) <= 150)
    Q[gm, 0], Q[gm, 2] = P[gm, 2], P[gm, 0]
    return Q


def synth_ms(P: np.ndarray, n_ms: int, rng) -> np.ndarray:
    """Zero-mean Gaussian series sampled at 1 kHz with PSD P[f, U] (frequency index = Hz). Units independent."""
    f = np.fft.rfftfreq(n_ms, d=1e-3)
    grid = np.arange(P.shape[0], dtype=float)
    amp = np.stack([np.sqrt(np.interp(f, grid, P[:, u])) for u in range(P.shape[1])], axis=1)
    W = np.fft.rfft(rng.standard_normal((n_ms, P.shape[1])), axis=0)
    return np.fft.irfft(W * amp, n=n_ms, axis=0)


def synthetic_run(root, segments, crossing: bool = True, tag: str = "synth", seed: int = 0) -> Path:
    """Writes <tag>.json, <tag>.npz (g_nS) and <tag>_motif.json into root. segments: list of (n_ms, P[501, 3]).
    Each unit's ampa conductance is 10 nS plus noise with the segment's PSD and gaba is 0, so the RWS proxy is the
    ampa current: positive, with the planted spectrum. Returns root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = np.concatenate([synth_ms(P, n, rng) for n, P in segments], axis=0)
    g = 10.0 + 0.3 * x / x.std()  # [ms, U], positive
    total = g.shape[0]
    gs = np.zeros((total * N_STEPS_PER_MS + D_STEPS, 2 * len(UNITS)))  # columns: (ampa, gaba) per unit
    gs[:, 0::2] = 10.0
    gs[: total * N_STEPS_PER_MS, 0::2] = np.repeat(g, N_STEPS_PER_MS, axis=0)
    np.savez(root / f"{tag}.npz", g_nS=gs)
    rec = {"tag": tag, "units": UNITS, "depth_rel_L4": DEPTH, "dt_ms": 0.1, "seed": seed}
    (root / f"{tag}.json").write_text(json.dumps(rec), encoding="utf-8")
    motif = {"tag": tag, "discard_s": 0.0, "units": UNITS,
             "sum_over_cells": {"scores": {"image": {"crossing_2se": crossing}}}}
    (root / f"{tag}_motif.json").write_text(json.dumps(motif), encoding="utf-8")
    return root


def planted_target(monkeypatch, P: np.ndarray) -> np.ndarray:
    """Set the observed map to the planted run's own noiseless map, so the planted run is a known positive.
    Reason: no diagonal 3-unit model reaches r_gm 0.6 against the real PROTO (best gamma-band fit 0.31, the cone
    maximum), so the gate tests cannot use PROTO. The real-tag run uses the real PROTO."""
    R = es.rp_map(diag_spectra(P), kernel(ZU))
    monkeypatch.setattr(es, "PROTO", R)
    return R


def test_verdict_passes_on_planted_run_and_fails_on_each_leg(tmp_path, monkeypatch):
    P = planted_power()
    planted_target(monkeypatch, P)
    base = es.verdict("synth", 1, synthetic_run(tmp_path / "base", [(20000, P)]))
    assert base["n_windows"] == 20 and base["n_freq_used"] == 150, "fixture must give 20 windows and 150 frequencies"
    assert base["pass"] is True and base["nonfinite"] == [], base
    legs = {
        "crossing": (dict(segments=[(20000, P)], crossing=False), lambda v: v["crossing_2se"] is False),
        "r_gm_low": (dict(segments=[(20000, swap_gamma(P))]), lambda v: v["r_gm"] < es.R_MIN),
        "rank": (dict(segments=[(20000, P[:, ::-1])]), lambda v: v["perm_rank"] != 1),
        "halves": (dict(segments=[(10000, P), (10000, swap_gamma(P))]), lambda v: v["halves"]["agree"] is False),
    }
    for name, (kw, field_fails) in legs.items():
        v = es.verdict("synth", 1, synthetic_run(tmp_path / name, **kw))
        assert v["pass"] is False, f"leg {name} must fail pass: {v}"
        assert field_fails(v), f"leg {name} must fail its own condition: {v}"


def test_halves_stationary_agrees_and_gamma_flip_disagrees():
    P = planted_power()
    rng = np.random.default_rng(0)
    idx = np.arange(3)

    def windows(Pw, n):
        """S[window, f, u, v] = Pw * Exp(1) per window, frequency and unit; diagonal."""
        S = np.zeros((n, Pw.shape[0], 3, 3))
        S[:, :, idx, idx] = Pw[None] * rng.exponential(1.0, size=(n, Pw.shape[0], 3))
        return S

    K = kernel(ZU)
    stat = windows(P, 20)
    hv = es.halves(stat, K)
    assert np.isfinite(hv["se_diff"]) and hv["se1"] > 0 and hv["se2"] > 0, hv
    assert hv["agree"] is True, hv
    flip = np.concatenate([windows(P, 10), windows(swap_gamma(P), 10)])
    g1 = gamma_slope(es.rp_map(flip[:10].mean(axis=0), K))
    g2 = gamma_slope(es.rp_map(flip[10:].mean(axis=0), K))
    assert g1 < 0 < g2, f"fixture: gamma must flip from superficial to deep ({g1:.3f}, {g2:.3f})"
    assert es.halves(flip, K)["agree"] is False
    short = es.halves(stat[:3], K)  # first half has 1 window
    assert np.isnan(short["se_diff"]) and short["agree"] is False, short


def test_nan_in_identical_control_gives_pass_false(tmp_path, monkeypatch):
    P = planted_power()
    planted_target(monkeypatch, P)
    root = synthetic_run(tmp_path / "nan", [(20000, P)])
    base = es.verdict("synth", 1, root)
    assert base["pass"] is True, "fixture must pass before the NaN is injected"
    monkeypatch.setattr(es, "identical_unit_control", lambda S_mean, K: float("nan"))
    v = es.verdict("synth", 1, root)
    assert v["pass"] is False and v["nonfinite"] == ["identical_control"], v
    assert "identical_control" in v["reason"], v
