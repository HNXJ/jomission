"""Shared scorer: does a model spectrolaminar map match the observed LANSA map (paper plan W2, section 2).

Usage:
  python scripts/e2_scorer.py TAG [--grid G] [--results-dir D]

Model map: rp_map, the 21 x 150 relative power at 1..150 Hz from the sum-over-cells row (as e2_motif_similarity).
Pass needs all of (PLAN_PAPER_JOMISSION_LANSA.md section 2):
  1 crossing_2se true in <TAG>_motif.json, sum_over_cells, verdict bands (mf.FIG_BANDS)
  2 r_ab >= 0.6 and r_gm >= 0.6 on the verdict bands
  3 the identity unit order ranks 1 of all unit orders (ties count for the identity)
  4 r_proto > max(smooth-null p95, identical-unit control)
  5 halves agree: |r1 - r2| <= 2 * SE_diff, SE_diff = sqrt(SE1^2 + SE2^2), where SE1 and SE2 are the jackknife
    SEs over windows within each half (first half S[:n//2], second half S[n//2:]). A half with fewer than 2
    windows gives SE_diff NaN and agree False.
  6 finite and complete: r_proto, identity r, every unit-order r, identical-unit control and smooth p95 are
    finite, and all 150 frequencies enter r_over_bins (n_freq_used == 150). Otherwise pass is False and
    `nonfinite` and `reason` say why.
Secondary, not in pass: r_ab_manuscript, r_gm_manuscript (8-30 and 31-150 Hz).
Writes <D>/<TAG>_verdict.json and prints it as one JSON line.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_motif_similarity as sim  # noqa: E402

F = np.arange(1, 151)  # 1..150 Hz; a frequency index equals Hz on the 1000-sample (1 Hz) grid
PROTO = sim.PROTO  # [21, 150] area-mean observed map
N_BUMPS = 6
R_MIN = 0.6


def unit_spectra(tag: str, discard_s: float, results_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, S[window, f, u, v], K[contact, u]) for the sum-over-cells row, built as e2_motif_similarity.model_maps."""
    rd = Path(results_dir)
    z = np.load(rd / f"{tag}.npz")
    rec = json.loads((rd / f"{tag}.json").read_text(encoding="utf-8"))
    units = list(rec["units"]) if "units" in rec else list(rec["sizes"])
    depth = rec.get("depth_rel_L4") or mf.cs.layout("e2_3unit")[3]
    dt_ms = rec["dt_ms"]
    x = mf.rws(z["g_nS"].astype(np.float64), dt_ms, rec, units)[int(round(discard_s * 1000.0 / dt_ms)):]
    k = int(round(1.0 / dt_ms))
    n1 = x.shape[0] // k
    x = x[: n1 * k].reshape(n1, k, -1).mean(axis=1)
    if min(1000, n1) != 1000:
        raise ValueError(f"{tag}: {n1} ms after discard; rp_map needs 1000-sample windows (1 Hz bins)")
    zu = np.array([depth[u] for u in units])
    K = np.exp(-((mf.CHANNELS[:, None] - zu[None, :]) ** 2) / (2 * mf.SIGMA ** 2))
    f, S = mf.cross_spectra(x, 1000.0, 1000)
    return f, S, K


def rp_map(S_mean: np.ndarray, K: np.ndarray) -> np.ndarray:
    """S_mean [f, u, v], frequency index = Hz -> [21, 150] relative power P / max_c P at 1..150 Hz."""
    if S_mean.shape[0] < 151:
        raise ValueError("S_mean needs 1 Hz bins up to 150 Hz")
    P = mf.probe_power(S_mean, K)[:, F]
    return P / P.max(axis=0, keepdims=True)


def _r_over_bins(A: np.ndarray, B: np.ndarray) -> tuple[float, int]:
    """Mean over f of Pearson r over bins, as e2_motif_similarity.r_over_bins, and the number of frequencies used
    (a frequency with a constant or non-finite column is dropped)."""
    a = A - A.mean(axis=0)
    b = B - B.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    ok = den > 0
    return float(((a * b).sum(axis=0)[ok] / den[ok]).mean()), int(ok.sum())


def r_proto(R: np.ndarray) -> float:
    return _r_over_bins(R, PROTO)[0]


def band_r(R: np.ndarray, bands) -> tuple[float, float]:
    """(r_ab, r_gm): Pearson r of the band-mean depth profiles against PROTO's, for bands ((a0, a1), (g0, g1))."""
    (a0, a1), (g0, g1) = bands
    ab, gm = (F >= a0) & (F <= a1), (F >= g0) & (F <= g1)
    return (sim.pearson(R[:, ab].mean(axis=1), PROTO[:, ab].mean(axis=1)),
            sim.pearson(R[:, gm].mean(axis=1), PROTO[:, gm].mean(axis=1)))


def smooth_null(K: np.ndarray, n_draw: int = 1000, seed: int = 0, grid_size: int = 1) -> tuple[float, np.ndarray]:
    """Null r_proto draws from smooth diagonal spectra. Each unit's log-power over 1..150 Hz: N_BUMPS Gaussian
    bumps in log f (centres U(log 1, log 150), widths U(0.2, 1.0), amplitudes N(0, 1)) plus -a log f, a ~ U(0, 2).
    Each draw is the max over grid_size such sets. Returns (p95 of the draws, draws)."""
    rng = np.random.default_rng(seed)
    U = K.shape[1]
    lf = np.log(F)
    draws = np.empty(n_draw)
    for i in range(n_draw):
        best = -np.inf
        for _ in range(grid_size):
            S = np.zeros((151, U, U))
            for u in range(U):
                mu = rng.uniform(lf[0], lf[-1], N_BUMPS)
                w = rng.uniform(0.2, 1.0, N_BUMPS)
                amp = rng.standard_normal(N_BUMPS)
                a = rng.uniform(0.0, 2.0)
                bumps = amp[:, None] * np.exp(-((lf[None, :] - mu[:, None]) ** 2) / (2 * w[:, None] ** 2))
                S[1:, u, u] = np.exp(bumps.sum(axis=0) - a * lf)
            best = max(best, r_proto(rp_map(S, K)))
        draws[i] = best
    return float(np.percentile(draws, 95)), draws


def identical_unit_control(S_mean: np.ndarray, K: np.ndarray) -> float:
    """Every unit gets the diagonal spectrum of unit 0; r_proto of that map."""
    U = K.shape[1]
    d = S_mean[:, 0, 0].real
    S = np.zeros((S_mean.shape[0], U, U))
    for u in range(U):
        S[:, u, u] = d
    return r_proto(rp_map(S, K))


def _perm_r(S_mean: np.ndarray, K: np.ndarray) -> dict:
    """r_proto for every unit order: spectra reassigned to depths, K fixed. Keys are order tuples, identity first."""
    U = K.shape[1]
    return {p: r_proto(rp_map(S_mean[:, list(p)][:, :, list(p)], K)) for p in itertools.permutations(range(U))}


def _rank(r: dict, U: int) -> int:
    return 1 + sum(v > r[tuple(range(U))] for v in r.values())


def permutation_rank(S_mean: np.ndarray, K: np.ndarray) -> int:
    """Rank of the identity unit order among all unit orders; 1 = best (ties count for the identity)."""
    return _rank(_perm_r(S_mean, K), K.shape[1])


def _jackknife_se(vals: np.ndarray) -> float:
    m = vals.size
    return float(np.sqrt((m - 1) / m * ((vals - vals.mean()) ** 2).sum()))


def halves(S: np.ndarray, K: np.ndarray) -> dict:
    """Halves rule: first half S[:n//2], second half S[n//2:]. r1, r2 = r_proto of each half's mean spectrum.
    SE1, SE2 = jackknife SE over windows within each half; SE_diff = sqrt(SE1^2 + SE2^2).
    agree = |r1 - r2| <= 2 * SE_diff. A half with fewer than 2 windows gives SE NaN, SE_diff NaN, agree False."""
    n = S.shape[0]
    h = n // 2
    parts = (S[:h], S[h:])
    r = [r_proto(rp_map(p.mean(axis=0), K)) if p.shape[0] else float("nan") for p in parts]
    se = []
    for p in parts:
        m = p.shape[0]
        if m < 2:
            se.append(float("nan"))
            continue
        loo = np.array([r_proto(rp_map(np.delete(p, i, axis=0).mean(axis=0), K)) for i in range(m)])
        se.append(_jackknife_se(loo))
    se_diff = float(np.sqrt(se[0] ** 2 + se[1] ** 2))
    return {"r1": r[0], "r2": r[1], "se1": se[0], "se2": se[1], "se_diff": se_diff,
            "agree": bool(abs(r[0] - r[1]) <= 2 * se_diff)}


def verdict(tag: str, grid_size: int = 1, results_dir=mf.D) -> dict:
    rd = Path(results_dir)
    motif = json.loads((rd / f"{tag}_motif.json").read_text(encoding="utf-8"))
    crossing = bool(motif["sum_over_cells"]["scores"][mf.FIG_BANDS]["crossing_2se"])
    _, S, K = unit_spectra(tag, float(motif["discard_s"]), rd)
    S_mean = S.mean(axis=0)
    R = rp_map(S_mean, K)
    r, n_used = _r_over_bins(R, PROTO)
    rab, rgm = band_r(R, mf.BANDS[mf.FIG_BANDS])
    rab_m, rgm_m = band_r(R, mf.BANDS["manuscript"])
    perm = _perm_r(S_mean, K)
    U = K.shape[1]
    rank = _rank(perm, U)
    r_id = perm[tuple(range(U))]
    p95, _ = smooth_null(K, n_draw=1000, seed=0, grid_size=grid_size)
    ctrl = identical_unit_control(S_mean, K)
    hv = halves(S, K)
    checked = {"r_proto": r, "identity_r": r_id, "identical_control": ctrl, "smooth_p95": p95}
    nonfinite = [name for name, v in checked.items() if not np.isfinite(v)]
    if not all(np.isfinite(v) for v in perm.values()):
        nonfinite.append("perm_r")
    reasons = []
    if nonfinite:
        reasons.append("nonfinite: " + ", ".join(nonfinite))
    if n_used != F.size:
        reasons.append(f"n_freq_used {n_used} != {F.size}")
    ok = bool(not nonfinite and n_used == F.size and crossing and rab >= R_MIN and rgm >= R_MIN and rank == 1
              and r > max(p95, ctrl) and hv["agree"])
    out = {"tag": tag, "units": motif["units"], "discard_s": float(motif["discard_s"]), "n_windows": int(S.shape[0]),
           "grid_size": grid_size, "bands": mf.FIG_BANDS, "crossing_2se": crossing,
           "r_proto": r, "r_ab": rab, "r_gm": rgm, "r_ab_manuscript": rab_m, "r_gm_manuscript": rgm_m,
           "perm_rank": rank, "smooth_p95": p95, "identical_control": ctrl, "identity_r": r_id,
           "n_freq_used": n_used, "nonfinite": nonfinite, "reason": "; ".join(reasons) or None,
           "halves": hv, "pass": ok}
    (rd / f"{tag}_verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Shared scorer for a model spectrolaminar map (W2).")
    ap.add_argument("tag")
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--results-dir", default=str(mf.D))
    a = ap.parse_args(argv)
    print(json.dumps(verdict(a.tag, a.grid, a.results_dir)))


if __name__ == "__main__":
    main()
