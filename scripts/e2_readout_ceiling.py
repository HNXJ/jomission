"""Readout ceilings: the best Pearson r any 3-unit model can reach against the observed LANSA map through the
Gaussian depth kernel, whatever its dynamics (paper plan T1: 0.95 is reported against the ceiling).

Usage:
  python scripts/e2_readout_ceiling.py [--sigma S] [--starts N]

Probe power at contact c and frequency f is diag(K C(f) K^T)[c], K[c,u] the depth kernel (e2_motif_figure),
C(f) the units' cross-spectral matrix. Its real part is PSD, so:
  coherent    any PSD C(f) (C = L L^T, L real 3x3);
  incoherent  diagonal C(f) >= 0 (units uncorrelated): P = K^2 p.
A band-mean depth profile of the max-normalized map lies in the same set (a nonnegative mix of members), so the
band ceiling is max Pearson(member, observed band profile). r_proto's ceiling is the mean over f of the per-f
ceiling. Bands: image (verdict) and manuscript (secondary). Optimizer: L-BFGS from N random starts (seed 0);
a ceiling is a lower bound on the true maximum. Prints one JSON line; writes results/laminar_e2/readout_ceiling.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_motif_similarity as ms  # noqa: E402


def neg_r(P: np.ndarray, y: np.ndarray) -> float:
    a, b = P - P.mean(), y - y.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return 0.0 if den == 0 else -float(a @ b / den)


def best(K: np.ndarray, y: np.ndarray, coherent: bool, starts: int, rng) -> float:
    n = K.shape[1]
    if coherent:
        f = lambda v: neg_r(((K @ v.reshape(n, n)) ** 2).sum(axis=1), y)  # diag(K L L^T K^T)
        dim = n * n
    else:
        K2 = K ** 2
        f = lambda v: neg_r(K2 @ (v ** 2), y)  # p = v^2 >= 0
        dim = n
    return max(-minimize(f, rng.standard_normal(dim), method="L-BFGS-B").fun for _ in range(starts))


def main(sigma: float, starts: int) -> None:
    depth = mf.cs.layout("e2_3unit")[3]
    zu = np.array(list(depth.values()), dtype=float)
    K = np.exp(-((mf.CHANNELS[:, None] - zu[None, :]) ** 2) / (2 * sigma ** 2))
    rng = np.random.default_rng(0)
    out = {"sigma": sigma, "unit_depths": depth, "starts": starts, "bands": {}}
    for name, (ab, gm) in mf.BANDS.items():
        for lab, (lo, hi) in (("ab", ab), ("gm", gm)):
            m = (ms.F >= lo) & (ms.F <= hi)
            y = ms.PROTO[:, m].mean(axis=1)
            out["bands"][f"{name}_{lab}"] = {"range_hz": [lo, hi],
                                            "coherent": best(K, y, True, starts, rng),
                                            "incoherent": best(K, y, False, starts, rng)}
    for kind in ("coherent", "incoherent"):
        out[f"r_proto_{kind}"] = float(np.mean([best(K, ms.PROTO[:, j], kind == "coherent", max(4, starts // 4), rng)
                                                for j in range(ms.PROTO.shape[1])]))
    (mf.D / "readout_ceiling.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=mf.SIGMA)
    ap.add_argument("--starts", type=int, default=32)
    a = ap.parse_args()
    main(a.sigma, a.starts)
