"""Joint readout capacity and circuit-vs-optimum comparison (reply to review 3, 2026-11-10).

Usage: python scripts/e2_readout_joint.py OUT_JSON

Geometry, kernel, admissible set (C = L L^T, symmetric PSD) and target y (observed 50-150 Hz band-mean profile,
/ max) as in e2_readout_capacity.py; one covariance for the whole band; sigma = 0.15 (declared) and 0.20.

joint: max over C of r(P, y) subject to |mid(P) - m| <= EPS, for m = mid(y) (0.946, the band-mean profile's own
  level, the matched convention) and m = 0.848 (mean of the per-frequency map), full and auto C. Penalty method:
  objective -r + PEN * max(0, |mid - m| - EPS), STARTS Nelder-Mead starts; feasible = the best solution meets the
  constraint. Reports r, mid, power shares, coherences and eigenvalues / trace of C.
circuit: for the reference column (kd1, SOM->E 15 ms, N 4000, seeds 1-3) and the T1 winner on confirmation seeds
  4-6, from the scorer's window-mean unit cross-spectra (2 s discard): 50-150 Hz power shares of S, M, D (diagonal
  summed over the band), band-mean coherence |S_uv| / sqrt(S_uu S_vv) and its real part (only Re S_uv enters the
  contact power), r_y (band-mean contact power vs y, the capacity convention), mid of that profile, and the
  scorer's r_gm, r_proto and mid-depth relative gamma of the per-frequency map.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_readout_capacity as rc  # noqa: E402
import e2_scorer as sc  # noqa: E402

EPS = 0.02
PEN = 100.0
SIGMAS = (0.15, 0.20)
TAGS = {"reference": ["sw_n4000_dens_k1_kd1_sst15_s1_T8", "lfs_n4000_s2_T8", "lfs_n4000_s3_T8"],
        "t1_winner_fresh": [f"t1_confirm_kd1p41_sst15_sstd21_n4000_s{s}_T12" for s in (4, 5, 6)]}
PAIRS = {"SM": (0, 1), "SD": (0, 2), "MD": (1, 2)}


def joint(K: np.ndarray, y: np.ndarray, m: float, full: bool, rng: np.random.Generator) -> dict:
    def prof(v):
        return rc.power(K, rc.tril(v)) if full else (K ** 2) @ (v ** 2)

    def obj(v):
        P = prof(v)
        if P.max() <= 0:
            return 1e6
        return -rc.corr(P, y) + PEN * max(0.0, abs(rc.mid(P) - m) - EPS)

    v = rc.best_of(obj, 6 if full else 3, rng)
    P = prof(v)
    C = rc.tril(v) @ rc.tril(v).T if full else np.diag(v ** 2)
    out = {"r": rc.corr(P, y), "mid": rc.mid(P), "feasible": bool(abs(rc.mid(P) - m) <= EPS + 1e-9)}
    if full:
        out.update(rc.describe(rc.tril(v)))
    else:
        out.update({"share": (np.diag(C) / np.trace(C)).round(4).tolist(),
                    "eig_over_trace": (np.linalg.eigvalsh(C) / np.trace(C)).tolist()})
    return out


def circuit(tag: str, y: np.ndarray) -> dict:
    f, S, _ = sc.unit_spectra(tag, 2.0, mf.D)
    Sm = S.mean(axis=0)  # [f, U, U]
    g = (f >= 50) & (f <= 150)
    Sg = Sm[g]
    d = np.einsum("fuu->fu", Sg).real
    out = {"share": (d.sum(axis=0) / d.sum()).round(4).tolist()}
    for k, (u, v) in PAIRS.items():
        den = np.sqrt(d[:, u] * d[:, v])
        out[f"coh_{k}"] = float((np.abs(Sg[:, u, v]) / den).mean())
        out[f"recoh_{k}"] = float((Sg[:, u, v].real / den).mean())
    K = rc.kern(rc.Z3, mf.SIGMA)
    P = mf.probe_power(Sm, K)  # [C, f]
    pb = P[:, g].mean(axis=1)
    out["r_y"] = rc.corr(pb, y)
    out["mid_profile"] = rc.mid(pb)
    R = P[:, sc.F]
    R = R / R.max(axis=0, keepdims=True)
    out["r_gm"] = sc.band_r(R, ((10, 30), (50, 150)))[1]
    out["r_proto"] = sc.r_proto(R)
    out["mid_map"] = float(R[rc.MID][:, rc.GM].mean())
    return out


def main(out: Path) -> None:
    rng = np.random.default_rng(20261110)
    y = sc.PROTO[:, rc.GM].mean(axis=1)
    y = y / y.max()
    targets = {"y_mid": rc.mid(y), "map_mid": float(sc.PROTO[rc.MID][:, rc.GM].mean())}
    res = {"eps": EPS, "penalty": PEN, "starts": rc.STARTS, "seed": 20261110, "targets": targets, "joint": {},
           "circuit": {}}
    for s in SIGMAS:
        K = rc.kern(rc.Z3, s)
        res["joint"][f"{s:.2f}"] = {f"{tn}_{'full' if full else 'auto'}": joint(K, y, m, full, rng)
                                    for tn, m in targets.items() for full in (True, False)}
    assert abs(mf.SIGMA - 0.15) < 1e-12
    for grp, tags in TAGS.items():
        res["circuit"][grp] = {t: circuit(t, y) for t in tags}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
