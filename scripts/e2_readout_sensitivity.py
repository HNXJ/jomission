"""Readout sensitivity of the scored map (manuscript, reply to review 2026-11-10).

Usage: python scripts/e2_readout_sensitivity.py OUT_JSON TAG [TAG ...]

Per TAG, from the scorer's window-mean cross-spectra S[f, u, v] (sum-over-cells RWS, 2 s discard):
  sigma: kernel width 0.10, 0.15, 0.20, 0.25 -> r_proto, r_ab, r_gm, mid-depth relative gamma (scorer bands).
  decomposition at the declared sigma: P_z(f) = sum_u K_zu^2 S_uu (auto) + sum_{u!=v} K_zu K_zv Re S_uv (cross);
    over 50-150 Hz and the contacts -0.2..0: cross fraction of P, and each unit's share of the auto part;
    mid-depth relative gamma of the auto-only map; at the contact that sets the per-frequency maximum, the unit
    with the largest auto share (counted over 50-150 Hz).
  gain: the factor on unit M's spectrum (row and column, i.e. M's signal amplitude squared) at which mid-depth
    relative gamma reaches the observed value, searched on a log grid 1..1e4; null if not reached.
Exact identity check: auto + cross equals the scorer's probe power to 1e-9 relative.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

SIGMAS = (0.10, 0.15, 0.20, 0.25)
BANDS = ((10, 30), (50, 150))
GM = (sc.F >= 50) & (sc.F <= 150)
MID = (mf.CHANNELS >= -0.2 - 1e-9) & (mf.CHANNELS <= 0.0 + 1e-9)


def kernel(depths: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-((mf.CHANNELS[:, None] - depths[None, :]) ** 2) / (2 * sigma ** 2))


def rel(P: np.ndarray) -> np.ndarray:
    """P [C, f>=151] -> relative map [C, 150] at 1..150 Hz."""
    Q = P[:, sc.F]
    return Q / Q.max(axis=0, keepdims=True)


def mid_gamma(R: np.ndarray) -> float:
    return float(R[MID][:, GM].mean())


def run(tag: str, observed_mid: float) -> dict:
    f, S, K0 = sc.unit_spectra(tag, 2.0, mf.D)
    Sm = S.mean(axis=0)
    rec = json.loads((Path(mf.D) / f"{tag}.json").read_text(encoding="utf-8"))
    units = list(rec["units"]) if "units" in rec else list(rec["sizes"])
    depth = rec.get("depth_rel_L4") or mf.cs.layout("e2_3unit")[3]
    zu = np.array([depth[u] for u in units])
    out = {"units": units, "sigma": {}}
    for s in SIGMAS:
        R = rel(mf.probe_power(Sm, kernel(zu, s)))
        rab, rgm = sc.band_r(R, BANDS)
        out["sigma"][f"{s:.2f}"] = {"r_proto": sc.r_proto(R), "r_ab": rab, "r_gm": rgm, "mid_gamma": mid_gamma(R)}
    K = kernel(zu, mf.SIGMA)
    assert np.allclose(K, K0), "kernel differs from the scorer's"
    P = mf.probe_power(Sm, K)
    diag = np.einsum("fuu->fu", Sm).real  # [f, U]
    auto_u = (K ** 2)[:, None, :] * diag[None, :, :]  # [C, f, U]
    auto = auto_u.sum(axis=2)
    cross = P - auto
    Pg = P[:, sc.F][:, GM]
    assert np.allclose(auto + cross, P, rtol=1e-9, atol=0)
    a_mid = auto_u[MID][:, sc.F][:, GM]  # [Cmid, fg, U]
    share = a_mid.sum(axis=(0, 1)) / a_mid.sum()
    imax = P[:, sc.F][:, GM].argmax(axis=0)  # contact of the per-frequency maximum
    top = auto_u[:, sc.F][:, GM].argmax(axis=2)  # [C, fg] unit with largest auto term
    lead = np.array([top[imax[j], j] for j in range(imax.size)])
    out["decomposition"] = {
        "cross_fraction_mid_gamma": float(cross[MID][:, sc.F][:, GM].sum() / Pg[MID].sum()),
        "auto_share_mid_gamma": {u: float(v) for u, v in zip(units, share)},
        "mid_gamma_auto_only": mid_gamma(rel(auto)),
        "max_contact_lead_unit_counts": {u: int((lead == i).sum()) for i, u in enumerate(units)},
        "max_contact_depths": sorted({float(mf.CHANNELS[i]) for i in imax}),
    }
    m = units.index("M")
    gain = None
    for gval in np.logspace(0, 4, 81):
        Sg = Sm.copy()
        Sg[:, m, :] *= np.sqrt(gval)
        Sg[:, :, m] *= np.sqrt(gval)
        if mid_gamma(rel(mf.probe_power(Sg, K))) >= observed_mid:
            gain = float(gval)
            break
    out["m_power_gain_to_observed_mid"] = gain
    return out


def main(argv: list[str]) -> None:
    out, tags = Path(argv[0]), argv[1:]
    obs = mid_gamma(sc.PROTO)
    res = {"observed_mid_gamma": obs, "sigmas": list(SIGMAS), "declared_sigma": mf.SIGMA, "runs": {}}
    for t in tags:
        res["runs"][t] = run(t, obs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1:])
