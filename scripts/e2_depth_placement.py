"""Readout-only test: do the measured unit spectra fit the observed map at other unit depths?

Usage:
  python scripts/e2_depth_placement.py SELECT_TAG [TAG ...]

Hypothesis H (FINDINGS_2026-11-10 §2): r_gm fails because no gamma generator sits at -0.2 to 0. Each run's measured
unit cross-spectra (sum-over-cells row, `e2_scorer.unit_spectra`) stay fixed; only the readout depths move
(zS < zM < zD on a 0.05 grid over the contacts' span -0.6..0.4; sigma 0.15 kept). Exploratory, post hoc: no model
is changed and nothing here is a verdict.
- sweep: per tag, best r_proto, r_ab, r_gm over placements and how many placements reach r_ab and r_gm >= 0.6.
- seed/N held out: the placement is selected on SELECT_TAG alone (max r_proto with r_ab and r_gm >= 0.6) and
  scored unchanged on every other TAG (same layout family; tests seed and N robustness, not generality).
- S only: S moved alone over the grid above M; M and D at their own depths.
- shift null: diagonal spectra with each unit's power shape circularly shifted in frequency (power kept), 20 draws,
  same search and selection; compared with the unshifted diagonal spectra.
- permutation null: the same selection on SELECT_TAG with the unit spectra permuted (5 non-identity orders). An
  order that reaches the same r_proto shows which unit's position the fit needs.
Control: own depths must reproduce each verdict file's r values (1e-9), else it raises.
Writes results/laminar_e2/depth_placement.json.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

GRID = np.round(np.arange(-0.6, 0.4001, 0.05), 2)


def kernel(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.exp(-((mf.CHANNELS[:, None] - z[None, :]) ** 2) / (2 * mf.SIGMA ** 2))


def scores(S_mean: np.ndarray, z) -> tuple[float, float, float]:
    R = sc.rp_map(S_mean, kernel(z))
    rab, rgm = sc.band_r(R, mf.BANDS[mf.FIG_BANDS])
    return sc.r_proto(R), rab, rgm


def table(S_mean: np.ndarray) -> list[tuple]:
    return [(z, *scores(S_mean, z)) for z in itertools.combinations(GRID, 3)]


def select(rows: list[tuple]):
    ok = [(float(rp), [float(x) for x in z]) for z, rp, rab, rgm in rows if rab >= sc.R_MIN and rgm >= sc.R_MIN]
    return max(ok, default=None)


def summary(rows: list[tuple]) -> dict:
    best = {}
    for i, k in enumerate(("r_proto", "r_ab", "r_gm"), start=1):
        z, *v = max(rows, key=lambda r: r[i] if np.isfinite(r[i]) else -np.inf)
        best[k] = (float(v[i - 1]), [float(x) for x in z])
    return {"best": best, "n_placements": len(rows),
            "n_r_ab_and_r_gm_ge_0p6": sum(r[2] >= sc.R_MIN and r[3] >= sc.R_MIN for r in rows),
            "selected": select(rows)}


def spectra(tag: str, d: Path = mf.D) -> tuple[np.ndarray, dict]:
    motif = json.loads((d / f"{tag}_motif.json").read_text(encoding="utf-8"))
    v = json.loads((d / f"{tag}_verdict.json").read_text(encoding="utf-8"))
    _, S, _ = sc.unit_spectra(tag, float(motif["discard_s"]), d)
    S_mean = S.mean(axis=0)
    own = [float(x) for x in motif["unit_depth_rel_L4"]]
    rp, rab, rgm = scores(S_mean, own)
    err = max(abs(rp - v["r_proto"]), abs(rab - v["r_ab"]), abs(rgm - v["r_gm"]))
    if err > 1e-9:
        raise ValueError(f"{tag}: own-depth scores differ from the verdict by {err}")
    return S_mean, {"own_depths": own, "own": {"r_proto": rp, "r_ab": rab, "r_gm": rgm}}


def s_only(S_mean: np.ndarray, own: list[float]) -> dict:
    """Move S alone (above own M); M and D stay at their own depths."""
    rows = [(float(z), *scores(S_mean, [z, *own[1:]])) for z in GRID if z < own[1]]
    return {"rows": [[z, rp, rab, rgm] for z, rp, rab, rgm in rows],
            "best_r_gm": max(([rgm, z] for z, _, _, rgm in rows), default=None)}


def diag(S_mean: np.ndarray) -> np.ndarray:
    return np.einsum("fu,uv->fuv", np.einsum("fuu->fu", S_mean), np.eye(S_mean.shape[1]))


def shift_null(S_mean: np.ndarray, n: int = 20, seed: int = 0) -> dict:
    """Null keeping each unit's power and destroying its spectral shape: diagonal (incoherent) spectra, each unit's
    1..150 Hz power circularly shifted along frequency by its own random offset; same search and selection.
    `diag_real` is the unshifted diagonal spectra through the same path (the comparator for the null)."""
    rng = np.random.default_rng(seed)
    D0 = diag(S_mean)
    draws = []
    for _ in range(n):
        Dn = D0.copy()
        for u in range(D0.shape[1]):
            Dn[1:151, u, u] = np.roll(D0[1:151, u, u], int(rng.integers(1, 150)))
        rows = table(Dn)
        s = select(rows)
        draws.append([None if s is None else s[0], sum(r[2] >= sc.R_MIN and r[3] >= sc.R_MIN for r in rows)])
    real = table(D0)
    return {"n": n, "seed": seed, "draws": draws, "diag_real": [select(real),
            sum(r[2] >= sc.R_MIN and r[3] >= sc.R_MIN for r in real)]}


def main(select_tag: str, tags: list[str], d: Path = mf.D) -> dict:
    S0, own0 = spectra(select_tag, d)
    rows0 = table(S0)
    sel = select(rows0)
    out = {"select_on": select_tag, "sweep": {select_tag: {**own0, **summary(rows0),
                                                           "s_only": s_only(S0, own0["own_depths"])}},
           "held_out": {}, "perm_null": {}, "shift_null": shift_null(S0)}
    for t in tags:
        S, own = spectra(t, d)
        out["sweep"][t] = {**own, **summary(table(S)), "s_only": s_only(S, own["own_depths"])}
        if sel is not None:
            rp, rab, rgm = scores(S, sel[1])
            out["held_out"][t] = {"r_proto": rp, "r_ab": rab, "r_gm": rgm}
    for p in itertools.permutations(range(S0.shape[1])):
        if p != tuple(range(S0.shape[1])):
            out["perm_null"]["".join("SMD"[i] for i in p)] = select(table(S0[:, p][:, :, p]))
    (d / "depth_placement.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main(sys.argv[1], sys.argv[2:])
    print(json.dumps({"selected": o["sweep"][o["select_on"]]["selected"], "held_out": o["held_out"],
                      "perm_null": o["perm_null"], "shift_null": o["shift_null"],
                      "s_only_at_m0p25": {t: [r for r in v["s_only"]["rows"] if r[0] == -0.25]
                                          for t, v in o["sweep"].items()}}))
