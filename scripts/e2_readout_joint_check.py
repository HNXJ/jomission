"""Known-answer and independent checks of the joint readout optimizer in e2_readout_joint.py (reply to review 4,
2026-11-10).

Usage: python scripts/e2_readout_joint_check.py OUT_JSON

known_answer: 4 synthetic targets made by the three-source readout itself (random C = L L^T, two of rank one), with
  m = the target's own mid; rj.joint (full C, same penalty, EPS and starts) must return feasible with r >= 0.999.
  Known negative: m = 1.2 is above every attainable mid (mid <= 1 by construction), so joint must return infeasible.
rank_one_grid: independent of Nelder-Mead. C = u u^T with u on a Fibonacci grid of M points over the unit sphere
  (sign is irrelevant, scale cancels in r and mid); per sigma and target, the largest r(P, y) with |mid - m| <= EPS.
  Rank one is a subset of the admissible set, so this is a lower bound on the constrained maximum.
random_full: Q random full-rank L (standard normal entries); the same constrained maximum over the sample, a second
  lower bound that includes rank-two and rank-three C.
Neither bound proves a global optimum; agreement with the optimizer within 0.005 is reported, not assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_readout_capacity as rc  # noqa: E402
import e2_readout_joint as rj  # noqa: E402
import e2_scorer as sc  # noqa: E402

M = 200_000
Q = 200_000
SEED = 20261111


def sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def batch_r_mid(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P [21, n] contact power per column -> Pearson r with y and mid of P / max."""
    R = P / P.max(axis=0, keepdims=True)
    md = R[rc.MID].mean(axis=0)
    Pc = P - P.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    r = (Pc * yc[:, None]).sum(axis=0) / np.sqrt((Pc ** 2).sum(axis=0) * (yc ** 2).sum())
    return r, md


def best_feasible(r: np.ndarray, md: np.ndarray, m: float) -> dict | None:
    ok = np.abs(md - m) <= rj.EPS
    if not ok.any():
        return None
    k = int(np.flatnonzero(ok)[np.argmax(r[ok])])
    return {"r": float(r[k]), "mid": float(md[k]), "n_feasible": int(ok.sum())}


def main(out: Path) -> None:
    rng = np.random.default_rng(SEED)
    y = sc.PROTO[:, rc.GM].mean(axis=1)
    y = y / y.max()
    targets = {"y_mid": rc.mid(y), "map_mid": float(sc.PROTO[rc.MID][:, rc.GM].mean())}
    K = rc.kern(rc.Z3, 0.15)
    ka = []
    for j in range(4):
        x = rng.normal(size=6)
        if j < 2:
            x[[2, 4, 5]] = 0.0  # second and third columns of L zero -> rank one
        Ps = rc.power(K, rc.tril(x))
        ys = Ps / Ps.max()
        got = rj.joint(K, ys, rc.mid(ys), True, rng)
        ka.append({"rank": int(np.linalg.matrix_rank(rc.tril(x) @ rc.tril(x).T)), "target_mid": rc.mid(ys),
                   "r": got["r"], "mid": got["mid"], "feasible": got["feasible"]})
    neg = rj.joint(K, y, 1.2, True, rng)
    res = {"seed": SEED, "eps": rj.EPS, "M": M, "Q": Q, "targets": targets,
           "known_answer": ka, "known_negative": {"m": 1.2, "r": neg["r"], "mid": neg["mid"],
                                                  "feasible": neg["feasible"]},
           "sigma": {}}
    assert all(k["feasible"] and k["r"] >= 0.999 for k in ka), ka
    assert not neg["feasible"]
    U = sphere(M)
    Ls = rng.normal(size=(Q, 3, 3))
    for s in rj.SIGMAS:
        Ks = rc.kern(rc.Z3, s)
        r1, m1 = batch_r_mid((Ks @ U) ** 2, y)
        KL = np.einsum("zu,quv->qzv", Ks, Ls)
        r3, m3 = batch_r_mid((KL ** 2).sum(axis=2).T, y)
        res["sigma"][f"{s:.2f}"] = {
            "rank_one_grid": {k: best_feasible(r1, m1, m) for k, m in targets.items()},
            "random_full": {k: best_feasible(r3, m3, m) for k, m in targets.items()},
            "rank_one_mid_max": float(m1.max()), "random_full_mid_max": float(m3.max())}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
