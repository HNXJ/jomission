"""Grid check of the no-cross-term (diagonal C) joint result in e2_readout_joint.py (reply to review 3, 2026-11-10).

Usage: python scripts/e2_readout_joint_grid.py OUT_JSON

Independent of Nelder-Mead: power shares (a_S, a_M, a_D) on a simplex grid of step 1/N (N = 400), contact power
P = (K^2) a with the declared Gaussian kernel. Per sigma in (0.15, 0.20): the largest mid-depth relative gamma over
the grid, and for each target m (y_mid, map_mid) the largest Pearson r(P, y) among grid points with |mid - m| <= EPS
(None when no grid point is feasible). Kernel, targets and EPS as in e2_readout_joint.py.
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

N = 400


def main(out: Path) -> None:
    y = sc.PROTO[:, rc.GM].mean(axis=1)
    y = y / y.max()
    targets = {"y_mid": rc.mid(y), "map_mid": float(sc.PROTO[rc.MID][:, rc.GM].mean())}
    res = {"grid_step": 1 / N, "eps": rj.EPS, "targets": targets, "sigma": {}}
    for s in rj.SIGMAS:
        K2 = rc.kern(rc.Z3, s) ** 2
        best_mid, best = -1.0, {k: None for k in targets}
        for i in range(N + 1):
            for j in range(N + 1 - i):
                a = np.array([i, j, N - i - j]) / N
                P = K2 @ a
                md = rc.mid(P)
                best_mid = max(best_mid, md)
                r = rc.corr(P, y)
                for k, m in targets.items():
                    if abs(md - m) <= rj.EPS and (best[k] is None or r > best[k]["r"]):
                        best[k] = {"r": r, "mid": md, "share": a.tolist()}
        res["sigma"][f"{s:.2f}"] = {"auto_mid_max": best_mid, "auto_best_feasible": best}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
