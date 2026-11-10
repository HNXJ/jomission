"""Leave-one-area-out check of the readout-capacity fit (owner request 2026-11-10).

Usage: python scripts/e2_readout_capacity_loao.py OUT_JSON

For each of the 11 LANSA areas a (n = 11 folds) and each kernel width sigma:
  train target = the 50-150 Hz band-mean depth profile of the mean map over the other 10 areas (/ max);
  test target  = the same profile of area a alone.
  The source covariance C (full, cross terms allowed; auto, diagonal) is chosen to maximize Pearson r with the
  train target only (the objectives of e2_readout_capacity.maxr_full / maxr_auto, same starts and optimizer);
  r is then computed
  between that fixed model profile and the test target.
  template_r: r between the train target itself and the test target, the held-out r of the 10-area mean profile
  with no readout model; it bounds what any model fitted to the train target can be expected to reach.
Reports per fold and mean, SD, min, max over folds; the in-sample bound on the 11-area mean is in
data/readout_capacity.json (maxr_full / maxr_auto).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_readout_capacity as rc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def stats(v: list[float]) -> dict:
    a = np.asarray(v)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "min": float(a.min()), "max": float(a.max())}


def main(out: Path) -> None:
    d = json.loads((ROOT / "docs/laminar_motif/data/lansa_lfp_Y.json").read_text(encoding="utf-8"))
    Y = np.asarray(d["Y"], dtype=float)  # [bin, f, area]
    areas = d["axes"]["areas"]
    assert Y.shape[1] == rc.GM.size and Y.shape[2] == len(areas) == 11
    band = Y[:, rc.GM, :].mean(axis=1)  # [bin, area]
    rng = np.random.default_rng(20261110)
    res = {"n_folds": len(areas), "areas": areas, "starts": rc.STARTS, "seed": 20261110, "sigma": {}}
    for s in rc.SIGMAS:
        K3 = rc.kern(rc.Z3, s)
        folds = []
        for i, a in enumerate(areas):
            keep = [j for j in range(len(areas)) if j != i]
            ytr = Y[:, :, keep].mean(axis=2)[:, rc.GM].mean(axis=1)
            ytr = ytr / ytr.max()
            yte = band[:, i]
            f = {"area": a, "template_r": rc.corr(ytr, yte)}
            for name, P in (("full", fit_full(K3, ytr, rng)), ("auto", fit_auto(K3, ytr, rng))):
                f[name] = {"train_r": rc.corr(P, ytr), "test_r": rc.corr(P, yte)}
            folds.append(f)
        res["sigma"][f"{s:.2f}"] = {
            "folds": folds,
            "template_r": stats([f["template_r"] for f in folds]),
            **{f"{k}_{w}": stats([f[k][w] for f in folds]) for k in ("full", "auto") for w in ("train_r", "test_r")},
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "folds"} for k, v in res["sigma"].items()}, indent=1))


def fit_full(K: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Profile of the full C maximizing r with y (the objective of rc.maxr_full)."""
    return rc.power(K, rc.tril(rc.best_of(lambda v: -rc.corr(rc.power(K, rc.tril(v)), y), 6, rng)))


def fit_auto(K: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Profile of the diagonal C maximizing r with y (the objective of rc.maxr_auto)."""
    return (K ** 2) @ (rc.best_of(lambda v: -rc.corr((K ** 2) @ (v ** 2), y), 3, rng) ** 2)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
