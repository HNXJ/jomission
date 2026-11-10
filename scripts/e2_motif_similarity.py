"""Similarity of model spectrolaminar maps to the observed LANSA maps (motif r, ITEM03 M1 form).

Usage:
  python scripts/e2_motif_similarity.py TAG [TAG ...] [--discard S]

Model map: relative power RP(c, f) = P / max_c P on the 21-contact virtual probe at 1..150 Hz, from the RWS
proxy as scripts/e2_motif_figure.py (row per_cell_mean and sum_over_cells). Observed: docs/laminar_motif/data/
lansa_lfp_Y.json (21 bins x 150 Hz x 11 areas). Metrics per row:
  r_proto  mean over f of Pearson r over the 21 bins against the area-mean observed map (primary for fitting)
  r_area   the same against each area, averaged over the 11 areas (ITEM03 M1 primary form)
  r_map    Pearson r of the unrolled 21 x 150 maps against the area-mean map
  r_ab, r_gm  Pearson r of the band-mean depth profiles, 8-30 and 31-150 Hz (LANSA pipeline bands)
  null95   95th percentile of r_proto over 1000 bin permutations of the model map (seed 0)
Frequencies where the model's RP is constant over depth are skipped in the mean over f. Prints one JSON line
per TAG and row; writes results/laminar_e2/<TAG>_similarity.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Yd = json.loads((ROOT / "docs/laminar_motif/data/lansa_lfp_Y.json").read_text(encoding="utf-8"))
Y = np.asarray(Yd["Y"])  # [bin, f, area]
PROTO = Y.mean(axis=2)
F = np.arange(1, 151)


def model_maps(tag: str, discard_s: float) -> dict[str, np.ndarray]:
    z = np.load(mf.D / f"{tag}.npz")
    rec = json.loads((mf.D / f"{tag}.json").read_text(encoding="utf-8"))
    units = list(rec.get("units", rec["sizes"]))
    depth = rec.get("depth_rel_L4") or mf.cs.layout("e2_3unit")[3]
    dt_ms = rec["dt_ms"]
    x = mf.rws(z["g_nS"].astype(np.float64), dt_ms, rec, units)[int(round(discard_s * 1000.0 / dt_ms)):]
    k = int(round(1.0 / dt_ms))
    n1 = x.shape[0] // k
    x = x[: n1 * k].reshape(n1, k, -1).mean(axis=1)
    n_e = np.array([rec["sizes"][u]["E"] for u in units], dtype=float)
    zu = np.array([depth[u] for u in units])
    K = np.exp(-((mf.CHANNELS[:, None] - zu[None, :]) ** 2) / (2 * mf.SIGMA ** 2))
    out = {}
    for key, scale in (("per_cell_mean", 1.0 / n_e), ("sum_over_cells", np.ones_like(n_e))):
        f, S = mf.cross_spectra(x * scale[None], 1000.0, min(1000, n1))
        P = mf.probe_power(S.mean(axis=0), K)
        idx = [int(np.argmin(np.abs(f - fh))) for fh in F]
        P = P[:, idx]
        out[key] = P / P.max(axis=0, keepdims=True)
    return out


def r_over_bins(A: np.ndarray, B: np.ndarray) -> float:
    """Mean over f of Pearson r over bins; A, B [bin, f]."""
    a = A - A.mean(axis=0)
    b = B - B.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    ok = den > 0
    return float(((a * b).sum(axis=0)[ok] / den[ok]).mean())


def pearson(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def score(R: np.ndarray) -> dict:
    ab, gm = (F >= 8) & (F <= 30), (F >= 31) & (F <= 150)
    rng = np.random.default_rng(0)
    null = [r_over_bins(R[rng.permutation(R.shape[0])], PROTO) for _ in range(1000)]
    return {"r_proto": r_over_bins(R, PROTO),
            "r_area": float(np.mean([r_over_bins(R, Y[:, :, a]) for a in range(Y.shape[2])])),
            "r_map": pearson(R.ravel(), PROTO.ravel()),
            "r_ab": pearson(R[:, ab].mean(axis=1), PROTO[:, ab].mean(axis=1)),
            "r_gm": pearson(R[:, gm].mean(axis=1), PROTO[:, gm].mean(axis=1)),
            "null95": float(np.percentile(null, 95))}


def main(tags: list[str], discard_s: float) -> None:
    for tag in tags:
        maps = model_maps(tag, discard_s)
        res = {k: score(R) for k, R in maps.items()}
        (mf.D / f"{tag}_similarity.json").write_text(json.dumps({"tag": tag, "discard_s": discard_s, **res}, indent=1),
                                                     encoding="utf-8")
        for k, v in res.items():
            print(json.dumps({"tag": tag, "row": k, **{m: round(x, 3) for m, x in v.items()}}))


if __name__ == "__main__":
    a = sys.argv[1:]
    d = 2.0
    if "--discard" in a:
        i = a.index("--discard")
        d = float(a[i + 1])
        del a[i:i + 2]
    if not a:
        raise SystemExit(__doc__)
    main(a, d)
