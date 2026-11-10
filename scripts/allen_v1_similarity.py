"""r_proto of an Allen V1 replica map against the observed LANSA map (report only; not a gate).

Usage:
  python scripts/allen_v1_similarity.py TAG [DISCARD_S]

Model map: RP(c, f) = P / max_c P at 1..150 Hz for the four E layers as units (RWS proxy, per-cell and sum rows), as
scripts/allen_v1_figure.py. Scoring is scripts/e2_motif_similarity.py score() (r_proto, r_area, r_map, r_ab, r_gm,
null95), imported and not copied. Two probes are reported:
  spec   21 contacts evenly over 50-829 um (ALLEN_V1_REPLICA_SPEC.md 7)
  lansa  21 contacts at the LANSA bin centres (e2_motif_figure.CHANNELS), the depth grid of the observed map
The observed map is bin-by-bin, so only the lansa probe shares the observed depth grid; the spec row is informative
only for depth-independent metrics. Writes results/allen_v1/<TAG>_similarity.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import allen_v1_figure as af  # noqa: E402
import e2_motif_figure as e2  # noqa: E402
import e2_motif_similarity as sim  # noqa: E402
from jomission.laminar import allen_v1_replica as av  # noqa: E402

OUT = ROOT / "results" / "allen_v1"


def rp_maps(tag: str, discard_s: float) -> dict[str, dict[str, np.ndarray]]:
    z = np.load(OUT / f"{tag}.npz")
    rec = json.loads((OUT / f"{tag}.json").read_text(encoding="utf-8"))
    data = av.load()
    mids = af.layer_mids(data)
    mid_l4 = mids["L4"]
    depth = (np.array([mids[L] for L in af.LAYERS]) - mid_l4) / af.THICK_UM
    n_e = np.array([sum(rec["sizes"][p] for p in data["populations"] if av.layer_of(p) == L and av.kind(p) == "E")
                    for L in af.LAYERS], dtype=float)
    x = af.rws_layers(z["g_nS"].astype(np.float64), rec["dt_ms"])[int(round(discard_s * 1000.0 / rec["dt_ms"])):]
    k = int(round(1.0 / rec["dt_ms"]))
    n1 = x.shape[0] // k
    x = x[: n1 * k].reshape(n1, k, -1).mean(axis=1)
    probes = {"spec": (np.asarray(af.PROBE_UM) - mid_l4) / af.THICK_UM, "lansa": e2.CHANNELS}
    out: dict[str, dict[str, np.ndarray]] = {}
    for pname, ys in probes.items():
        K = np.exp(-((ys[:, None] - depth[None, :]) ** 2) / (2 * af.SIGMA ** 2))
        out[pname] = {}
        for key, scale in (("per_cell_mean", 1.0 / n_e), ("sum_over_cells", np.ones_like(n_e))):
            f, S = e2.cross_spectra(x * scale[None], 1000.0, min(1000, n1))
            P = e2.probe_power(S.mean(axis=0), K)
            idx = [int(np.argmin(np.abs(f - fh))) for fh in sim.F]
            P = P[:, idx]
            out[pname][key] = P / P.max(axis=0, keepdims=True)
    return out


def main(tag: str, discard_s: float) -> None:
    maps = rp_maps(tag, discard_s)
    res = {p: {k: sim.score(R) for k, R in rows.items()} for p, rows in maps.items()}
    (OUT / f"{tag}_similarity.json").write_text(
        json.dumps({"tag": tag, "discard_s": discard_s, "observed": "docs/laminar_motif/data/lansa_lfp_Y.json",
                    "report_only": True, **res}, indent=1), encoding="utf-8")
    for p, rows in res.items():
        for k, v in rows.items():
            print(json.dumps({"tag": tag, "probe": p, "row": k, **{m: round(x, 3) for m, x in v.items()}}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (1, 2):
        raise SystemExit(__doc__)
    main(a[0], float(a[1]) if len(a) == 2 else 2.0)
