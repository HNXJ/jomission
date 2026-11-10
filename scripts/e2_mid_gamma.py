"""Mean relative gamma power by depth band, model runs vs the observed 11-area map (manuscript, Results R3).

Usage: python scripts/e2_mid_gamma.py OUT_JSON TAG [TAG ...]

For each TAG the scorer's map R (21 contacts x 1..150 Hz, relative power per frequency, sum-over-cells RWS, 2 s
discard) is averaged over 50-150 Hz and over the contacts in each depth band; PROTO gives the observed values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402

BANDS = {"S [-0.6,-0.3]": (-0.6, -0.3), "mid [-0.2,0]": (-0.2, 0.0), "D [0.15,0.4]": (0.15, 0.4)}
GM = (sc.F >= 50) & (sc.F <= 150)


def band_means(R: np.ndarray) -> dict[str, float]:
    g = R[:, GM].mean(axis=1)
    return {k: float(g[(mf.CHANNELS >= a - 1e-9) & (mf.CHANNELS <= b + 1e-9)].mean()) for k, (a, b) in BANDS.items()}


def main(argv: list[str]) -> None:
    out, tags = Path(argv[0]), argv[1:]
    res = {"observed": band_means(sc.PROTO), "runs": {}}
    for t in tags:
        f, S, K = sc.unit_spectra(t, 2.0, mf.D)
        res["runs"][t] = band_means(sc.rp_map(S.mean(axis=0), K))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1:])
