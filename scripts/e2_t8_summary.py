"""T8 summary (docs/laminar_motif/SEAL_T8.md): per-run outcomes, stop states and the sealed paired decision rule.

Usage: python scripts/e2_t8_summary.py OUT_JSON

Per run t8_{arm}_n4000_s{seed}_T8: r_ab, r_gm, r_proto, crossing_2se, pass (verdict file); mid-depth relative
gamma (e2_mid_gamma.band_means); unit rates and stop state (e2_select). Per control arm c and outcome X:
d_s = X(ref, s) - X(c, s); "ref higher" if all d_s > 0 and mean > 2 SD / sqrt(3) (SD ddof 1), "control higher"
for the mirror case, else "no consistent difference". An arm with a missing, stopped or refused seed, or a ref
with one, is "not evaluable".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_mid_gamma as mg  # noqa: E402
import e2_motif_figure as mf  # noqa: E402
import e2_scorer as sc  # noqa: E402
import e2_select as sel  # noqa: E402

ARMS = ("ref", "dflat", "drev", "rflat", "rrev")
SEEDS = (1, 2, 3)
OUTCOMES = ("r_ab", "r_gm", "r_proto", "mid_gamma")  # r_ab primary
DISCARD_S = 2.0


def run(arm: str, s: int, d: Path = mf.D) -> dict:
    tag = f"t8_{arm}_n4000_s{s}_T8"
    vf = d / f"{tag}_verdict.json"
    if not vf.exists():
        return {"tag": tag, "missing": True}
    v = json.loads(vf.read_text(encoding="utf-8"))
    r = sel.rates(tag, DISCARD_S, d)
    _, S, K = sc.unit_spectra(tag, DISCARD_S, d)
    mid = mg.band_means(sc.rp_map(S.mean(axis=0), K))["mid [-0.2,0]"]
    return {"tag": tag, "missing": False, "refused": v.get("reason") is not None, "stop": sel.stop_state(r),
            "rates_E_hz": {u: r[u]["E"] for u in r}, "r_ab": v["r_ab"], "r_gm": v["r_gm"], "r_proto": v["r_proto"],
            "mid_gamma": mid, "crossing_2se": v["crossing_2se"], "pass": v["pass"]}


def evaluable(rows: list[dict]) -> bool:
    return all(not x["missing"] and not x["refused"] and x["stop"] is None for x in rows)


def decide(d: np.ndarray) -> str:
    m, sd = float(d.mean()), float(d.std(ddof=1))
    if (d > 0).all() and m > 2 * sd / np.sqrt(len(d)):
        return "ref higher"
    if (d < 0).all() and m < -2 * sd / np.sqrt(len(d)):
        return "control higher"
    return "no consistent difference"


def main(out: Path) -> dict:
    runs = {a: [run(a, s) for s in SEEDS] for a in ARMS}
    res = {"seal": "docs/laminar_motif/SEAL_T8.md", "runs": runs, "comparisons": {}}
    for c in ARMS[1:]:
        if not (evaluable(runs["ref"]) and evaluable(runs[c])):
            res["comparisons"][c] = "not evaluable"
            continue
        res["comparisons"][c] = {}
        for x in OUTCOMES:
            d = np.array([a[x] - b[x] for a, b in zip(runs["ref"], runs[c])])
            res["comparisons"][c][x] = {"d": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                        "decision": decide(d)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]))
    for c, v in r["comparisons"].items():
        print(c, v if isinstance(v, str) else {x: (round(v[x]["mean"], 3), v[x]["decision"]) for x in v})
