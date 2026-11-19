"""E1b scorer (docs/laminar_motif/E1B_SEAL.md): gates 3 and 4, multimodal flags, then E1's verdict rule per arm.

  python -B scripts/e1b_score.py --runs DIR --out results/laminar_e1b/e1b_result.json

Gate 3 (positive control): the sealed E1 20-80 Hz readout on the saved E1 counts must reproduce every recorded
E1 call (oscillates, and f_pk where it oscillates); missing counts stop the scorer. Gate 4 (null): E1's
permutation test on f_pk independent of rho must not pass. The verdict is jomission.laminar.e1_verdict.verdict,
unchanged; E1b seeds 6, 7, 8 are passed to it as 1, 2, 3 (the rule reads seed labels only as keys).
Exit 0 when scored, 2 when a gate fails or a run is missing.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from jaxfne.analysis import aperiodic_spectral_peak, oscillation_call  # noqa: E402
from jaxfne.analysis.spectral import aperiodic_log10  # noqa: E402

from jomission.laminar import caminos_analysis as ca  # noqa: E402
from jomission.laminar import e1_verdict as ev  # noqa: E402

DT = 1e-4
BAND = (10.0, 80.0)
SEEDS = (6, 7, 8)
SEED_KEY = {6: 1, 7: 2, 8: 3}
E1_COUNTS = Path("D:/tmp/_temp_o51_a1")
E1_JSON = REPO / "results/laminar_e1"


def readout(e_rate, band):
    """Primary peak (the sealed call) and the multimodal flag (seal, peak handling rule 3)."""
    f, p = ca.e_welch(e_rate, DT)
    pk = aperiodic_spectral_peak(f, p, fit_lo_hz=ca.FIT_LO_HZ, fit_hi_hz=ca.FIT_HI_HZ, band_lo_hz=band[0],
                                 band_hi_hz=band[1], excl_half_hz=ca.EXCL_HALF_HZ, edge_hz=ca.EDGE_HZ)
    osc = oscillation_call(pk["peak_excess"], pk["edge_ok"], ca.PE_THRESHOLD)
    return {"peak_freq_Hz": pk["peak_freq_Hz"], "peak_excess": pk["peak_excess"], "edge_ok": pk["edge_ok"],
            "oscillates": osc, **multimodal(f, p, pk, band)}


def multimodal(f, p, pk, band):
    """Second local maximum in the band, >= 8 Hz from f_pk, with excess >= 4 over the refit background."""
    excess = p / 10.0 ** aperiodic_log10(f, *pk["fit"])
    idx = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    cands = [i for i in idx[1:-1] if p[i] > p[i - 1] and p[i] > p[i + 1]
             and abs(f[i] - pk["peak_freq_Hz"]) >= ca.EXCL_HALF_HZ and excess[i] >= ca.PE_THRESHOLD]
    if not cands:
        return {"multimodal": False, "second_peak_Hz": None, "second_peak_excess": None}
    j = max(cands, key=lambda i: excess[i])
    return {"multimodal": True, "second_peak_Hz": float(f[j]), "second_peak_excess": float(excess[j])}


def e_rate(counts, n_e):
    keep = counts[ca.kept_slice(counts.shape[0], DT)]
    return keep[:, 0] / DT / n_e


def gate3_positive_control():
    rows = []
    for rho in ev.RHOS:
        for s in (1, 2, 3):
            rec = json.loads((E1_JSON / f"e1_rho{rho:g}_seed{s}.json").read_text(encoding="utf-8"))["analysis"]
            if rho == 1.0:
                hits = sorted(E1_COUNTS.glob(f"*/e0_seed{s}_counts.npy"))
            else:
                hits = sorted(E1_COUNTS.glob(f"*/e1_rho{rho:g}_seed{s}_counts.npy"))
            if not hits:
                rows.append({"rho": rho, "seed": s, "ok": False, "why": "counts missing"})
                continue
            r = readout(e_rate(np.load(hits[0]), 8000), (ca.BAND_LO_HZ, ca.BAND_HI_HZ))
            ok = r["oscillates"] == rec["oscillates"] and (
                not rec["oscillates"] or r["peak_freq_Hz"] == rec["peak_freq_E_Hz"])
            rows.append({"rho": rho, "seed": s, "ok": bool(ok), "file": str(hits[0])})
    return {"pass": len(rows) == 15 and all(x["ok"] for x in rows), "n_checked": len(rows), "rows": rows}


def gate4_null():
    rng = np.random.default_rng(20261119)
    rhos = [list(ev.RHOS)] * 3
    freqs = [list(30.0 + rng.normal(0, 0.5, 5)) for _ in range(3)]
    p = ev.permutation_p_val(rhos, freqs)
    return {"pass": p >= ev.ALPHA_PERM, "p_value": p}


def score_arm(arm, recs):
    runs = []
    for (a, rho, s), rec in recs.items():
        if a not in (arm, "AB"):
            continue
        r = rec["readout"]
        runs.append({"rho": rho, "seed": SEED_KEY[s], "steps": rec["steps"],
                     "analysis": {"oscillates": r["oscillates"], "peak_freq_E_Hz": r["peak_freq_Hz"],
                                  "peak_excess_E": r["peak_excess"], "undefined": [],
                                  "rates_Hz": rec.get("rates_Hz")}})
    v = ev.verdict(runs)
    v["seed_labels"] = {str(k): v2 for v2, k in SEED_KEY.items()}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs_dir = Path(a.runs)
    g3, g4 = gate3_positive_control(), gate4_null()
    res = {"schema": "jomission.e1b_result.v1", "seal": "docs/laminar_motif/E1B_SEAL.md",
           "gate3_positive_control": g3, "gate4_null": g4}
    code = 0
    if not (g3["pass"] and g4["pass"]):
        res["verdict"] = "STOPPED: gate failed"
        code = 2
    else:
        recs, missing = {}, []
        for arm, rho, s in ([("A", r, s) for r in (0.25, 0.5, 2.0, 4.0) for s in SEEDS]
                            + [("B", r, s) for r in (0.25, 0.5, 2.0, 4.0) for s in SEEDS]
                            + [("A", 1.0, s) for s in SEEDS]):
            tag = f"e1b{arm}_rho{rho:g}_seed{s}"
            rp, cp = runs_dir / f"{tag}.receipt.json", runs_dir / f"{tag}_counts.npy"
            if not (rp.exists() and cp.exists()):
                missing.append(tag)
                continue
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("outcome") != "complete":
                missing.append(tag)
                continue
            rec["readout"] = readout(e_rate(np.load(cp), rec["realized_sizes"]["E"]), BAND)
            key = ("AB" if rho == 1.0 else arm, rho, s)
            recs[key] = rec
        res["runs"] = {f"{k[0]}_{k[1]:g}_s{k[2]}": {"readout": v["readout"], "rates_Hz": v.get("rates_Hz"),
                                                   "sizes": v["realized_sizes"], "steps": v["steps"]}
                       for k, v in recs.items()}
        if missing:
            res["verdict"] = "INCOMPLETE"
            res["missing"] = missing
            code = 2
        else:
            res["arm_A"] = score_arm("A", recs)
            res["arm_B"] = score_arm("B", recs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"gate3": g3["pass"], "gate4": g4["pass"],
                      "A": res.get("arm_A", {}).get("verdict"), "B": res.get("arm_B", {}).get("verdict"),
                      "missing": res.get("missing")}))
    sys.exit(code)


if __name__ == "__main__":
    main()
