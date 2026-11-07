"""E0 verdict rule (docs/laminar_motif/E0_SEAL.md, Reference and Verdict rule).

Pure function of per-seed analysis records; no simulation here. Means over the five scored seeds are compared
with the P5 references inside the P6 margins. An undefined estimand, a run that did not complete, or a missing
seed is never a pass.
"""
from __future__ import annotations

STEPS_EXPECTED = 120_000  # 12 s at dt 0.1 ms
SEEDS = (1, 2, 3, 4, 5)

# estimand -> (reference, margin, kind); kind "abs" margin in the estimand's unit, "rel" fraction of the reference
REFERENCE = {
    "f_pk_Hz": (33.8, 1.0, "abs"),
    "lag_PV_SOM_ms": (7.04, 0.34, "abs"),
    "rate_E_Hz": (1.106, 0.061, "rel"),
    "rate_PV_Hz": (11.37, 0.012, "rel"),
    "rate_SST_Hz": (7.05, 0.082, "rel"),
}


def _estimates(rec: dict) -> dict:
    """Map one seed's analyze_run record to the five estimands (None when undefined)."""
    if rec["undefined"] or not rec["oscillates"]:
        f_pk = None
    else:
        f_pk = rec["peak_freq_E_Hz"]
    rates = rec["rates_Hz"]
    return {
        "f_pk_Hz": f_pk,
        "lag_PV_SOM_ms": rec["lag_PV_SOM_ms"],
        "rate_E_Hz": rates["E"],
        "rate_PV_Hz": rates["PV"],
        "rate_SST_Hz": rates["SOM"],
    }


def verdict(seed_records: dict) -> dict:
    """seed_records: {seed: {"steps": int, "analysis": analyze_run record}} -> verdict record.

    verdict is PASS, FAIL or INCOMPLETE. INCOMPLETE: a seed of SEEDS is absent or did not run 12 s.
    FAIL: any estimand undefined or outside its margin. PASS: all five inside.
    """
    missing = [s for s in SEEDS if s not in seed_records]
    short = [s for s in SEEDS if s in seed_records and seed_records[s]["steps"] != STEPS_EXPECTED]
    extra = sorted(set(seed_records) - set(SEEDS))
    if missing or short or extra:
        return {"verdict": "INCOMPLETE", "missing_seeds": missing, "short_seeds": short, "unexpected_seeds": extra,
                "estimands": {}}
    per_seed = {s: _estimates(seed_records[s]["analysis"]) for s in SEEDS}
    out = {}
    for name, (ref, margin, kind) in REFERENCE.items():
        vals = [per_seed[s][name] for s in SEEDS]
        if any(v is None for v in vals):
            out[name] = {"status": "UNDEFINED", "reference": ref, "mean": None, "per_seed": vals}
            continue
        mean = sum(vals) / len(vals)
        dev = abs(mean - ref)
        limit = margin if kind == "abs" else margin * ref
        out[name] = {"status": "IN" if dev <= limit else "OUT", "reference": ref, "mean": mean,
                     "deviation": dev, "limit": limit, "kind": kind, "per_seed": vals}
    ok = all(v["status"] == "IN" for v in out.values())
    return {"verdict": "PASS" if ok else "FAIL", "missing_seeds": [], "short_seeds": [], "unexpected_seeds": [],
            "estimands": out}
