"""Classify SCI-WS-OSC-1: does the mode survive the local E->E cut?

Reads the two sealed arm receipts and applies jomission.harness.oscillation.compare, whose
four outcomes were predeclared in results/whole_system_oscillator_spec.json before either arm
ran. Nothing is chosen here; the comparison rule and its tolerance are inputs.

    python scripts/whole_system_oscillator_compare.py
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = "results/whole_system_oscillator_spec.json"
INTACT = "results/whole_system_oscillator_single_area.json"
ABLATED = "results/whole_system_oscillator_ee_cut.json"
OUT = "results/whole_system_oscillator_comparison.json"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> int:
    from jomission.harness import oscillation

    spec, a, b = load(SPEC), load(INTACT), load(ABLATED)
    tol = float(spec["predeclared_outcomes"]["rule"].rsplit(" ", 1)[-1])
    areas = sorted(a["oscillation"]["by_area"])
    if areas != sorted(b["oscillation"]["by_area"]):
        raise ValueError("the two arms do not measure the same areas")

    by_area = {}
    for area in areas:
        mi, mb = a["oscillation"]["by_area"][area], b["oscillation"]["by_area"][area]
        by_area[area] = {
            **oscillation.compare(mi, mb, period_tolerance=tol),
            "intact": {k: mi[k] for k in ("periodic_mode_present", "period_ms", "peak_autocorr")},
            "ablated": {k: mb[k] for k in ("periodic_mode_present", "period_ms", "peak_autocorr")},
            "null_threshold": {"intact": mi["null"]["threshold"], "ablated": mb["null"]["threshold"]},
            "peak_autocorr_ratio": round(mb["peak_autocorr"] / mi["peak_autocorr"], 4),
        }
    pooled = oscillation.compare(a["oscillation"]["pooled"], b["oscillation"]["pooled"],
                                 period_tolerance=tol)
    outcomes = sorted({v["outcome"] for v in by_area.values()})

    rec = {
     "lineage_id": spec["lineage_id"],
     "spec": SPEC,
     "arms": {"SINGLE_AREA": INTACT, "EE_CUT": ABLATED},
     "estimator": spec["estimator"]["module"] + "." + spec["estimator"]["function"],
     "rule": spec["predeclared_outcomes"]["rule"],
     "period_tolerance": tol,
     "by_area": by_area,
     "pooled": pooled,
     "outcomes_observed": outcomes,
     "unanimous": len(outcomes) == 1,
     "verdict": (outcomes[0] if len(outcomes) == 1 else "OUTCOME_NOT_UNANIMOUS_ACROSS_AREAS"),
     "arm_gate_verdicts": {"SINGLE_AREA": a["verdict"], "EE_CUT": b["verdict"]},
     "ee_cut_realization": b["ee_cut_realization"]["realized"],
     "scope": spec["shared_regime"]["consequence"],
    }
    (ROOT / OUT).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")

    print(f"{rec['verdict']}   (tolerance {tol})")
    for area in areas:
        v = by_area[area]
        print(f"  {area:6s} {v['intact']['period_ms']!s:>6} -> {v['ablated']['period_ms']!s:>6} ms | "
              f"peak {v['intact']['peak_autocorr']:.3f} -> {v['ablated']['peak_autocorr']:.3f} "
              f"(x{v['peak_autocorr_ratio']:.2f}) | {v['outcome']}")
    p = a["oscillation"]["pooled"], b["oscillation"]["pooled"]
    print(f"  pooled {p[0]['period_ms']} -> {p[1]['period_ms']} ms | "
          f"peak {p[0]['peak_autocorr']:.3f} -> {p[1]['peak_autocorr']:.3f} | {pooled['outcome']}")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
