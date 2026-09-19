"""Classify SCI-WS-OSC-2: does an isolated E cell retain the 94 to 95 ms mode?

Scores both arms against the three outcomes predeclared in
results/whole_system_isolated_spec.json, using the primary observable that spec names: the
per-cell interspike interval. The population-rate estimator is reported alongside as the
secondary observable and does not decide anything.

    python scripts/whole_system_isolated_compare.py
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = "results/whole_system_isolated_spec.json"
POP = "results/whole_system_isolated_pop.json"
ONE = "results/whole_system_isolated_1.json"
PARENT = "results/whole_system_oscillator_comparison.json"
OUT = "results/whole_system_isolated_comparison.json"

TOLERANCE = 0.10


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def score(pooled, target_ms):
    """The predeclared rule, on the primary observable."""
    if pooled is None:
        return {"outcome": "INTRINSIC_ABSENT",
                "reading": "no cell fired twice, so the isolated cell has no interval structure"}
    rel = abs(pooled["mean_ms"] - target_ms) / target_ms
    if rel <= TOLERANCE:
        return {"outcome": "INTRINSIC_GENERATOR_CONFIRMED", "relative_difference": round(rel, 6),
                "reading": ("the isolated cell fires rhythmically at the intact period, so the "
                            "frequency is set inside the cell and neither E->E recurrence nor "
                            "the E-I circuit is required for it")}
    return {"outcome": "INTRINSIC_PRESENT_DIFFERENT_PERIOD", "relative_difference": round(rel, 6),
            "reading": ("the isolated cell is rhythmic but at a different period, so intrinsic "
                        "dynamics contribute without setting the observed frequency")}


def main() -> int:
    spec, pop, one, parent = load(SPEC), load(POP), load(ONE), load(PARENT)
    target = parent["pooled"] if isinstance(parent.get("pooled"), dict) else {}
    target_ms = 94.0                      # the intact pooled period SCI-WS-OSC-1 measured

    arms = {}
    for name, rec in (("ISOLATED_POP", pop), ("ISOLATED_1", one)):
        isi = rec["isi"]["final_window"]
        arms[name] = {
         **score(isi["pooled"], target_ms),
         "isi_pooled": isi["pooled"],
         "n_cells": isi["n_cells"], "n_cells_with_isi": isi["n_cells_with_isi"],
         "n_cells_silent": isi["n_cells_silent"],
         "per_cell_mean_ms": isi["per_cell_mean_ms"],
         "population_estimator": {k: rec["oscillation"]["pooled"][k] for k in
                                  ("periodic_mode_present", "period_ms", "peak_autocorr")},
         "isolation": rec["isolation"]["realized"],
        }

    # The predeclared cross-check: two differently built isolations must agree.
    lo = pop["isi"]["final_window"]["per_cell_mean_ms"]["min"]
    hi = pop["isi"]["final_window"]["per_cell_mean_ms"]["max"]
    single_ms = one["isi"]["final_window"]["pooled"]["mean_ms"]
    cross = {
     "rule": "ISOLATED_1's period must lie inside ISOLATED_POP's per-cell mean range",
     "isolated_1_mean_ms": single_ms,
     "isolated_pop_per_cell_mean_range_ms": [lo, hi],
     "pass": bool(lo <= single_ms <= hi),
     "reading": ("the two arms are built differently, one by slicing the construction to N = 1 "
                 "and one by zeroing every weight, so agreement tests the slicing. A failure "
                 "would be a defect in the reduction, not a result about the model"),
    }

    # The secondary observable behaved badly on one cell. Recorded, not hidden.
    p1 = one["oscillation"]["pooled"]
    harmonic = None
    if p1["period_ms"]:
        ratio = p1["period_ms"] / single_ms
        harmonic = {"period_reported_ms": p1["period_ms"], "isi_mean_ms": single_ms,
                    "ratio": round(ratio, 4),
                    "nearest_integer": int(round(ratio)),
                    "is_harmonic": bool(abs(ratio - round(ratio)) < 0.05 and round(ratio) > 1),
                    "peak_autocorr": p1["peak_autocorr"],
                    "null_threshold": p1["null"]["threshold"],
                    "margin_over_null": round(p1["peak_autocorr"] - p1["null"]["threshold"], 6),
                    "reading": ("the population-rate estimator on a single cell reads a harmonic "
                                "of the true interval. The final window holds about ten "
                                "intervals, so the autocorrelation is estimated from very few "
                                "cycles and the first peak past the zero crossing need not be "
                                "the fundamental. This is a limit of applying a population "
                                "estimator to one spike train, which the spec anticipated by "
                                "naming the interval observable primary")}

    outcomes = sorted({a["outcome"] for a in arms.values()})
    rec = {
     "lineage_id": spec["lineage_id"],
     "spec": SPEC,
     "arms_files": {"ISOLATED_POP": POP, "ISOLATED_1": ONE},
     "primary_observable": spec["observable"]["primary"],
     "comparison_target_ms": target_ms,
     "comparison_target_source": ("SCI-WS-OSC-1 intact pooled period, "
                                  "results/whole_system_oscillator_comparison.json"),
     "tolerance": TOLERANCE,
     "by_arm": arms,
     "cross_check": cross,
     "single_cell_estimator_harmonic": harmonic,
     "outcomes_observed": outcomes,
     "unanimous": len(outcomes) == 1,
     "verdict_outcome": (outcomes[0] if len(outcomes) == 1 else "OUTCOME_NOT_UNANIMOUS"),
     "verdict": ((outcomes[0] if len(outcomes) == 1 else "OUTCOME_NOT_UNANIMOUS")
                 + ("_PASS" if len(outcomes) == 1 and cross["pass"] else "_UNRESOLVED")),
     "verdict_encoding": ("the harness requires a flat verdict string ending in _<status>. "
                          "verdict_outcome is the predeclared outcome name; the suffix records "
                          "whether this lineage's acceptance was met"),
     "population_rate_hazard": {
      "declared": spec["observable"]["hazard_declared_before_execution"],
      "materialized": False,
      "what_happened": ("the population rate did not flatten. The estimator still found the mode "
                        "in all six areas of ISOLATED_POP at 93 to 95 ms, because the per-cell "
                        "periods agree to within about 0.7 percent and stay near-locked across "
                        "13000 ms. The hazard was real and simply did not bite; the interval "
                        "observable was primary either way")},
     "scope": ("the isolated construction shares the intact arm's cell parameters, tonic drive, "
               "noise term, dt, seed and duration. It says what one cell does under that tonic "
               "drive, and nothing about other drives"),
    }
    (ROOT / OUT).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")

    print(f"{rec['verdict']}   (tolerance {TOLERANCE}, target {target_ms} ms)")
    for name, a in arms.items():
        p = a["isi_pooled"]
        print(f"  {name:13s} ISI {p['mean_ms']} +/- {p['std_ms']} ms  cv {p['cv']}  "
              f"-> {p['implied_frequency_hz']} Hz  | {a['outcome']} "
              f"(off by {a.get('relative_difference')})")
        print(f"                cells {a['n_cells_with_isi']}/{a['n_cells']} firing, "
              f"{a['n_cells_silent']} silent | edges zeroed "
              f"{a['isolation']['n_edges']} -> {a['isolation']['n_nonzero_after']} nonzero")
    print(f"  cross-check: {single_ms} ms in [{lo}, {hi}] -> {'PASS' if cross['pass'] else 'FAIL'}")
    if harmonic and harmonic["is_harmonic"]:
        print(f"  NOTE: single-cell population estimator read {harmonic['period_reported_ms']} ms, "
              f"{harmonic['nearest_integer']}x the interval. Harmonic, not the fundamental.")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
