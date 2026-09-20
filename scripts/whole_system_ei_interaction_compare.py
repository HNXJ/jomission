"""SCI-EI-INTERACTION-1 compare: classify the 2x2 factorial from four cell receipts.

Read-only. Each cell ran the sealed intact construction plus its predeclared
repairs; this script computes departures from additivity and applies the sealed
verdict mapping. No simulation, no tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ("baseline", "w_only", "a_only", "both")
CLASSES = ("E", "PV", "SST", "VIP")


def load(cell):
    matches = sorted((ROOT / "results").glob(f"whole_system_ei_interaction_*_{cell}.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one {cell} cell receipt, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def main() -> None:
    recs = {c: load(c) for c in CELLS}
    for c, r in recs.items():
        if r.get("cell") != c:
            raise ValueError(f"cell receipt mismatch: file {c} carries {r.get('cell')!r}")
        if r.get("verdict") != "EI_CELL_COMPLETE":
            raise ValueError(f"cell {c} did not complete: {r.get('verdict')!r}")
    rates = {
        c: {k: float(r["ei_regime"]["rates_full_resolution"][k]["full"]) for k in CLASSES}
        for c, r in recs.items()
    }

    def delta(which, cls):
        return rates[which][cls] - rates["baseline"][cls]

    dw = {c: delta("w_only", c) for c in CLASSES}
    da = {c: delta("a_only", c) for c in CLASSES}
    both = {c: delta("both", c) for c in CLASSES}
    interact = {c: round(both[c] - dw[c] - da[c], 4) for c in CLASSES}

    def viable(c):
        r = rates[c]
        return bool(
            (12.0 <= r["PV"] <= 25.0)
            and (8.0 <= r["VIP"] <= 15.0)
            and (6.0 <= r["SST"] <= 12.0)
            and (r["E"] < r["PV"])
        )

    v = {c: viable(c) for c in CELLS}
    if v["w_only"] and not v["a_only"]:
        verdict = "WEIGHT_SUFFICIENT"
    elif v["a_only"] and not v["w_only"]:
        verdict = "ADAPTATION_SUFFICIENT"
    elif v["both"] and not v["w_only"] and not v["a_only"]:
        verdict = "JOINT_REQUIRED"
    elif v["w_only"] and v["a_only"]:
        verdict = "MULTIPLE_SUFFICIENT"
    else:
        verdict = "EI_REGIME_STILL_UNRESOLVED"

    out = {
        "spec": "results/whole_system_ei_interaction_spec.json",
        "lineage": "SCI-EI-INTERACTION-1",
        "cells": {
            c: {
                "rates_full": {k: round(rates[c][k], 4) for k in CLASSES},
                "viable": v[c],
                "mode_present": recs[c].get("mode", {}).get("present"),
                "mode_period_ms": recs[c].get("mode", {}).get("period_ms"),
            }
            for c in CELLS
        },
        "decomposition": {
            "delta_W": {c: round(v, 4) for c, v in dw.items()},
            "delta_A": {c: round(v, 4) for c, v in da.items()},
            "delta_WxA": interact,
        },
        "verdict": verdict,
        "fail_boundary": (
            "viability pattern across the four cells: "
            + ", ".join(f"{c} {'viable' if v[c] else 'not viable'}" for c in CELLS)
        ),
    }
    (ROOT / "results" / "whole_system_ei_interaction.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    print("verdict:", verdict, "|", out["fail_boundary"])


if __name__ == "__main__":
    main()
