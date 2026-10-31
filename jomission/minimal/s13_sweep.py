"""MIN-06 option 2: S-13 sweep over an extra VIP-only drive on a fixed log-spaced grid.

Writes docs/minimal/s13_sweep_receipt.json. Each grid point is one 1000 ms run with the
owner's DC 1.0 on every neuron plus the grid value on VIP neurons. The decision rule is
declared in docs/minimal/s13_sweep_spec.json and is applied to the receipt, not to the
rates.

Usage: python -m jomission.minimal.s13_sweep [--out PATH]
Exit 0 when the sweep completes; the decision is in the receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jomission.minimal.builder import REPO_ROOT
from jomission.minimal.smoke import smoke_receipt

RECEIPT_PATH = REPO_ROOT / "docs" / "minimal" / "s13_sweep_receipt.json"
DC_CURRENT = 1.0
VIP_GRID = [0.1, 0.3162, 1.0, 3.162, 10.0]  # log-spaced, half-decade steps; fixed in the spec


def sweep(duration_ms: float = 1000.0) -> dict:
    points = []
    for vip in VIP_GRID:
        r = smoke_receipt(dc_current=DC_CURRENT, vip_dc_current=vip, duration_ms=duration_ms)
        points.append(
            {
                "vip_dc_current": vip,
                "S-13": r["checks"]["S-13"],
                "S-12": r["checks"]["S-12"],
                "S-14": r["checks"]["S-14"],
                "S-15": r["checks"]["S-15"],
                "silent_neurons": {ct: r["classes"][ct]["silent_neurons"] for ct in r["classes"]},
                "mean_Hz": {ct: r["classes"][ct]["mean_Hz"] for ct in r["classes"]},
            }
        )
    passing = [p["vip_dc_current"] for p in points if p["S-13"]]
    return {
        "schema": "jomission.minimal_s13_sweep.v1",
        "spec": "docs/minimal/s13_sweep_spec.json",
        "dc_current": DC_CURRENT,
        "vip_grid": VIP_GRID,
        "points": points,
        "decision": {
            "S-13": bool(passing),
            "scored_vip_dc_current": min(passing) if passing else None,
            "rule": "PASS if any grid point has S-13 true; the scored point is the smallest such value",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RECEIPT_PATH)
    args = parser.parse_args(argv)
    receipt = sweep()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["decision"]), file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
