"""Smoke run of the minimal typed column: finiteness, class rates, HDP bounds. No scientific verdict.

Writes docs/minimal/smoke_receipt.json. Scoreboard items S-12 to S-15 read from this receipt.
Usage: python -m jomission.minimal.smoke [--out PATH] [--n N] [--seed S] [--duration-ms MS]
Exit 0 when the run completes; each check is recorded as true or false in the receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jaxfne as jtfne
import numpy as np

from jomission.minimal.builder import (
    CELL_TYPES,
    REPO_ROOT,
    build_minimal_model,
    load_connectivity,
    minimal_runtime,
)

RECEIPT_PATH = REPO_ROOT / "docs" / "minimal" / "smoke_receipt.json"
# Reported, not gated (plan decision 3). Source: scripts/ei_replay_058_compare.py.
BANDS_HZ = {"PV": (12.0, 25.0), "VIP": (8.0, 15.0), "SST": (6.0, 12.0)}
RUNAWAY_HZ = 100.0  # S-14 ceiling; chosen after the first probe run, recorded as such in the receipt
H_BOUNDS = (0.1, 10.0)  # H_min and H_max defaults of the JaxFNE HDP kernel (_hdp_kernel_kwargs)


def smoke_receipt(n: int = 100, seed: int = 0, duration_ms: float = 1000.0, dt_ms: float = 0.1) -> dict:
    """Run the minimal model once and return the receipt as a dict."""
    model = build_minimal_model(n=n, seed=seed)
    labels = np.asarray(model.params["emitter"].labels).tolist()
    sim = jtfne.simulation(
        duration_ms=duration_ms, dt_ms=dt_ms, seed=seed, runtime=minimal_runtime(model)
    )
    signals = model.simulate(sim)
    diag = model.last_hdp_diagnostics()
    spikes = np.asarray(signals.spikes, dtype=float)
    vm = np.asarray(signals.V_m, dtype=float)
    H = np.asarray(diag["H_trace"], dtype=float)
    H_final = np.asarray(diag["H_final"], dtype=float)
    seconds = duration_ms / 1000.0

    classes = {}
    for ct in CELL_TYPES:
        idx = [i for i, lb in enumerate(labels) if lb == ct]
        rate = spikes[:, idx].sum(axis=0) / seconds
        band = BANDS_HZ.get(ct)
        classes[ct] = {
            "n": len(idx),
            "mean_Hz": round(float(rate.mean()), 3),
            "max_Hz": round(float(rate.max()), 3),
            "silent_neurons": int((rate == 0).sum()),
            "band_Hz": list(band) if band else None,
            "mean_in_band": None if band is None else bool(band[0] <= rate.mean() <= band[1]),
        }

    checks = {
        "S-12": bool(np.isfinite(spikes).all() and np.isfinite(vm).all() and np.isfinite(H).all()),
        "S-13": all(classes[ct]["mean_Hz"] > 0 and classes[ct]["silent_neurons"] == 0 for ct in CELL_TYPES),
        "S-14": classes["E"]["max_Hz"] < RUNAWAY_HZ,
        "S-15": bool(H.min() >= H_BOUNDS[0] and H.max() <= H_BOUNDS[1]),
    }
    return {
        "schema": "jomission.minimal_smoke.v1",
        "status": "smoke run; no scientific verdict; checks only",
        "n_neurons": n,
        "seed": seed,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "connectivity_sha256": model.cfg.metadata["connectivity_sha256"],
        "rules": [r["name"] for r in load_connectivity()["rules"]],
        "classes": classes,
        "E_mean_below_PV_mean": bool(classes["E"]["mean_Hz"] < classes["PV"]["mean_Hz"]),
        "H": {
            "bounds": list(H_BOUNDS),
            "trace_min": round(float(H.min()), 6),
            "trace_max": round(float(H.max()), 6),
            "final_min": round(float(H_final.min()), 6),
            "final_max": round(float(H_final.max()), 6),
            "per_neuron_shape": list(H_final.shape),
        },
        "nonfinite": {"spikes": int((~np.isfinite(spikes)).sum()), "V_m": int((~np.isfinite(vm)).sum())},
        "checks": checks,
        "notes": [
            "S-13 reads 'none silent' per neuron; a 1 s window at about 1 Hz gives about one spike, so a zero count is a sampling fact, not evidence of a dead cell",
            "E drives PV, SST, VIP and E through the four E rules cited from Campagnola 2022 Fig. 2B and 3D (S-10)",
            "S-14 ceiling is set after the first probe run, not pre-registered",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-ms", type=float, default=1000.0)
    args = parser.parse_args(argv)
    receipt = smoke_receipt(n=args.n, seed=args.seed, duration_ms=args.duration_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["checks"]), file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
