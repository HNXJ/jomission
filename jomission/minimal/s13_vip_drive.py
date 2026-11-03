"""MIN-08: VIP effective drive 25.0 on the minimal column; score the VIP window from the run.

Spec: docs/minimal/s13_vip_drive_spec.json. The realization sets the emitter drive of the VIP rows to
VIP_DRIVE and of every other row to tonic + DC_CURRENT. Writes docs/minimal/s13_vip_drive_receipt.json.
Usage: python -m jomission.minimal.s13_vip_drive [--out PATH] [--seed S]
Exit 0 when the run completes; checks.VIP_window is the only scored check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jaxfne as jtfne
import numpy as np

from jomission.minimal.builder import CELL_TYPES, REPO_ROOT, build_minimal_model, minimal_runtime

RECEIPT_PATH = REPO_ROOT / "docs" / "minimal" / "s13_vip_drive_receipt.json"
VIP_DRIVE = 25.0  # spec drive.VIP (owner D1, 2026-11-02)
DC_CURRENT = 1.0  # spec drive.E_PV_SST, unchanged from docs/minimal/s13_receipt.json
WINDOW_MS = (200.0, 1000.0)  # spec criterion.window_ms (owner D3, 2026-11-02)


def realize_vip_drive(model, vip_drive: float = VIP_DRIVE, dc_current: float = DC_CURRENT):
    """Return the model with the VIP rows of the emitter drive set to vip_drive, others tonic + dc_current."""
    params = model.params["emitter"]
    labels = np.asarray(params.labels)
    tonic = np.asarray(params.drive, dtype=float)
    drive = np.where(labels == "VIP", vip_drive, tonic + dc_current)
    return model.with_emitter_parameters(drive_per_neuron=drive)


def vip_drive_receipt(
    n: int = 100,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    window_ms: tuple[float, float] = WINDOW_MS,
    vip_drive: float = VIP_DRIVE,
    dc_current: float = DC_CURRENT,
) -> dict:
    """Build, realize, simulate once, and return the receipt as a dict."""
    base = build_minimal_model(n=n, seed=seed)
    labels = np.asarray(base.params["emitter"].labels).tolist()
    tonic = np.asarray(base.params["emitter"].drive, dtype=float)
    model = realize_vip_drive(base, vip_drive=vip_drive, dc_current=dc_current)
    effective = np.asarray(model.params["emitter"].drive, dtype=float)
    runtime = minimal_runtime(model)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed, runtime=runtime)
    signals = model.simulate(sim)
    spikes = np.asarray(signals.spikes, dtype=float)
    n_steps = round(duration_ms / dt_ms)
    if spikes.shape[0] != n_steps:
        raise ValueError(f"spikes has {spikes.shape[0]} steps; expected {n_steps}")

    i0, i1 = round(window_ms[0] / dt_ms), round(window_ms[1] / dt_ms)
    window = spikes[i0:i1].sum(axis=0)
    window_s = (window_ms[1] - window_ms[0]) / 1000.0
    vip_idx = [i for i, lb in enumerate(labels) if lb == "VIP"]

    def first_spike_ms(i: int):
        hits = np.flatnonzero(spikes[:, i] > 0)
        return None if hits.size == 0 else round(float(hits[0] * dt_ms), 3)

    classes = {}
    for ct in CELL_TYPES:
        idx = [i for i, lb in enumerate(labels) if lb == ct]
        classes[ct] = {
            "n": len(idx),
            "tonic_mean": round(float(tonic[idx].mean()), 6),
            "effective_mean": round(float(effective[idx].mean()), 6),
            "window_spikes_min_per_cell": int(window[idx].min()),
            "silent_in_window": int((window[idx] < 1).sum()),
            "mean_Hz_window": round(float(window[idx].mean() / window_s), 3),
        }

    checks = {
        "VIP_window": bool(all(window[i] >= 1 for i in vip_idx)),
        "VIP_drive_realized": bool(np.allclose(effective[vip_idx], vip_drive)),
    }
    return {
        "schema": "jomission.minimal_s13_vip_drive_receipt.v1",
        "status": "MIN-08 receipt: checks.VIP_window is the scored check; all-class window counts are reported only",
        "spec": "docs/minimal/s13_vip_drive_spec.json",
        "n_neurons": n,
        "seed": seed,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "window_ms": list(window_ms),
        "vip_drive": vip_drive,
        "dc_current": dc_current,
        "noise_scale": None,
        "runtime_dtype": runtime.dtype,
        "connectivity_sha256": model.cfg.metadata["connectivity_sha256"],
        "classes": classes,
        "vip_cells": [
            {
                "index": i,
                "drive": round(float(effective[i]), 6),
                "window_spikes": int(window[i]),
                "first_spike_ms": first_spike_ms(i),
            }
            for i in vip_idx
        ],
        "all_class_each_cell_in_window": bool(all(window[i] >= 1 for i in range(len(labels)))),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    receipt = vip_drive_receipt(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["checks"]), file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
