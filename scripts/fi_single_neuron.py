"""Single-neuron f-I check per cell class (authorized 2026-11-04: single-neuron check only).

Izhikevich neuron, numpy, dt 0.1 ms, duration 2000 ms. Parameters are read from the engine's
preset table (jaxfne.emitters.IZHIKEVICH_CELL_TYPE_DEFAULTS); the 'jomission' variant applies the
repository's corrected b values (VIP b 0.20; SST b 0.21). Tonic drive is the preset drive; an
extra current I is swept. Output: docs/receipts/fi_single_neuron_v1.json (receipt, unsealed; kept out of results/ so the registry guard and tier-3 write rules do not apply).
Caveat: rheobase is a finite-duration estimate (2 s, start at v=c); a transient spike can appear below the steady threshold.
Not a network result. No parameter is fitted or tuned.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jaxfne.emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS as PRESETS

ROOT = Path(__file__).resolve().parents[1]
DT_MS = 0.1
DURATION_MS = 2000.0
EXTRA = np.round(np.arange(0.0, 20.0001, 0.25), 4)  # extra current grid
CLASSES = ["E", "PV", "SST", "VIP"]
JOMISSION_B = {"VIP": 0.20, "SST": 0.21}


def firing_rates(a, b, c, d, tonic, extra):
    """Vectorized Izhikevich over an array of extra currents; returns Hz per extra value."""
    n = extra.size
    v = np.full(n, c, dtype=float)
    u = b * v
    I = tonic + extra
    steps = int(round(DURATION_MS / DT_MS))
    spikes = np.zeros(n, dtype=np.int64)
    for _ in range(steps):
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = a * (b * v - u)
        v = v + DT_MS * dv
        u = u + DT_MS * du
        fired = v >= 30.0
        spikes += fired
        v = np.where(fired, c, v)
        u = np.where(fired, u + d, u)
    return spikes / (DURATION_MS / 1000.0)


def summarize(rates):
    nz = np.nonzero(rates > 0)[0]
    rheobase = float(EXTRA[nz[0]]) if nz.size else None
    if nz.size >= 2:
        slope = float(np.polyfit(EXTRA[nz], rates[nz], 1)[0])
    else:
        slope = None
    return {"rheobase_extra": rheobase, "gain_hz_per_unit_above_rheobase": slope}


def main() -> None:
    out = {"schema": "jomission.fi_single_neuron.v1", "dt_ms": DT_MS, "duration_ms": DURATION_MS,
           "extra_grid": EXTRA.tolist(), "classes": {}}
    for cls in CLASSES:
        p = PRESETS[cls]
        variants = {"engine_preset": dict(p)}
        if cls in JOMISSION_B:
            variants["jomission"] = dict(p, b=JOMISSION_B[cls])
        out["classes"][cls] = {}
        for name, q in variants.items():
            rates = firing_rates(q["a"], q["b"], q["c"], q["d"], float(q["drive"]), EXTRA)
            entry = {"a": q["a"], "b": q["b"], "c": q["c"], "d": q["d"], "tonic_drive": float(q["drive"]),
                     "rate_hz": [round(float(r), 3) for r in rates], **summarize(rates)}
            out["classes"][cls][name] = entry
    dest = ROOT / "docs" / "receipts" / "fi_single_neuron_v1.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for cls in CLASSES:
        for name, e in out["classes"][cls].items():
            print(cls, name, "rheobase_extra", e["rheobase_extra"], "gain", e["gain_hz_per_unit_above_rheobase"])
    print("wrote", dest)


if __name__ == "__main__":
    main()
