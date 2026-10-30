"""Check the SCI-EI-REPLAY-058 V1 and V2 probes against the sealed probes, spike for spike.

Usage (from the repo root): python scripts/ei_replay_058_probe_check.py

Compares results/viz/SCI-EI-REPLAY-058/probe_initial.npz with the sealed
results/viz/SCI-EI-REGIME-1/probe_initial.npz, and probe_final.npz with
results/viz/SCI-EI-REGIME-1/probe_final.npz. Writes results/ei_replay_058/probe_check.json:
spike totals, the fraction of identical time-by-neuron entries, and the first time step
at which any neuron differs. Not a gate; a locator for where the 0.5.8 shift begins.
Exit 0 on success; exit 2 when a probe is missing.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAIRS = {
    "initial": (
        "results/viz/SCI-EI-REGIME-1/probe_initial.npz",
        "results/viz/SCI-EI-REPLAY-058/probe_initial.npz",
    ),
    "final": (
        "results/viz/SCI-EI-REGIME-1/probe_final.npz",
        "results/viz/SCI-EI-REPLAY-058/probe_final.npz",
    ),
}
OUT = ROOT / "results" / "ei_replay_058" / "probe_check.json"


def load_spikes(rel):
    path = ROOT / rel
    if not path.exists():
        sys.exit(f"missing probe {rel}")
    return np.load(path)["spikes"].astype(np.int64)


def check(sealed, replay):
    n = min(sealed.shape[0], replay.shape[0])
    a, b = sealed[:n], replay[:n]
    differ = a != b
    first = int(np.argmax(differ.any(axis=1))) if differ.any() else None
    return {
        "shape_sealed": list(sealed.shape),
        "shape_replay": list(replay.shape),
        "spikes_sealed": int(sealed.sum()),
        "spikes_replay": int(replay.sum()),
        "identical_fraction": float(1.0 - differ.mean()),
        "first_differing_ms": first,
        "neurons_with_any_difference": int(differ.any(axis=0).sum()),
    }


def main() -> int:
    out = {}
    for name, (sealed_rel, replay_rel) in PAIRS.items():
        out[name] = check(load_spikes(sealed_rel), load_spikes(replay_rel))
        out[name]["sealed"] = sealed_rel
        out[name]["replay"] = replay_rel
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for name, v in out.items():
        print(
            f"{name}: spikes sealed {v['spikes_sealed']} replay {v['spikes_replay']}, "
            f"identical {v['identical_fraction']:.6f}, first differing ms {v['first_differing_ms']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
