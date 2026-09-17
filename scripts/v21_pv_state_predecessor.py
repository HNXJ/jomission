"""SCI-V21-PV-STATE-PREDECESSOR: read-only causal-predecessor analysis of the sealed PV trajectories.

    python scripts/v21_pv_state_predecessor.py          # derived JSON on stdout

No simulation. Inputs: results/v21_pv_stationarity.json (60 s, windows 1-11) and
results/v21_coupled_background.json (the same trajectories' windows 1-3, which carry more per-window
statistics). Active-PV trajectory: lambda_1_end_state. Collapsing-PV trajectories: fresh, lambda_0_end_state.
Windows are 5 s; window k covers 5k to 5(k+1) s after the switch to lambda 0.65625; chunk 0 is unrecorded.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("E", "PV", "SST", "VIP")
ACTIVE = "lambda_1_end_state"
COLLAPSING = ("fresh", "lambda_0_end_state")
TOL_ABS_HZ, TOL_REL = 1.0, 0.10  # sealed state-dependence tolerance (results/v21_pv_stationarity_spec.json)
REQUIRED = {
    "V_PV": "PV membrane potential",
    "u_PV": "PV recovery variable",
    "I_E->PV": "E-to-PV synaptic current",
    "I_PV->PV": "PV-to-PV synaptic current",
    "I_SST->PV": "SST-to-PV synaptic current",
    "synaptic_state": "any per-neuron or per-edge synaptic trace",
    "spike_raster": "per-neuron spike times (would allow sub-window timing of the divergence)",
}


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def leaf_paths(obj, prefix=""):
    if isinstance(obj, dict):
        return sorted({p for k, v in obj.items() for p in leaf_paths(v, f"{prefix}/{k}")})
    if isinstance(obj, list):
        return sorted({p for v in obj for p in leaf_paths(v, f"{prefix}[]")})
    return [prefix]


def first_divergence(a, b):
    """First 1-based window where |a - b| > max(1 Hz, 0.1 * a) (a = active trajectory), else None."""
    for k, (x, y) in enumerate(zip(a, b), start=1):
        if abs(x - y) > max(TOL_ABS_HZ, TOL_REL * x):
            return k
    return None


def derive():
    st = load("results/v21_pv_stationarity.json")
    cb = load("results/v21_coupled_background.json")
    paths = leaf_paths(st) + leaf_paths(cb)
    tokens = ("/v", "/u", "current", "syn", "raster", "spike", "I_")
    state_like = sorted({p for p in paths if any(t in p for t in tokens) and "sync" not in p and p != "/verdict"})
    tr = st["trajectories"]
    rates = {ic: {c: [w[c] for w in tr[ic]["window_rates_hz"]] for c in CLASSES} for ic in tr}
    divergence = {ic: {c: first_divergence(rates[ACTIVE][c], rates[ic][c]) for c in CLASSES} for ic in COLLAPSING}
    raw = {ic: {"active_fraction_PV": [[round(tr[ACTIVE]["reported_not_gating"]["active_fraction"][k]["PV"], 4),
                                        round(tr[ic]["reported_not_gating"]["active_fraction"][k]["PV"], 4)]
                                       for k in range(11)],
                "sync": [[tr[ACTIVE]["reported_not_gating"]["sync"][k], tr[ic]["reported_not_gating"]["sync"][k]]
                         for k in range(11)]} for ic in COLLAPSING}
    probes = {"V_PV": ("/v",), "u_PV": ("/u",), "I_E->PV": ("I_", "current"), "I_PV->PV": ("I_", "current"),
              "I_SST->PV": ("I_", "current"), "synaptic_state": ("syn",), "spike_raster": ("raster", "spike")}
    required_present = {k: any(t in p for p in state_like for t in probes[k]) for k in REQUIRED}
    pv_first = min(divergence[ic]["PV"] for ic in COLLAPSING)
    label = "MISSING_OBSERVABLE" if not all(required_present.values()) else "UNRESOLVED"
    return {
        "recorded_state_like_paths": state_like,
        "recorded_observables": "per 5 s window and class: rate, active fraction; per window: E sync; windows 1-3 only: ISI fraction, rate CV (results/v21_coupled_background.json)",
        "required_observables_present": required_present,
        "first_divergence_window_vs_active": divergence,
        "pv_rate_first_divergence_window": pv_first,
        "earliest_recorded_window": 1,
        "raw_pairs_active_vs_collapsing": raw,
        "label": label,
    }


if __name__ == "__main__":
    print(json.dumps(derive(), indent=1))
