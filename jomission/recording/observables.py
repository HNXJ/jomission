"""Compiled experiment observables union — must be owned by recorder before expensive compute.

Union of all observables needed for RF validation, Δ_exposure, T1–T7, plasticity-timescale analysis and mechanistic interpretation.
"""

REQUIRED_OBSERVABLES = {
    "visual_field": "visual_field[trial,time,32,32] — 32×32 pixel lattice input per trial slot (A/B/R patterns)",
    "external_drive": "external_drive[trial,time,unit] — post-RF drive per unit (target_indices × amplitude)",
    "RF_operator": "RF_operator[unit,1024] — per V1 unit Gaussian weights, L1-normalized",
    "spikes": "spikes[trial,unit,time/event] — per-unit spike raster",
    "rate": "rate[trial,area,time] — ms area population rates",
    "metadata": "area/layer/class/unit metadata — neuron_metadata per unit",
    "field_proxy": "field_proxy[trial,area_attribution,contact,time] where scientifically valid — area-attributed contributions at declared probe geometry, proxy_readout",
    "H_t": "H(t) — per-neuron hidden state trajectory, dense early-time sampling to resolve tau_Theta ~2.5s effective",
    "Theta_t": "Theta(t) — HDP w trajectory, dense early-time sampling, bounds [w_floor,w_ceiling]",
    "event_ledger": "event ledger — trial/condition/onset/duration per slot, omission zero-drive preserved",
    "continuation_state": "continuation/checkpoint state — (X,H,Θ,D,RNG,cursor) per trial boundary"
}

# Recorder ownership check: must be able to produce all before long runs
def assert_recorder_owns(available: set[str]) -> list[str]:
    missing = [k for k in REQUIRED_OBSERVABLES if k not in available]
    return missing
