"""Stability criteria — frozen bounds before long-run.

All long exposure must stay within these or be excluded/failed.
"""

from __future__ import annotations

STABILITY_CRITERIA: dict = {
    "finite_state": {
        "V_m_min": -150.0,
        "V_m_max": 100.0,
        "V_m_mean_range": (-90.0, -50.0),
        "must_be_finite": True,
    },
    "population_rates": {
        "global_hz_mean_range": (1.0, 50.0),
        "per_area_hz_range": (0.5, 80.0),
        "per_classnote": "E/PV/SST/VIP rates must stay finite; exclusion if any area >100 Hz sustained",
    },
    "spike_bounds": {
        "spike_rate_hz_mean_max": 100.0,
        "spike_count_finite": True,
    },
    "field_bounds": {
        "lfp_proxy_finite": True,
        "csd_proxy_finite": True,
        "abs_mean_not_nan": True,
    },
    "h_state_domains": {
        "H_fast_tau_0_1_s": {"range": (0.0, 10.0), "must_be_finite": True},
        "H_medium_tau_1_s": {"range": (0.0, 10.0), "must_be_finite": True},
        "note": "H_k conceptual ranges; kernel H_i is bounded [H_min,H_max] when HDP enabled",
    },
    "hdp_theta_domains": {
        "theta_m_EI_bounds": (0.1, 5.0),
        "theta_eta_a_bounds": (0.25, 4.0),
        "must_be_finite": True,
        "note": "Only checked when HDP enabled; Θ bounded per channel",
    },
    "exclusion_rules": [
        "NaN or inf in V_m/spikes/sources/field => immediate exclusion",
        "sustained global rate >100 Hz for >5s => exclusion (epileptic-like)",
        "sustained V_m mean > -40 or < -80 for >10s => review",
        "checkpoint/restart mismatch beyond tolerance => exclusion (ALWAYS-31 failure)",
    ],
    "failure_rules": [
        "Shortened exposure (30-120s) must be finite and within bounds before authorizing ≥1000s",
        "Any H/HDP out-of-bounds => fail T2 gate",
    ],
}


def check_stability(metrics: dict) -> dict:
    issues: list[str] = []
    v_mean = metrics.get("v_mean")
    if v_mean is not None:
        lo, hi = STABILITY_CRITERIA["finite_state"]["V_m_mean_range"]
        if not (lo <= v_mean <= hi):
            issues.append(f"v_mean {v_mean} outside {lo,hi}")
    rate = metrics.get("rate_hz_mean")
    if rate is not None:
        lo, hi = STABILITY_CRITERIA["population_rates"]["global_hz_mean_range"]
        if not (lo <= rate <= hi):
            issues.append(f"global rate {rate} outside {lo,hi}")
    if not metrics.get("finite", True):
        issues.append("non-finite state")
    # Area rates
    for area, ar in (metrics.get("area_rates_hz") or {}).items():
        lo, hi = STABILITY_CRITERIA["population_rates"]["per_area_hz_range"]
        if not (lo <= ar <= hi):
            issues.append(f"area {area} rate {ar} outside {lo,hi}")
    return {"pass": not issues, "issues": issues, "criteria": STABILITY_CRITERIA}
