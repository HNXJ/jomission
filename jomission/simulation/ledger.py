"""Production seed ledger, pairing scheme, checkpoint cadence, schedule.

Frozen before ≥1000s exposure. Seeds are ledger entries, not ad-hoc choices.
"""

from __future__ import annotations

# Pairing: AAAB/BBBA counterbalanced, naive vs habituated, intact vs omission
# Each replicate pairs identical seed across conditions where appropriate.

PRODUCTION_SEED_LEDGER: dict = {
    "seeds": list(range(0, 32)),  # 0-31 provisioned; actual production uses subset
    "replicates": 8,  # n_replicates for main exposure
    "pairing_scheme": "paired_by_replicate",
    "description": "seed = base + replicate_index for within-replicate pairing; omission vs intact share seed",
    "base_seed": 0,
    "habituated_vs_naive": "same seed set, different exposure history (Δ_exposure)",
    "intact_vs_omission": "same seed per replicate, condition varies",
}

CHECKPOINT_CADENCE: dict = {
    "every_n_trials": 10,
    "every_ms": 46240.0,  # 10 * 4624 ms
    "every_s": 46.24,
    "keep_last_n": 5,
    "format": "checkpoint_state .npz + .json + ContinuationState pytree",
    "verification": "uninterrupted vs restarted equivalence (ALWAYS-31/32) with tolerance rtol=1e-5 atol=1e-4 for V_m, exact for spikes",
}

CANONICAL_SCHEDULE = None  # lazy import to avoid cycle; use get_canonical_schedule()

def get_canonical_schedule(**kwargs):
    from jomission.simulation.schedule import canonical_schedule
    return canonical_schedule(**kwargs)

# Derived canonical schedule — single source of truth (trial_ms=4624, dt_ms=0.1)
# All phase durations/trials/steps/checkpoints derived; no manual mismatch.
_DERIVED = get_canonical_schedule(
    dt_ms=0.1,
    initialization_s=2.0,
    baseline_s=10.0,
    exposure_s=1200.0,
    recovery_s=30.0,
    testing_n_conditions=12,
    testing_n_reps=8,
    checkpoint_every_n_trials=10,
)

PRODUCTION_SCHEDULE: dict = {
    "derived": _DERIVED,
    # Backward-compatible shallow keys (derived, not manual)
    "exposure": {
        "duration_s": _DERIVED["phases"]["exposure"]["wall_s"],
        "requested_s": 1200.0,
        "actual_wall_s": _DERIVED["phases"]["exposure"]["wall_s"],
        "actual_wall_ms": _DERIVED["phases"]["exposure"]["wall_ms"],
        "n_trials": _DERIVED["phases"]["exposure"]["trials"],
        "n_trials_expected": _DERIVED["phases"]["exposure"]["trials"],
        "note": _DERIVED["phases"]["exposure"]["note"],
        "meets_ge_1000s": _DERIVED["phases"]["exposure"]["wall_s"] >= 1000.0,
    },
    "testing": {
        "n_trials": _DERIVED["phases"]["testing"]["trials"],
        "total_trials": _DERIVED["phases"]["testing"]["trials"],
        "duration_s": _DERIVED["phases"]["testing"]["wall_s"],
        "wall_s": _DERIVED["phases"]["testing"]["wall_s"],
        "wall_ms": _DERIVED["phases"]["testing"]["wall_ms"],
        "n_reps_per_condition": _DERIVED["phases"]["testing"]["reps"],
        "conditions": ["AAAB", "AXAB", "AAXB", "AAAX", "BBBA", "BXBA", "BBXA", "BBBX", "RRRR", "RXRR", "RRXR", "RRRX"],
        "note": _DERIVED["phases"]["testing"]["note"],
        "warning": "NOT 300s wall-clock; 96 full trials = 443.9s wall (derived). Recorded/test windows vs wall must be distinguished.",
    },
    "exposure_corrected": {
        "requested_s": 1200.0,
        "actual_wall_s": _DERIVED["phases"]["exposure"]["wall_s"],
        "actual_wall_ms": _DERIVED["phases"]["exposure"]["wall_ms"],
        "n_trials": _DERIVED["phases"]["exposure"]["trials"],
        "note": _DERIVED["phases"]["exposure"]["note"],
        "meets_ge_1000s": _DERIVED["phases"]["exposure"]["wall_s"] >= 1000.0,
    },
    "testing_corrected": {
        "n_trials": _DERIVED["phases"]["testing"]["trials"],
        "wall_s": _DERIVED["phases"]["testing"]["wall_s"],
        "wall_ms": _DERIVED["phases"]["testing"]["wall_ms"],
        "note": _DERIVED["phases"]["testing"]["note"],
        "warning": "NOT 300s wall-clock; 96 full trials = 443.9s wall (derived). Recorded/test windows vs wall must be distinguished.",
    },
    "total_derived": _DERIVED["total"],
    "legacy_manual_total_s_was_inconsistent": "1542.0 was manual estimate mixing 300s testing with 96 trials; corrected total is 1688.144s",
}

RESOURCE_ESTIMATE: dict = {
    "per_trial_ms": _DERIVED["trial_ms"],
    "per_trial_steps_dt0_1": int(_DERIVED["trial_ms"] / 0.1),
    "memory_per_400_neurons_4624ms_dt0_1": "~ V_m 400*46240*4B ≈ 74 MB + spikes/field",
    "full_exposure_steps": _DERIVED["phases"]["exposure"]["steps"],
    "testing_steps": _DERIVED["phases"]["testing"]["steps"],
    "total_steps": _DERIVED["total"]["steps"],
    "checkpoint_size_400_neurons": "~ a few MB per .npz + json",
    "checkpoint_every_ms": _DERIVED["checkpoint"]["every_ms"],
    "n_checkpoints_exposure": _DERIVED["checkpoint"]["n_checkpoints_exposure"],
    "note": "dt=0.5 reduces steps 5x (tested in milestone 01); production dt=0.1 is 5x cost — run on HPC with sharding if needed",
}
