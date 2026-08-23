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

PRODUCTION_SCHEDULE: dict = {
    "initialization": {"duration_s": 2.0, "description": "quiescent baseline, no structured input"},
    "baseline": {"duration_s": 10.0, "sequence": "RRRR interleaved, low rate"},
    "exposure": {"duration_s": 1200.0, "sequence": "balanced AAAB/BBBA", "n_trials_expected": 259, "note": "≥1000s requirement; 1200s chosen to cover 1000s + margin"},
    "testing": {
        "duration_s": 300.0,
        "conditions": ["AAAB", "AXAB", "AAXB", "AAAX", "BBBA", "BXBA", "BBXA", "BBBX", "RRRR", "RXRR", "RRXR", "RRRX"],
        "n_reps_per_condition": 8,
        "total_trials": 96,
    },
    "recovery": {"duration_s": 30.0, "sequence": "RRRR"},
    "total_estimated_s": 1542.0,
    "dt_ms": 0.1,
    "n_steps_total": 15420000,  # ~15M steps at 0.1ms
}

RESOURCE_ESTIMATE: dict = {
    "per_trial_ms": 4624.0,
    "per_trial_steps_dt0_1": 46240,
    "memory_per_400_neurons_4624ms_dt0_1": "~ V_m 400*46240*4B ≈ 74 MB + spikes/field",
    "full_exposure_1200s_dt0_1_steps": 12000000,
    "checkpoint_size_400_neurons": "~ a few MB per .npz + json",
    "note": "dt=0.5 reduces steps 5x (tested in milestone 01); production dt=0.1 is 5x cost — run on HPC with sharding if needed",
}
