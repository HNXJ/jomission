"""Canonical schedule algebra — single source for phase durations, trials, steps, checkpoints.

Derived programmatically; no separate manual entries to drift.
"""

from __future__ import annotations

from jomission.paradigm.epochs import FULL_TRIAL_MS


def canonical_schedule(
    *,
    dt_ms: float = 0.1,
    initialization_s: float = 2.0,
    baseline_s: float = 10.0,
    exposure_s: float = 1200.0,
    recovery_s: float = 30.0,
    testing_n_conditions: int = 12,
    testing_n_reps: int = 8,
    checkpoint_every_n_trials: int = 10,
) -> dict:
    # Trial geometry
    trial_ms = float(FULL_TRIAL_MS)  # 4624.0
    # Exposure trials: ceil so that ≥ exposure_s is satisfied
    import math
    exposure_n_trials = math.ceil(exposure_s * 1000 / trial_ms)
    exposure_wall_ms = exposure_n_trials * trial_ms
    exposure_wall_s = exposure_wall_ms / 1000

    testing_n_trials = testing_n_conditions * testing_n_reps  # 96
    testing_wall_ms = testing_n_trials * trial_ms
    testing_wall_s = testing_wall_ms / 1000

    # Steps
    def steps(ms: float) -> int:
        return int(round(ms / dt_ms))

    init_steps = steps(initialization_s * 1000)
    base_steps = steps(baseline_s * 1000)
    exp_steps = steps(exposure_wall_ms)
    test_steps = steps(testing_wall_ms)
    rec_steps = steps(recovery_s * 1000)
    total_ms = initialization_s * 1000 + baseline_s * 1000 + exposure_wall_ms + testing_wall_ms + recovery_s * 1000
    total_steps = steps(total_ms)

    # Checkpoint locations (exposure only, where H/HDP matters)
    ckpts_exposure = list(range(checkpoint_every_n_trials, exposure_n_trials + 1, checkpoint_every_n_trials))

    return {
        "trial_ms": trial_ms,
        "dt_ms": float(dt_ms),
        "phases": {
            "initialization": {"wall_s": float(initialization_s), "wall_ms": float(initialization_s * 1000), "steps": init_steps, "trials": None},
            "baseline": {"wall_s": float(baseline_s), "wall_ms": float(baseline_s * 1000), "steps": base_steps, "trials": None},
            "exposure": {"requested_s": float(exposure_s), "wall_s": float(exposure_wall_s), "wall_ms": float(exposure_wall_ms), "steps": exp_steps, "trials": int(exposure_n_trials), "note": f"ceil({exposure_s}s / {trial_ms}ms) = {exposure_n_trials} trials = {exposure_wall_s}s"},
            "testing": {"wall_s": float(testing_wall_s), "wall_ms": float(testing_wall_ms), "steps": test_steps, "trials": int(testing_n_trials), "conditions": int(testing_n_conditions), "reps": int(testing_n_reps), "note": f"{testing_n_conditions}×{testing_n_reps}={testing_n_trials} trials @ {trial_ms}ms = {testing_wall_s}s"},
            "recovery": {"wall_s": float(recovery_s), "wall_ms": float(recovery_s * 1000), "steps": rec_steps, "trials": None},
        },
        "total": {"wall_ms": float(total_ms), "wall_s": float(total_ms / 1000), "steps": int(total_steps), "n_400_neurons_per_step_bytes": 400 * 4},
        "checkpoint": {
            "every_n_trials": int(checkpoint_every_n_trials),
            "every_ms": checkpoint_every_n_trials * trial_ms,
            "exposure_checkpoint_trial_numbers": ckpts_exposure,
            "n_checkpoints_exposure": len(ckpts_exposure),
        },
        "derived_from": "trial_ms=4624.0 (fx -500 to d4 4124) and dt_ms; no manual duration/trial mismatch allowed",
    }
