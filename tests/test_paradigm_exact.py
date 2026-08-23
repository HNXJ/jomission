"""Tests for PARADIGM_EXACT — authoritative timing, conditions, omission encoding."""

import jaxfne as jtfne
from jomission.paradigm.epochs import validate_epochs, P1_TO_D4_MS, FULL_TRIAL_MS, OMISSION_LOCAL_WINDOW_MS, OMISSION_SLOT_MS
from jomission.paradigm.conditions import validate_conditions, UNRESOLVED_FIELDS
from jomission.paradigm.spec import paradigm_exact_gate, JOMISSION_PARADIGM, SLOT_ONSET_MS, SLOT_DURATION_MS, condition_to_stimulus_schedule


def test_epochs_exact():
    v = validate_epochs()
    assert v["valid"], v["issues"]
    assert P1_TO_D4_MS == 4124.0
    assert FULL_TRIAL_MS == 4624.0


def test_conditions_exact():
    v = validate_conditions()
    assert v["valid"], v["issues"]
    assert len(JOMISSION_PARADIGM.conditions) == 12


def test_paradigm_exact_gate():
    gate = paradigm_exact_gate()
    assert gate["valid"], gate["issues"]


def test_omission_preserves_timing():
    for cond in JOMISSION_PARADIGM.conditions:
        if cond.omission_position is not None:
            # Check omitted slot onset preserved
            for ev in cond.events:
                if ev.label == cond.omission_position:
                    assert ev.is_omission is True
                    assert ev.stimulus is None
                    assert ev.onset_ms == SLOT_ONSET_MS[ev.label]
                if ev.label in ("p2", "p3", "p4") and ev.label != cond.omission_position:
                    assert ev.is_omission is False


def test_slot_durations():
    assert SLOT_DURATION_MS["p1"] == 531.0
    assert SLOT_DURATION_MS["d1"] == 500.0
    assert SLOT_DURATION_MS["p2"] == 531.0
    assert SLOT_DURATION_MS["fx"] == 500.0


def test_local_windows():
    assert OMISSION_LOCAL_WINDOW_MS == (-1000.0, 1000.0)
    assert OMISSION_SLOT_MS == (0.0, 531.0)


def test_unresolved_explicit():
    assert "UNRESOLVED" in JOMISSION_PARADIGM.metadata["unresolved"]["reward_schedule"]
    for k in UNRESOLVED_FIELDS:
        assert k in JOMISSION_PARADIGM.metadata["unresolved"]


def test_condition_to_stimulus_schedule_omission_zero():
    # Only p slots drive; omission zero but timing preserved
    axab = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AXAB"][0]
    aaab = [c for c in JOMISSION_PARADIGM.conditions if c.name == "AAAB"][0]
    sched_omit = condition_to_stimulus_schedule(axab, n_neurons=10, drive_amplitude=5.0)
    sched_intact = condition_to_stimulus_schedule(aaab, n_neurons=10, drive_amplitude=5.0)
    # 4 vs 3 sensory drives
    assert sum(1 for e in sched_intact.events if e["is_drive_event"]) == 4
    assert sum(1 for e in sched_omit.events if e["is_drive_event"]) == 3
    # Array zero check
    import jax.numpy as jnp
    dt = 0.5
    n_steps = int(4624 / dt)
    arr_omit = sched_omit.to_array(n_steps, dt)
    arr_intact = sched_intact.to_array(n_steps, dt)
    p2_idx = int(round(SLOT_ONSET_MS["p2"] / dt))
    p2_end = int(round((SLOT_ONSET_MS["p2"] + 531) / dt))
    assert float(jnp.sum(arr_omit[p2_idx:p2_end])) == 0.0
    assert float(jnp.sum(arr_intact[p2_idx:p2_end])) > 0
    # Timing preserved: p2 onset identical in both schedules
    assert sched_omit.events[3]["onset_ms"] == sched_intact.events[3]["onset_ms"] == 1031.0


def test_no_pooling_until_tested():
    # Ensure p2/p3/p4 omission groups are distinct
    from jomission.paradigm.conditions import OMISSION_POSITIONS
    assert set(OMISSION_POSITIONS["p2"]).isdisjoint(OMISSION_POSITIONS["p3"])
    assert set(OMISSION_POSITIONS["p3"]).isdisjoint(OMISSION_POSITIONS["p4"])
