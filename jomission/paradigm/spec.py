"""Jomission paradigm spec — builds JaxFNE Paradigm from canonical epochs/conditions.

Does NOT modify JaxFNE's standard_visual_omission(); jomission owns its paradigm.
All omission trials preserve identical temporal geometry; only drive is zeroed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from jaxfne import Paradigm, ParadigmCondition, ParadigmEvent

from jomission.paradigm.epochs import (
    CANONICAL_EPOCHS,
    EPOCH_BY_NAME,
    EVENT_CODES,
    COMPARISON_CODE,
    COMPARISON_LABEL,
    TRIAL_ANALYSIS_WINDOWS,
    P1_TO_D4_MS,
    FULL_TRIAL_MS,
    validate_epochs,
)
from jomission.paradigm.conditions import (
    CANONICAL_CONDITIONS,
    CONDITION_FAMILIES,
    OMISSION_POSITIONS,
    STIMULUS_OMITTED,
    UNRESOLVED_FIELDS,
    validate_conditions,
)


# Map stimulus slot to epoch onset (p1-relative, fx at -500)
SLOT_ONSET_MS: dict[str, float] = {
    "fx": -500.0,
    "p1": 0.0,
    "d1": 531.0,
    "p2": 1031.0,
    "d2": 1562.0,
    "p3": 2062.0,
    "d3": 2593.0,
    "p4": 3093.0,
    "d4": 3624.0,
}
SLOT_DURATION_MS: dict[str, float] = {
    "fx": 500.0,
    "p1": 531.0,
    "d1": 500.0,
    "p2": 531.0,
    "d2": 500.0,
    "p3": 531.0,
    "d3": 500.0,
    "p4": 531.0,
    "d4": 500.0,
}


def _events_for_condition(name: str) -> tuple[ParadigmEvent, ...]:
    info = CANONICAL_CONDITIONS[name]
    seq = info["sequence"]  # p1..p4 identities
    omission = info["omission"]
    # slot -> stimulus identity
    slot_stim = {"p1": seq[0], "p2": seq[1], "p3": seq[2], "p4": seq[3]}
    events: list[ParadigmEvent] = []
    # fx always present — not a sensory drive event
    events.append(
        ParadigmEvent(
            label="fx",
            onset_ms=SLOT_ONSET_MS["fx"],
            code=EVENT_CODES["fx"],
            stimulus="fixation",
            is_omission=False,
        )
    )
    # p1..p4 and interleaving delays d1..d4
    for slot in ("p1", "d1", "p2", "d2", "p3", "d3", "p4", "d4"):
        onset = SLOT_ONSET_MS[slot]
        code = EVENT_CODES.get(slot, 0) if slot.startswith("p") else 0
        if slot.startswith("p"):
            stim = slot_stim[slot]
            is_om = stim == STIMULUS_OMITTED
            # Omission: stimulus slot is temporally present, stimulus=None, is_omission=True
            # Delay/fx/rw are NOT sensory drives even when not omission — ensure this is explicit
            events.append(
                ParadigmEvent(
                    label=slot,
                    onset_ms=onset,
                    code=code,
                    stimulus=None if is_om else stim,
                    is_omission=is_om,
                )
            )
        else:
            # delay epoch: visually empty, fixation/background only — never a sensory drive
            events.append(
                ParadigmEvent(label=slot, onset_ms=onset, code=code, stimulus="delay_fixation", is_omission=False)
            )
    # reward marker — time UNRESOLVED, place at d4 end + small buffer (not canonical)
    # We include the event but mark its timing as UNRESOLVED via metadata; not a sensory drive
    events.append(
        ParadigmEvent(label="rw", onset_ms=SLOT_ONSET_MS["d4"] + SLOT_DURATION_MS["d4"], code=EVENT_CODES["rw"], stimulus=None, is_omission=False)
    )
    return tuple(events)


def condition_to_stimulus_schedule(
    condition: ParadigmCondition,
    *,
    n_neurons: int,
    drive_amplitude: float = 6.0,
) -> "StimulusSchedule":
    """Convert a jomission ParadigmCondition to a JaxFNE StimulusSchedule with exact timing.

    Only p1-p4 slots with non-omitted stimulus inject drive; fx/delay/rw are always zero.
    Duration respects SLOT_DURATION_MS (531 for p, 500 for delays/fx).

    This bypasses jaxfne.stimulus_schedule helper which incorrectly treats delays as drives.
    """
    from jaxfne import StimulusSchedule  # local import to avoid cycle

    ev_dicts: list[dict[str, Any]] = []
    for ev in condition.events:
        dur = float(SLOT_DURATION_MS.get(ev.label, 531.0))
        # Only p slots are sensory drives
        is_sensory = ev.label in ("p1", "p2", "p3", "p4")
        is_drive = bool(is_sensory and not ev.is_omission and ev.stimulus is not None)
        ev_dicts.append(
            {
                "label": ev.label,
                "onset_ms": float(ev.onset_ms),
                "duration_ms": dur,
                "amplitude": float(drive_amplitude) if is_drive else 0.0,
                "is_drive_event": is_drive,
            }
        )
    return StimulusSchedule(events=tuple(ev_dicts), n_neurons=int(n_neurons))


def _condition_numbers() -> dict[str, tuple[int, ...]]:
    # Stable numbering: 1-12 in taxonomy order
    order = ["AAAB", "AXAB", "AAXB", "AAAX", "BBBA", "BXBA", "BBXA", "BBBX", "RRRR", "RXRR", "RRXR", "RRRX"]
    return {name: (i + 1,) for i, name in enumerate(order)}


def build_paradigm_conditions() -> tuple[ParadigmCondition, ...]:
    cn = _condition_numbers()
    conditions: list[ParadigmCondition] = []
    for name, info in CANONICAL_CONDITIONS.items():
        seq = info["sequence"]
        omission = info["omission"]
        # Use canonical seq with omitted token for documentation; JaxFNE expects is_omission flags
        conditions.append(
            ParadigmCondition(
                name=name,
                sequence=seq,
                omission_position=omission,
                probability=None,
                condition_numbers=cn[name],
                events=_events_for_condition(name),
            )
        )
    # Sort by condition_numbers for determinism
    conditions.sort(key=lambda c: c.condition_numbers[0])
    return tuple(conditions)


def jomission_paradigm() -> Paradigm:
    """Construct the authoritative jomission Paradigm (exact timing)."""
    conditions = build_paradigm_conditions()
    # Analysis windows include both trial-level and omission-local
    analysis_windows = {
        "baseline": (-500.0, 0.0),
        "omission_local": (-1000.0, 1000.0),
        "omission_baseline": (-250.0, -50.0),
        "omission_slot": (0.0, 531.0),
        "post_omission": (531.0, 1000.0),
        "p1_to_d4": (0.0, 4124.0),
        "full_trial": (-500.0, 4124.0),
    }
    return Paradigm(
        name="jomission_exact",
        conditions=conditions,
        comparison_code=COMPARISON_CODE,
        comparison_label=COMPARISON_LABEL,
        pre_stimulus_buffer_ms=500.0,
        analysis_windows=analysis_windows,
        event_codes=EVENT_CODES,
        metadata={
            "paradigm_exact": True,
            "p1_to_d4_ms": P1_TO_D4_MS,
            "full_trial_ms": FULL_TRIAL_MS,
            "n_conditions": len(conditions),
            "families": {k: list(v) for k, v in CONDITION_FAMILIES.items()},
            "omission_positions": {k: list(v) for k, v in OMISSION_POSITIONS.items()},
            "epochs": {e.name: {"start_ms": e.start_ms, "end_ms": e.end_ms, "duration_ms": e.duration_ms} for e in CANONICAL_EPOCHS},
            "slot_onsets": SLOT_ONSET_MS,
            "slot_durations": SLOT_DURATION_MS,
            "unresolved": UNRESOLVED_FIELDS,
            "omission_encoding": "temporal slot preserved, drive zeroed, not time-jump",
            "fixation_delay_encoding": "visually empty except fixation/background; surround: stimulus -> delay -> 531ms omitted slot -> delay",
            "source": "attached paradigm specification (authoritative for timing/condition taxonomy)",
        },
    )


# Frozen singleton for import convenience
JOMISSION_PARADIGM: Paradigm = jomission_paradigm()

CANONICAL_CONDITIONS = CANONICAL_CONDITIONS
CONDITION_FAMILIES = CONDITION_FAMILIES
OMISSION_POSITIONS = OMISSION_POSITIONS
CANONICAL_EPOCHS = CANONICAL_EPOCHS


def paradigm_exact_gate() -> dict[str, Any]:
    """Validate PARADIGM_EXACT for currently authoritative variables.

    Returns {valid, issues, checks}. UNRESOLVED fields are correctly None/unresolved, not failures.
    """
    issues: list[str] = []
    checks: dict[str, Any] = {}

    # Epoch checks
    epoch_res = validate_epochs()
    checks["epochs"] = epoch_res
    if not epoch_res["valid"]:
        issues.extend([f"epochs:{i}" for i in epoch_res["issues"]])

    # Condition checks
    cond_res = validate_conditions()
    checks["conditions"] = cond_res
    if not cond_res["valid"]:
        issues.extend([f"conditions:{i}" for i in cond_res["issues"]])

    # Paradigm object checks
    p = JOMISSION_PARADIGM
    checks["paradigm_name"] = p.name
    if p.name != "jomission_exact":
        issues.append(f"paradigm name {p.name} != jomission_exact")
    if len(p.conditions) != 12:
        issues.append(f"paradigm conditions {len(p.conditions)} != 12")
    if p.comparison_code != 101 or p.comparison_label != "p1":
        issues.append("comparison mismatch")
    if p.event_codes != EVENT_CODES:
        issues.append("event_codes mismatch")
    # Check each condition's events preserve temporal geometry
    for cond in p.conditions:
        ev_labels = [e.label for e in cond.events]
        expected = ["fx", "p1", "d1", "p2", "d2", "p3", "d3", "p4", "d4", "rw"]
        if ev_labels != expected:
            issues.append(f"{cond.name} event labels {ev_labels} != {expected}")
        # Check that omission trials have correct is_omission and preserved onset
        om_pos = cond.omission_position
        for ev in cond.events:
            if ev.label in ("p2", "p3", "p4"):
                should_be_om = ev.label == om_pos
                if ev.is_omission != should_be_om:
                    issues.append(f"{cond.name} {ev.label} is_omission {ev.is_omission} != {should_be_om}")
                # Onset must be canonical even when omitted
                if ev.onset_ms != SLOT_ONSET_MS[ev.label]:
                    issues.append(f"{cond.name} {ev.label} onset {ev.onset_ms} != {SLOT_ONSET_MS[ev.label]}")
                # Omitted stimulus must be None
                if should_be_om and ev.stimulus is not None:
                    issues.append(f"{cond.name} {ev.label} omitted but stimulus {ev.stimulus!r} not None")

    # Unresolved fields must remain explicitly unresolved
    unresolved = p.metadata.get("unresolved", {})
    for k in UNRESOLVED_FIELDS:
        if k not in unresolved or "UNRESOLVED" not in str(unresolved[k]):
            issues.append(f"unresolved field {k} not marked UNRESOLVED")

    # Timing invariants
    if p.metadata.get("p1_to_d4_ms") != 4124.0:
        issues.append("p1_to_d4_ms metadata != 4124")
    if p.metadata.get("full_trial_ms") != 4624.0:
        issues.append("full_trial_ms metadata != 4624")

    checks["n_conditions"] = len(p.conditions)
    checks["n_epochs"] = len(CANONICAL_EPOCHS)
    checks["unresolved_fields"] = list(UNRESOLVED_FIELDS.keys())

    return {"valid": not issues, "issues": issues, "checks": checks, "paradigm_exact": "exact for authoritative variables; unresolved fields explicit"}
