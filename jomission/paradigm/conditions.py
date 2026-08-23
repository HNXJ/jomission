"""Canonical condition taxonomy — 12 conditions, omission positions, families.

Authoritative source: attached paradigm specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Stimulus identities — canonical labels (visual parameter values are UNRESOLVED)
STIMULUS_A: str = "stimulus_A"
STIMULUS_B: str = "stimulus_B"
STIMULUS_R: str = "random_stimulus"
STIMULUS_OMITTED: str = "stimulus_omitted"  # X = absence during 531-ms slot

# Condition -> sequence (p1,p2,p3,p4) and omission position
# None = intact, "p2"/"p3"/"p4" = expected-but-absent 531-ms slot.
CANONICAL_CONDITIONS: dict[str, dict] = {
    # A-family (AAAB)
    "AAAB": {"sequence": (STIMULUS_A, STIMULUS_A, STIMULUS_A, STIMULUS_B), "omission": None},
    "AXAB": {"sequence": (STIMULUS_A, STIMULUS_OMITTED, STIMULUS_A, STIMULUS_B), "omission": "p2"},
    "AAXB": {"sequence": (STIMULUS_A, STIMULUS_A, STIMULUS_OMITTED, STIMULUS_B), "omission": "p3"},
    "AAAX": {"sequence": (STIMULUS_A, STIMULUS_A, STIMULUS_A, STIMULUS_OMITTED), "omission": "p4"},
    # B-family (BBBA)
    "BBBA": {"sequence": (STIMULUS_B, STIMULUS_B, STIMULUS_B, STIMULUS_A), "omission": None},
    "BXBA": {"sequence": (STIMULUS_B, STIMULUS_OMITTED, STIMULUS_B, STIMULUS_A), "omission": "p2"},
    "BBXA": {"sequence": (STIMULUS_B, STIMULUS_B, STIMULUS_OMITTED, STIMULUS_A), "omission": "p3"},
    "BBBX": {"sequence": (STIMULUS_B, STIMULUS_B, STIMULUS_B, STIMULUS_OMITTED), "omission": "p4"},
    # R-family (RRRR random-control)
    "RRRR": {"sequence": (STIMULUS_R, STIMULUS_R, STIMULUS_R, STIMULUS_R), "omission": None},
    "RXRR": {"sequence": (STIMULUS_R, STIMULUS_OMITTED, STIMULUS_R, STIMULUS_R), "omission": "p2"},
    "RRXR": {"sequence": (STIMULUS_R, STIMULUS_R, STIMULUS_OMITTED, STIMULUS_R), "omission": "p3"},
    "RRRX": {"sequence": (STIMULUS_R, STIMULUS_R, STIMULUS_R, STIMULUS_OMITTED), "omission": "p4"},
}

CONDITION_FAMILIES: dict[str, tuple[str, ...]] = {
    "A": ("AAAB", "AXAB", "AAXB", "AAAX"),
    "B": ("BBBA", "BXBA", "BBXA", "BBBX"),
    "R": ("RRRR", "RXRR", "RRXR", "RRRX"),
}

OMISSION_POSITIONS: dict[str, tuple[str, ...]] = {
    "p2": ("AXAB", "BXBA", "RXRR"),
    "p3": ("AAXB", "BBXA", "RRXR"),
    "p4": ("AAAX", "BBBX", "RRRX"),
    "intact": ("AAAB", "BBBA", "RRRR"),
}

# Reverse map: condition -> family
CONDITION_TO_FAMILY: dict[str, str] = {c: fam for fam, cs in CONDITION_FAMILIES.items() for c in cs}

# For stimulus drive mapping: which positions carry expected drive?
# Even on omission trials, the temporal slot exists; only the drive amplitude is zero.
CONDITION_DRIVE_MAP: dict[str, dict[str, Optional[str]]] = {
    name: {f"p{i}": seq[i - 1] if seq[i - 1] != STIMULUS_OMITTED else None for i in (1, 2, 3, 4)}
    for name, seq in ((k, v["sequence"]) for k, v in CANONICAL_CONDITIONS.items())
}

# Explicit UNRESOLVED fields — never invented
UNRESOLVED_FIELDS: dict[str, str] = {
    "reward_schedule": "UNRESOLVED — reward timing/magnitude/contingency not yet canonical",
    "fixation_schedule": "UNRESOLVED — beyond fx epoch; acquisition/break/reward-gated timing not canonical",
    "stimulus_visual_params": "UNRESOLVED — grating/orientation/contrast/size/duration is canonical at slot level (531 ms) but visual feature values are placeholder",
    "iti_distribution": "UNRESOLVED — inter-trial interval distribution not canonical in this source",
    "block_structure": "UNRESOLVED — AAAB:BBBA frequency, block duration, randomization not canonical",
    "omission_probability": "UNRESOLVED — exact omission rates not canonical; condition taxonomy is canonical",
    "habituation_schedule": "UNRESOLVED — ≥1000 s exposure is requirement; trial order during exposure is UNRESOLVED",
}


def validate_conditions() -> dict:
    issues: list[str] = []
    if len(CANONICAL_CONDITIONS) != 12:
        issues.append(f"condition count {len(CANONICAL_CONDITIONS)} != 12")
    for fam, members in CONDITION_FAMILIES.items():
        if len(members) != 4:
            issues.append(f"family {fam} size {len(members)} != 4")
    # Check each omission grouping
    for pos, expected in [("p2", 3), ("p3", 3), ("p4", 3), ("intact", 3)]:
        if len(OMISSION_POSITIONS[pos]) != expected:
            issues.append(f"omission {pos} count {len(OMISSION_POSITIONS[pos])} != {expected}")
    # Check drive map
    for name, info in CANONICAL_CONDITIONS.items():
        seq = info["sequence"]
        om = info["omission"]
        if om is None and STIMULUS_OMITTED in seq:
            issues.append(f"{name} intact but contains omitted")
        if om is not None and seq[["p2", "p3", "p4"].index(om) + 1] != STIMULUS_OMITTED:
            issues.append(f"{name} omission {om} but sequence {seq} mismatch")
        if om == "p2" and name not in OMISSION_POSITIONS["p2"]:
            issues.append(f"{name} p2 omission not in p2 group")
    return {"valid": not issues, "issues": issues, "n_conditions": len(CANONICAL_CONDITIONS)}
