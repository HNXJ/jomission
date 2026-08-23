"""Canonical epoch timing — immutable, authoritative.

Source: attached paradigm specification (4.124 s p1→d4 sequence).
All values are hard-frozen; any drift is a PARADIGM_EXACT failure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Epoch:
    name: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


# p1-relative timing. fx precedes p1 by 500 ms.
CANONICAL_EPOCHS: tuple[Epoch, ...] = (
    Epoch("fx", -500.0, 0.0),
    Epoch("p1", 0.0, 531.0),
    Epoch("d1", 531.0, 1031.0),
    Epoch("p2", 1031.0, 1562.0),
    Epoch("d2", 1562.0, 2062.0),
    Epoch("p3", 2062.0, 2593.0),
    Epoch("d3", 2593.0, 3093.0),
    Epoch("p4", 3093.0, 3624.0),
    Epoch("d4", 3624.0, 4124.0),
)

# Quick lookup
EPOCH_BY_NAME: dict[str, Epoch] = {e.name: e for e in CANONICAL_EPOCHS}

# Derived constants
FIXATION_DURATION_MS: float = 500.0
STIMULUS_SLOT_MS: float = 531.0
DELAY_SLOT_MS: float = 500.0
P1_TO_D4_MS: float = 4124.0  # p1 onset to d4 end
FULL_TRIAL_MS: float = 4624.0  # fx onset (-500) to d4 end
TRIALS_PER_1000S: float = 1000_000.0 / P1_TO_D4_MS  # ~242.48 if ignoring ITI

# Omission-local windows (reindexed t=0 at expected omission onset)
OMISSION_LOCAL_WINDOW_MS: tuple[float, float] = (-1000.0, 1000.0)
OMISSION_LOCAL_BASELINE_MS: tuple[float, float] = (-250.0, -50.0)
OMISSION_SLOT_MS: tuple[float, float] = (0.0, 531.0)
POST_OMISSION_WINDOW_MS: tuple[float, float] = (531.0, 1000.0)

# Analysis windows (trial-level, p1-relative for full trial)
TRIAL_ANALYSIS_WINDOWS: dict[str, tuple[float, float]] = {
    "baseline": (-500.0, 0.0),  # fx
    "omission_local": OMISSION_LOCAL_WINDOW_MS,
    "omission_baseline": OMISSION_LOCAL_BASELINE_MS,
    "omission_slot": OMISSION_SLOT_MS,
    "post_omission": POST_OMISSION_WINDOW_MS,
}

# Event codes — stable, used for comparison alignment
EVENT_CODES: dict[str, int] = {
    "fx": 10,
    "p1": 101,
    "p2": 103,
    "p3": 105,
    "p4": 107,
    "rw": 96,  # reward marker (UNRESOLVED schedule, code reserved)
}

COMPARISON_CODE: int = EVENT_CODES["p1"]
COMPARISON_LABEL: str = "p1"

# Validation helper
def validate_epochs() -> dict:
    issues: list[str] = []
    # Check contiguous, no gaps/overlaps, correct durations
    expected = [
        ("fx", -500.0, 0.0),
        ("p1", 0.0, 531.0),
        ("d1", 531.0, 1031.0),
        ("p2", 1031.0, 1562.0),
        ("d2", 1562.0, 2062.0),
        ("p3", 2062.0, 2593.0),
        ("d3", 2593.0, 3093.0),
        ("p4", 3093.0, 3624.0),
        ("d4", 3624.0, 4124.0),
    ]
    if len(CANONICAL_EPOCHS) != len(expected):
        issues.append(f"epoch count {len(CANONICAL_EPOCHS)} != {len(expected)}")
    for epoch, (name, s, e) in zip(CANONICAL_EPOCHS, expected):
        if epoch.name != name or epoch.start_ms != s or epoch.end_ms != e:
            issues.append(f"epoch mismatch {epoch} != {(name,s,e)}")
        if epoch.duration_ms not in (500.0, 531.0):
            issues.append(f"unexpected duration {epoch.name}: {epoch.duration_ms}")
    # Check p1->d4 = 4124
    if P1_TO_D4_MS != 4124.0:
        issues.append(f"P1_TO_D4 {P1_TO_D4_MS} != 4124")
    if FULL_TRIAL_MS != 4624.0:
        issues.append(f"FULL_TRIAL {FULL_TRIAL_MS} != 4624")
    # Check stimulus slots all 531
    for name in ("p1", "p2", "p3", "p4"):
        if EPOCH_BY_NAME[name].duration_ms != 531.0:
            issues.append(f"stim slot {name} not 531")
    for name in ("fx", "d1", "d2", "d3", "d4"):
        if EPOCH_BY_NAME[name].duration_ms != 500.0:
            issues.append(f"delay/fx slot {name} not 500")
    return {"valid": not issues, "issues": issues, "n_epochs": len(CANONICAL_EPOCHS)}
