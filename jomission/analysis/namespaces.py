"""Evidence namespaces — mechanically separate pilot/developmental from canonical/confirmatory.

Canonical fields reject inheritance from pilot namespace. Clock semantics explicit:
p1→d4 = 4124 ms (Method Plan); scheduler trial interval = 4624 ms including fx −500→0.
"""

from __future__ import annotations

# Frozen clocks
P1_TO_D4_MS = 4124.0  # Method Plan experimental sequence
SCHEDULER_TRIAL_MS = 4624.0  # includes preceding fixation −500→0 ms


class NamespaceError(Exception):
    pass


def seal_pilot(result: dict) -> dict:
    """Tag a result as developmental/pilot evidence only."""
    out = dict(result)
    out["namespace"] = "pilot_developmental"
    out["canonical_confirmatory"] = "NOT_ESTABLISHED"
    out["promotable_to_canonical"] = False
    out["clock_note"] = {
        "p1_to_d4_ms": P1_TO_D4_MS,
        "scheduler_trial_ms": SCHEDULER_TRIAL_MS,
        "note": "scheduler interval includes 500 ms fixation; p1→d4 is 4124 ms",
    }
    return out


def seal_canonical(result: dict, *, computational_valid: bool) -> dict:
    """Tag a result as canonical/confirmatory. Requires independent execution evidence."""
    if "namespace" in result and result.get("namespace") == "pilot_developmental":
        raise NamespaceError("cannot promote pilot namespace into canonical")
    out = dict(result)
    out["namespace"] = "canonical_confirmatory"
    out["computational_valid"] = bool(computational_valid)
    if not computational_valid:
        for k in ("delta_exposure", "T1", "T2", "T3", "T4", "T5", "T6", "T7"):
            out[k] = "UNRESOLVED"
    return out


def promote_pilot_to_canonical() -> None:
    raise NamespaceError("pilot_developmental results can never be promoted to canonical_confirmatory")


def annotate_numerical_validity(result: dict, *, theta_min: float | None, theta_lo_bound: float,
                                rate_hz_mean: float | None, rate_range: tuple[float, float]) -> dict:
    """Record threshold violations explicitly; never silently relabel as bounded/stable."""
    issues: list[str] = []
    if theta_min is not None and theta_min < theta_lo_bound:
        issues.append(f"theta_boundary_violation: {theta_min} < {theta_lo_bound}")
    lo, hi = rate_range
    if rate_hz_mean is not None and not (lo <= rate_hz_mean <= hi):
        issues.append(f"rate_outside_frozen_criterion: {rate_hz_mean} outside [{lo},{hi}]")
    out = dict(result)
    out["numerical_validity"] = {
        "threshold_violations": issues,
        "stable_claim": len(issues) == 0,
        "interpretability": "scientifically_interpretable" if len(issues) == 0 else "requires_stop_semantics_review_per_frozen_rules",
    }
    return out
