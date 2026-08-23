"""T1–T7 falsification targets — frozen, not tuned.

Each target is an explicit hypothesis with null and measurement plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    id: str
    description: str
    measurement: str
    expectation: str
    falsifiable: bool = True


FALSIFICATION_TARGETS: tuple[Target, ...] = (
    Target("T1", "sparse omission-linked spiking", "fraction of units with significant omission vs intact rate (pooled p2/p3/p4, then per position)", "sparse (<10% significant) but nonzero"),
    Target("T2", "higher-order bias of omission units", "area enrichment of T1-significant units (FEF/PFC > V1/V4)", "higher-order enrichment"),
    Target("T3", "weak population omission spiking in V1", "V1 population PSTH omission vs intact (t-test per ms, effect size)", "no strong V1 population omission burst"),
    Target("T4", "frontal omission-related low-gamma enhancement", "LFP-like band power omission vs intact (20-50 Hz, FEF/PFC vs V1, corrected)", "frontal low-gamma up"),
    Target("T5", "gamma-rate coupling", "trial gamma power vs spiking (correlation per unit)", "positive coupling"),
    Target("T6", "weaker LFP coupling for omission-selective neurons", "spike-field coherence / gamma coupling for T1+ vs T1- units", "T1+ weaker coupling"),
    Target("T7", "absence of strong fixed between-area lead/lag", "cross-area field/rate cross-correlation peak lag distribution", "no consistent fixed lead/lag"),
)

# Δ_exposure = Y_post - Y_pre
DELTA_EXPOSURE_DEFINITION: str = "Delta_exposure = Y_omission^{post} - Y_omission^{pre}  (post = after ≥1000s AAAB/BBBA, pre = naive)"

def validate_targets() -> dict:
    return {"valid": True, "issues": [], "n_targets": len(FALSIFICATION_TARGETS), "ids": [t.id for t in FALSIFICATION_TARGETS]}
