"""Connectivity — explicit W_{(a,l,c)->(a',l',c')} as config tables.

FF: superficial E -> granular/middle (L2/3 E -> L4) etc.
FB: deep -> superficial (L5/6 -> L1/L2/3) etc.
Between-area only via inter_column_connectivity; within-area via within_gain.
Tables are declarative and replaceable via literature priors.
"""

from __future__ import annotations

from typing import Mapping

# Within-area gain (baseline recurrent)
WITHIN_GAIN_DEFAULT: float = 0.35

# Between-area probabilities (JaxFNE builder p_feedforward/p_feedback)
P_FEEDFORWARD_DEFAULT: float = 0.30
P_FEEDBACK_DEFAULT: float = 0.20

# Hierarchical order low->high
HIERARCHY: tuple[str, ...] = ("V1", "V4", "FEF", "PFC")

# Declarative FF/FB target layer mappings (for documentation and future explicit W)
# These mirror jaxfne.build_multi_area_columns defaults but made explicit here:
FF_SOURCE_LAYER: str = "L2/3"
FF_TARGET_LAYER: str = "L4"
FF_SOURCE_CELL: str = "E"

FB_SOURCE_LAYER: str = "L5"  # could be L5/L6; using L5 for now, L6 also included via builder
FB_TARGET_LAYERS: tuple[str, ...] = ("L1", "L2/3", "L5")

# Full W table placeholder — shape (source_pop -> target_pop) weight scale
# For now we expose scalar gains; future literature update can fill per-pop matrix.
# Key: (src_area, src_layer, src_ct, tgt_area, tgt_layer, tgt_ct) -> gain
CONNECTIVITY_TABLE: dict[tuple[str, str, str, str, str, str], float] = {}

# Delays — explicit per edge type (ms), proxy values, replaceable
DELAY_FF_MS: float = 8.0
DELAY_FB_MS: float = 12.0
DELAY_WITHIN_MS: float = 2.0
DELAY_LAMINAR_MS: dict[str, float] = {
    "within": DELAY_WITHIN_MS,
    "feedforward": DELAY_FF_MS,
    "feedback": DELAY_FB_MS,
}

def validate_connectivity() -> dict:
    issues: list[str] = []
    if not (0 < P_FEEDFORWARD_DEFAULT <= 1):
        issues.append("p_feedforward out of (0,1]")
    if not (0 < P_FEEDBACK_DEFAULT <= 1):
        issues.append("p_feedback out of (0,1]")
    if HIERARCHY != ("V1", "V4", "FEF", "PFC"):
        issues.append(f"hierarchy {HIERARCHY} != V1->V4->FEF->PFC")
    return {"valid": not issues, "issues": issues, "hierarchy": "->".join(HIERARCHY), "p_ff": P_FEEDFORWARD_DEFAULT, "p_fb": P_FEEDBACK_DEFAULT}
