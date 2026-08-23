"""Factorial ablation matrix — 11-way counterfactual design.

Y = F(sequence, omission, hierarchy, FF/FB, H, HDP)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ablation:
    name: str
    rbd_h: bool
    hdp: bool | str
    hierarchy: bool | str
    ff_fb: str
    purpose: str


ABLATION_MATRIX: tuple[Ablation, ...] = (
    Ablation("full", True, True, True, "FF+FB", "candidate mechanism"),
    Ablation("no-H", False, "fixed", True, "FF+FB", "state-memory requirement"),
    Ablation("no-HDP", True, False, True, "FF+FB", "parameter-memory requirement"),
    Ablation("naive", True, True, True, "FF+FB", "before 1000s exposure"),
    Ablation("habituated", True, True, True, "FF+FB", "history effect; Δ_exposure"),
    Ablation("hierarchy_shuffle", True, True, False, "altered", "anatomical specificity"),
    Ablation("FB-off", True, True, True, "FF only", "feedback contribution"),
    Ablation("FF-off", True, True, True, "FB only", "feedforward contribution"),
    Ablation("delay_shuffle", True, True, True, "FF+FB+shuffled delays", "temporal structure"),
    Ablation("omission->stimulus", True, True, True, "FF+FB", "sensory control: replace omission with stimulus"),
    Ablation("random_sequence", True, True, True, "FF+FB", "sequence prediction control: RRRR instead of AAAB/BBBA"),
)

def validate_ablations() -> dict:
    return {"valid": len(ABLATION_MATRIX) == 11, "issues": [] if len(ABLATION_MATRIX) == 11 else ["ablation count !=11"], "n": len(ABLATION_MATRIX)}
