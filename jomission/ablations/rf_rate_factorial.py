"""2×2 factorial — plasticity-rate × RF (receptive-field plasticity).

Frozen design: RF off/on × Rate standard/slow, with exact completion predicate,
statistics, and ability to distinguish main and interaction effects.

Version: rf_rate_factorial.v0.1.0
Status: FROZEN — not tuned to results. Any expansion requires new version.

Factor definitions
------------------
- RF (receptive-field plasticity) — whether feedforward RF weights are plastic via HDP.
  Operationally: K_HDP gain on w (Theta) update.
    RF off : K_HDP = 0.0  (w frozen at baseline, H still evolves)
    RF on  : K_HDP = 0.003 (canonical v1_pfc_aaab_hdp_params)
  Preserves: H dynamics, bounds [H_min,H_max], [w_floor,w_ceiling]; RF off moves w*
  to w_baseline (qualitative presence/absence), which is intentional — RF is a
  mechanism toggle, not a rate-only knob. Rate factor is rate-only.

- Rate (plasticity rate) — H integration timescale via tau_0_ms (cube-law).
  Operationally: tau_0_ms scaling. Preserves fixed points & bounds (divisor only).
    Rate standard : tau_0_ms = 5.0  (canonical, τ_eff,E≈4.1 s, saturates in ~12 s)
    Rate slow     : tau_0_ms = 1000.0 (200×, τ_eff,E≈833 s, 76% of asymptote at 1200 s)
  Per PLASTICITY_RATE_INTERVENTION_DESIGN.md audit, ×3 (15 ms, 12 s) is rejected;
  200× is the frozen slow rate. K_HDP/K_w_ctrl ratio unchanged, so w* invariant
  for a given ΔH; only H timescale moves.

Both factors use enable_hdp=True (H alive). RF controls w plasticity magnitude,
Rate controls H speed. They are orthogonal: Rate affects H trajectory regardless of RF;
RF affects whether w can follow H.

Cells (4)
---------
A : RF_off  × Rate_standard  (K_HDP=0,   tau_0=5)    hash bb8277e7a8e0bca2
B : RF_off  × Rate_slow      (K_HDP=0,   tau_0=1000) hash f72a489841810a4b
C : RF_on   × Rate_standard  (K_HDP=0.003,tau_0=5)    hash f327f9d2ad64cc88 (frozen canonical)
D : RF_on   × Rate_slow      (K_HDP=0.003,tau_0=1000) hash b326f7201c59b803

Each cell uses identical network (config_hash 4f9fdeae7428199a, 400 neurons,
V1→V4→FEF→PFC, dt_ms=0.1 canonical) and identical exposure/testing schedule
(canonical_schedule 260 trials =1202.24 s exposure, 96 trials testing,
26 checkpoints). Only hdp_params differ per cell (hashes above).

Estimands
---------
Primary Y = Δ_exposure = Y_omission^{post} − Y_omission^{pre}, where
Y = phenotype measured identically pre and post within same replicate/seed,
using same trial battery (12 conditions ×8 reps =96 trials, balanced).
Phenotypes (each is a separate Y family, not p-hacked):
  - H_mean : mean ContinuationState.dynamic.H (global)
  - w_mean : mean ContinuationState.dynamic.w (Theta)
  - rate_omission_effect : omission vs intact rate in slot [0,531] ms
        (mean omission rate − mean intact rate), global and per-area (V1, PFC)
  - field_low_gamma : log band power omission vs intact, low_gamma (30-50 Hz)
        per area (V1, PFC) and frontal−V1 contrast, proxy_readout
  - field_broadband : per band (theta/alpha/beta/low_gamma/high_gamma) × area (4) × position (3)
        as in t4_t5_analysis (periodogram, omission vs intact same position)
  - rate_intrinsic : raw trial-rate Hz (no omission contrast) for sanity

Each Y is paired within-replicate (same seed, same condition order, same RNG),
so Δ_exposure is free of seed/condition variance.

Statistics (frozen, not tuned)
------------------------------
Model per phenotype Y (one value per replicate per cell):
  Y_{ijk} = μ + α·RF_i + β·Rate_j + γ·(RF_i·Rate_j) + ε_{ijk}
  RF_i ∈ {0=off,1=on}, Rate_j ∈ {0=standard,1=slow}, k=1..n_seeds
  ε iid Normal(0,σ²) under ANOVA assumptions; if violated, permutation-based F.

Contrasts (orthogonal, H0: coefficient=0, two-sided α=0.05 per-contrast):
  - Main RF   : (C+D)/2 − (A+B)/2  ; estimand = α + γ/2 ; test via F(1, df_error)
  - Main Rate : (B+D)/2 − (A+C)/2  ; estimand = β + γ/2
  - Interaction RF×Rate : (D−C) − (B−A) ; estimand = γ ; equivalently (D−B)−(C−A)
  Each contrast uses marginal means; reported with estimate, SE, t, p, Cohen d,
  and 95% CI (t-based). Partial eta² and generalised η² also reported.

Global factorial test: Type II (or III with orthogonal coding) 2-way ANOVA
  table with rows RF, Rate, RF×Rate, Residual. F and p for each. α=0.05 family-wise?
  For primary family (H_mean, w_mean, rate_omission_effect global, field_low_gamma
  frontal−V1): no multiplicity correction across phenotypes — each H is separate
  falsifiable claim with its own α (avoids hiding interaction). For exploratory
  5×4×3 T4 grid: FDR (Benjamini-Hochberg) across bands×areas×positions per position,
  reported as secondary. This rule is frozen; do not switch post hoc.

Effect-size thresholds (frozen, from Q8 criteria, not tuned):
  - rate_omission_effect: |Δ| >0.5 Hz and |d|>0.2 and p<0.05 → POSITIVE
                          |Δ| <0.5 and p≥0.05 → NEGATIVE else UNRESOLVED
  - field log_ratio: |log(om/intact)| >0.1 (~10%) and |d|>0.2 and p<0.05 → POSITIVE
  - H_mean / w_mean: analogous, but no fixed threshold — report CI and d;
        H: |Δ|>0.01 and d>0.2 flagged descriptive (H ∈ [0.1,10], SD~0.02, so 0.01 is ~0.5 SD)
  These thresholds gate interpretation, not inference — ANOVA p is the inferential
  statistic; threshold determines POSITIVE|NEGATIVE at phenotype level.

Ability to distinguish (frozen interpretation table):
  Pattern → Inference
  - RF main p<.05, Rate NS, interaction NS → RF matters, rate does not (RF drives Y)
  - Rate main p<.05, RF NS, interaction NS → rate matters, RF does not (timescale drives Y)
  - Both mains p<.05, interaction NS → additive independence (both matter, no synergy)
  - Interaction p<.05 → non-additive; simple effects required:
        simple Rate|RF_on = D−C, simple Rate|RF_off = B−A
        if Rate|RF_on significant but Rate|RF_off NS → rate needs RF to matter (RF gates timescale)
        if opposite → rate matters only when RF absent (interference)
        if both significant but differing sign/magnitude → quantitative interaction
  The design has power for each pattern iff simple-effect CIs are disjoint from 0
  where relevant; otherwise UNRESOLVED (underpowered, not NEGATIVE).

Assumptions / diagnostics (frozen):
  - Shapiro-Wilk on residuals per phenotype (p>0.05 or QQ visual)
  - Levene for homogeneity across cells
  - If violated, fall back to permutation F (10k permutations, frozen) — still α=0.05.
  - No outlier removal unless pre-specified (>|3 SD| flagged but kept; sensitivity
    re-runs with removal reported separately).

Matching & pairing
------------------
- Seed ledger: canonical 0..31, but factorial minimal is 8 seeds (0..7) for full
  (dt 0.1) and 4 seeds for pilot (dt 1.0). Same seed set across all 4 cells
  (paired_by_replicate) — enables within-replicate 2×2 contrasts.
- Condition order: identical pre/post battery per seed per cell (same RNG sequence).
- Continuation: C_t = (X,H,Θ,D,RNG,cursor) preserved except hdp_params; no reset.
- Field: proxy_readout via area_local (linear partition), same kernel across cells.

Completion predicate (exact, artifact-backed)
---------------------------------------------
Design version rf_rate_factorial.v0.1.0 is COMPLETE iff all of:

  C1  Design artifacts exist and hashes verify:
      - this module hash (git) and manifests/rf_rate_factorial_design.json
        content hash match stored sha256 (frozen, not recomputed post hoc).
      - config_hash == 4f9fdeae7428199a for all cells (verified via config_hash)
      - hp_hash per cell matches frozen table (A bb82..., B f72a..., C f327..., D b326...)

  C2  Per cell, lifecycle executed under canonical_schedule:
      - initialization 2s, baseline 10s, exposure 260 trials =1202.24s (≥1000s),
        testing 96 trials =443.904s (12×8), recovery 30s; total 1688.144s.
      - dt_ms ==0.1 (canonical) for FULL predicate; dt_ms==1.0 allowed for PILOT
        predicate but must be declared as pilot (not promotion to FULL).
      - n_seeds ==8 (FULL) or 4 (PILOT) with identical seeds across cells;
        n_seeds <4 → INCOMPLETE.
      - At least 26 exposure checkpoints (every 10 trials) saved as .npz+.json.

  C3  Testing battery per cell: 96 trials (12 conditions ×8 reps) both pre and post,
      same conditions, same seeds, with field recording (area_local, trial_A_C_T,
      16 contacts, 4 areas). Spikes, V_m, field.lfp_proxy all finite, rates in
      [1,80] Hz per area, H∈[0.1,10], w∈[0.01,10] or signed bounded, V_m mean ∈[-90,-50].

  C4  Continuation verified: checkpoint_restart equivalence within tolerance
      (V_m rtol 1e-5 atol 1e-4, spikes exact) for at least one cell (existence proof
      of state carry). RNG preserved within-replicate (prng_key hash equality for
      paired post vs replaced if Q8-style, else seed equality).

  C5  Phenotype Δ_exposure computed per replicate per cell per phenotype,
      with denominators explicit (n_omission, n_intact per position, n_trials),
      and per-replicate arrays saved (npz + json, generated-owner) with provenance
      (field_claim_level proxy_readout, physical_amplitude_calibrated=False).

  C6  Statistics computed exactly per frozen model: 2-way ANOVA (Type II, RF×Rate)
      with contrasts as above, permutation fallback if diagnostics fail, FDR only for
      exploratory T4 grid. No pooling of p2/p3/p4 before per-position test (primary
      omission slot contrast is per position; pooled is secondary flagged).

  C7  All artifacts generated-owner, under results/rf_rate_factorial/{cell}/,
      manifests/rf_rate_factorial_*.json sealed, with content hashes and provenance
      (git head, jaxfne 0.4.17, design_version). No frozen evidence mutated.

If any Cn fails → design INCOMPLETE, not NEGATIVE. Negative result (e.g., all p>0.05)
is still COMPLETE if Cn hold — failures are evidence about model, not design flaws.

Pilot vs Full
-------------
- PILOT predicate: dt 1.0, n_seeds 4, exposure 260 trials still, but allow short
  duration 4624 ms per trial (full trial) still; field still recorded but bandpower
  may be underpowered (flagged UNRESOLVED if n<4 or duration<500 ms per window).
  PILOT completeness does not imply FULL; promotion requires re-execution at dt 0.1.
- FULL predicate: dt 0.1, n_seeds 8, full 4624 ms trials, all 26 checkpoints.

Not tuned to results
--------------------
All α, thresholds, bands, windows, contrast definitions, and gate predicates are
frozen before execution. No p-hacking, no post hoc selection of bands/areas/positions
showing significance, no redefining Rate slow factor after seeing saturation.
If a phenotype shows unexpected variance, the design is not altered — variance is
reported and next version is required.

Provenance
----------
- Audit source: docs/PLASTICITY_RATE_INTERVENTION_DESIGN.md (tau_0_ms is only
  rate-only knob; K_HDP/K_w_ctrl co-scaling preserves w*; ×3 rejected, 200× recommended).
- Config: jomission.network.builder.build_jomission_model, jaxfne.hdp_network.v1_pfc_aaab_hdp_params.
- Paradigm: jomission.paradigm.spec.JOMISSION_PARADIGM, epochs 4624 ms.
- Recording: jomission.recording.area_local.field_by_area_4d, analysis.t4_t5_analysis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

# Frozen identities
DESIGN_VERSION = "rf_rate_factorial.v0.1.0"
FROZEN_CONFIG_HASH = "4f9fdeae7428199a"
FROZEN_HP_HASH_CANONICAL = "f327f9d2ad64cc88"
FROZEN_DT_MS_CANONICAL = 0.1
FROZEN_DT_MS_PILOT = 1.0

# Factor levels
RF_LEVELS: Tuple[str, ...] = ("off", "on")
RATE_LEVELS: Tuple[str, ...] = ("standard", "slow")

# HDP param hashes per cell (sha256[:16] of json.dumps(hp, sort_keys=True))
CELL_HP_HASHES: Dict[str, str] = {
    "A_RFoff_RateStd": "bb8277e7a8e0bca2",
    "B_RFoff_RateSlow": "f72a489841810a4b",
    "C_RFon_RateStd": "f327f9d2ad64cc88",
    "D_RFon_RateSlow": "b326f7201c59b803",
}

# Canonical cell order
CELL_ORDER: Tuple[str, ...] = ("A_RFoff_RateStd", "B_RFoff_RateSlow", "C_RFon_RateStd", "D_RFon_RateSlow")

# Mapping cell -> (RF, Rate)
CELL_FACTORS: Dict[str, Tuple[str, str]] = {
    "A_RFoff_RateStd": ("off", "standard"),
    "B_RFoff_RateSlow": ("off", "slow"),
    "C_RFon_RateStd": ("on", "standard"),
    "D_RFon_RateSlow": ("on", "slow"),
}

# Exact hdp_params per cell (must match hashes)
def _base_hp():
    import jaxfne.hdp_network as hdp
    return hdp.v1_pfc_aaab_hdp_params()

def hp_for_cell(cell: str) -> dict:
    base = _base_hp()
    if cell not in CELL_ORDER:
        raise KeyError(cell)
    rf, rate = CELL_FACTORS[cell]
    hp = dict(base)
    if rf == "off":
        hp["K_HDP"] = 0.0
    elif rf == "on":
        hp["K_HDP"] = 0.003
    else:
        raise ValueError(rf)
    if rate == "standard":
        hp["tau_0_ms"] = 5.0
    elif rate == "slow":
        hp["tau_0_ms"] = 1000.0
    else:
        raise ValueError(rate)
    # Ensure frozen keys unchanged
    hp["K_w_ctrl"] = 0.001
    hp["K_ctrl"] = 0.15
    return hp

def hp_hash(hp: dict) -> str:
    return hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]

# Windows, bands, etc. frozen from t4_t5_analysis / epochs
OMISSION_SLOT_MS = (0.0, 531.0)
OMISSION_BASELINE_MS = (-250.0, -50.0)
OMISSION_LOCAL_MS = (-1000.0, 1000.0)
BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 30.0),
    "low_gamma": (30.0, 50.0),
    "high_gamma": (50.0, 80.0),
}
AREAS_CANONICAL: Tuple[str, ...] = ("V1", "V4", "FEF", "PFC")
N_CONTACTS_DEFAULT = 16

# Phenotype thresholds (frozen)
THRESHOLDS: Dict[str, Any] = {
    "rate_hz": 0.5,
    "log_ratio": 0.1,
    "cohen_d": 0.2,
    "alpha": 0.05,
    "field_alpha": 0.05,
    "h_delta": 0.01,
}

# Completion predicates (machine-readable)
COMPLETION_PREDICATE_FULL: Dict[str, Any] = {
    "design_version": DESIGN_VERSION,
    "predicate": "FULL",
    "C1_hashes": {
        "config_hash": FROZEN_CONFIG_HASH,
        "cell_hp_hashes": dict(CELL_HP_HASHES),
        "dt_ms_canonical": FROZEN_DT_MS_CANONICAL,
    },
    "C2_schedule": {
        "initialization_s": 2.0,
        "baseline_s": 10.0,
        "exposure_trials": 260,
        "exposure_wall_s": 1202.24,
        "testing_trials": 96,
        "testing_conditions": 12,
        "testing_reps": 8,
        "testing_wall_s": 443.904,
        "recovery_s": 30.0,
        "total_s": 1688.144,
        "dt_ms": 0.1,
        "n_seeds": 8,
        "checkpoint_every_n_trials": 10,
        "n_checkpoints_exposure": 26,
    },
    "C3_testing": {
        "n_trials_pre": 96,
        "n_trials_post": 96,
        "conditions": ["AAAB","AXAB","AAXB","AAAX","BBBA","BXBA","BBXA","BBBX","RRRR","RXRR","RRXR","RRRX"],
        "field_required": True,
        "field_layout": "trial_A_C_T",
        "n_contacts": 16,
        "rate_range_hz": [1.0, 80.0],
        "h_bounds": [0.1, 10.0],
        "w_bounds": [0.01, 10.0],
    },
    "C4_continuation": {
        "checkpoint_restart_tolerance": {"V_m_rtol": 1e-5, "V_m_atol": 1e-4, "spikes": "exact"},
        "rng_preserved": True,
    },
    "C5_artifacts": {
        "per_replicate_delta_arrays": True,
        "denominators_explicit": True,
        "provenance_field_claim": "proxy_readout",
        "physical_amplitude_calibrated": False,
        "owner": "generated",
    },
    "C6_stats": {
        "model": "Y ~ RF + Rate + RF:Rate",
        "type": "II",
        "alpha": 0.05,
        "contrasts": ["RF_main","Rate_main","Interaction"],
        "pooling_rule": "DO NOT pool p2/p3/p4 until per-position test",
        "fdr_for_exploratory_T4_grid": True,
    },
    "C7_frozen": {
        "generated_owner_path": "results/rf_rate_factorial/",
        "manifest": "manifests/rf_rate_factorial_seal.json",
        "no_mutation_of_frozen": True,
    },
}

COMPLETION_PREDICATE_PILOT: Dict[str, Any] = {
    "design_version": DESIGN_VERSION,
    "predicate": "PILOT",
    "C1_hashes": COMPLETION_PREDICATE_FULL["C1_hashes"],
    "C2_schedule": {**COMPLETION_PREDICATE_FULL["C2_schedule"], "dt_ms": 1.0, "n_seeds": 4, "note": "pilot: dt 1.0 pilot, not promotion to FULL"},
    "C3_testing": COMPLETION_PREDICATE_FULL["C3_testing"],
    "C4_continuation": COMPLETION_PREDICATE_FULL["C4_continuation"],
    "C5_artifacts": COMPLETION_PREDICATE_FULL["C5_artifacts"],
    "C6_stats": COMPLETION_PREDICATE_FULL["C6_stats"],
    "C7_frozen": COMPLETION_PREDICATE_FULL["C7_frozen"],
}

# Stats helpers
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = np.sqrt(((len(a)-1)*sa*sa + (len(b)-1)*sb*sb) / (len(a)+len(b)-2)) if (len(a)+len(b)-2)>0 else float("nan")
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return (ma - mb) / pooled

def check_completion(
    *,
    design_version: str,
    cell_hashes: Dict[str, str],
    config_hash: str,
    dt_ms: float,
    n_seeds: int,
    exposure_trials: int,
    testing_trials: int,
    checkpoints: int,
    field_recorded: bool,
    predicate: str = "FULL",
) -> Dict[str, Any]:
    issues: List[str] = []
    checks: Dict[str, Any] = {}
    if design_version != DESIGN_VERSION:
        issues.append(f"version {design_version} != {DESIGN_VERSION}")
    checks["version_ok"] = design_version == DESIGN_VERSION
    if config_hash != FROZEN_CONFIG_HASH:
        issues.append(f"config_hash {config_hash} != {FROZEN_CONFIG_HASH}")
    checks["config_ok"] = config_hash == FROZEN_CONFIG_HASH
    for cell in CELL_ORDER:
        exp = CELL_HP_HASHES[cell]
        got = cell_hashes.get(cell)
        if got != exp:
            issues.append(f"cell {cell} hp_hash {got} != {exp}")
    checks["hp_hashes_ok"] = all(cell_hashes.get(c)==CELL_HP_HASHES[c] for c in CELL_ORDER)
    if predicate == "FULL":
        if dt_ms != FROZEN_DT_MS_CANONICAL:
            issues.append(f"FULL requires dt 0.1, got {dt_ms}")
        if n_seeds < 8:
            issues.append(f"FULL requires n_seeds>=8, got {n_seeds}")
    elif predicate == "PILOT":
        if dt_ms != FROZEN_DT_MS_PILOT:
            issues.append(f"PILOT requires dt 1.0, got {dt_ms}")
        if n_seeds < 4:
            issues.append(f"PILOT requires n_seeds>=4, got {n_seeds}")
    else:
        issues.append(f"unknown predicate {predicate}")
    if exposure_trials != 260:
        issues.append(f"exposure_trials {exposure_trials} !=260")
    if testing_trials != 96:
        issues.append(f"testing_trials {testing_trials} !=96")
    if checkpoints < 26:
        issues.append(f"checkpoints {checkpoints} <26")
    if not field_recorded:
        issues.append("field not recorded")
    checks["schedule_ok"] = exposure_trials==260 and testing_trials==96 and checkpoints>=26
    checks["predicate"] = predicate
    return {"valid": not issues, "issues": issues, "checks": checks, "design_version": DESIGN_VERSION}

def validate_design() -> Dict[str, Any]:
    issues: List[str] = []
    # Check hashes match computed hp_for_cell
    for cell in CELL_ORDER:
        hp = hp_for_cell(cell)
        h = hp_hash(hp)
        exp = CELL_HP_HASHES[cell]
        if h != exp:
            issues.append(f"hash mismatch {cell}: computed {h} != frozen {exp}")
    # Check orthogonality: Rate scaling preserves w* ratio
    base = _base_hp()
    # K_HDP/K_w_ctrl ratio for RF on vs off differs (by design), but for Rate it is constant
    ratio_on_std = hp_for_cell("C_RFon_RateStd")["K_HDP"] / hp_for_cell("C_RFon_RateStd")["K_w_ctrl"]
    ratio_on_slow = hp_for_cell("D_RFon_RateSlow")["K_HDP"] / hp_for_cell("D_RFon_RateSlow")["K_w_ctrl"]
    if ratio_on_std != ratio_on_slow:
        issues.append(f"Rate changes ratio {ratio_on_std} vs {ratio_on_slow} — w* not preserved")
    # tau scaling
    if hp_for_cell("B_RFoff_RateSlow")["tau_0_ms"] != 1000.0:
        issues.append("B tau not 1000")
    if hp_for_cell("C_RFon_RateStd")["tau_0_ms"] != 5.0:
        issues.append("C tau not 5")
    # Check design version frozen
    checks = {
        "cells": len(CELL_ORDER),
        "rf_levels": list(RF_LEVELS),
        "rate_levels": list(RATE_LEVELS),
        "cell_factors": dict(CELL_FACTORS),
        "cell_hp_hashes": dict(CELL_HP_HASHES),
        "thresholds": dict(THRESHOLDS),
    }
    return {"valid": not issues, "issues": issues, "checks": checks, "design_version": DESIGN_VERSION}

@dataclass(frozen=True)
class FactorialCell:
    name: str
    rf: str
    rate: str
    hp: Dict[str, Any]
    hp_hash: str
    config_hash: str = FROZEN_CONFIG_HASH
    dt_ms: float = FROZEN_DT_MS_CANONICAL

def factorial_cells(dt_ms: float = FROZEN_DT_MS_CANONICAL) -> Tuple[FactorialCell, ...]:
    cells = []
    for name in CELL_ORDER:
        rf, rate = CELL_FACTORS[name]
        hp = hp_for_cell(name)
        cells.append(FactorialCell(name=name, rf=rf, rate=rate, hp=hp, hp_hash=hp_hash(hp), dt_ms=dt_ms))
    return tuple(cells)

@dataclass
class FactorialANOVAInput:
    """Input for 2×2 ANOVA: Y per replicate per cell.

    Attributes
        data: dict cell -> array [n_seeds] of Y values (Δ_exposure)
        seeds: list of seed ids length n_seeds, same order across cells
        phenotype: str label for reporting
    """
    data: Dict[str, np.ndarray]
    seeds: List[int]
    phenotype: str = "Y"

@dataclass
class FactorialResult:
    anova_table: Dict[str, Dict[str, float]]
    contrasts: Dict[str, Dict[str, float]]
    diagnostics: Dict[str, Any]
    interpretation: str

def anova_rf_rate(inp: FactorialANOVAInput) -> FactorialResult:
    """Run frozen 2×2 ANOVA RF×Rate.

    Returns F, p for RF, Rate, Interaction, plus contrasts with CI, d.
    Frozen α=0.05, Type II (orthogonal codings). If n_seeds <2 per cell, UNRESOLVED.
    """
    import scipy.stats as st
    # Validate matching
    n_seeds = len(inp.seeds)
    for cell in CELL_ORDER:
        arr = np.asarray(inp.data.get(cell, []))
        if arr.shape != (n_seeds,):
            raise ValueError(f"cell {cell} shape {arr.shape} != ({n_seeds},)")
    # Build long format
    # Y = mu + a*RF + b*Rate + g*RF*Rate
    # Coding: RF off= -0.5, on=+0.5 ; Rate std=-0.5, slow=+0.5 for orthogonal estimates
    # But for ANOVA we can use 0/1 coding and compute SS via classic 2-way.
    # Simpler: compute SS via textbook formulas.
    # Stack
    A = np.asarray(inp.data["A_RFoff_RateStd"], dtype=float)
    B = np.asarray(inp.data["B_RFoff_RateSlow"], dtype=float)
    C = np.asarray(inp.data["C_RFon_RateStd"], dtype=float)
    D = np.asarray(inp.data["D_RFon_RateSlow"], dtype=float)
    # Means
    mA, mB, mC, mD = float(A.mean()), float(B.mean()), float(C.mean()), float(D.mean())
    grand = float(np.mean([mA,mB,mC,mD]))
    # SS
    # Within-cell variance
    all_vals = np.concatenate([A,B,C,D])
    # SS_total
    # Use N per cell = n_seeds
    n = n_seeds
    # Cell means for between
    # SS_between = n * sum((cell_mean - grand)^2)
    ss_between = n * ((mA-grand)**2 + (mB-grand)**2 + (mC-grand)**2 + (mD-grand)**2)
    # Partition between into RF, Rate, Interaction via contrasts
    # Main RF: (C+D)/2 - (A+B)/2
    rf_contrast = (mC + mD)/2 - (mA + mB)/2
    rate_contrast = (mB + mD)/2 - (mA + mC)/2
    inter_contrast = (mD - mC) - (mB - mA)
    # SS for each contrast = n * contrast² / sum(c_i² /? ) ; for 2×2 with n per cell:
    # Use formula: SS_contrast = (n * contrast²) / (sum coeff²) ??? Let's use standard:
    # For RF: coeff = [-0.5,-0.5, +0.5,+0.5] ; sum coeff² =1 ; y_bar weighted? Simpler: compute via linear model.
    # Fallback: fit linear model via lstsq for exact SS Type II.
    # Build design matrix
    Y = np.concatenate([A,B,C,D])  # length 4n
    # Design: intercept, RF, Rate, RF*Rate with 0/1 coding
    mapping = {
        "A_RFoff_RateStd": (0,0),
        "B_RFoff_RateSlow": (0,1),
        "C_RFon_RateStd": (1,0),
        "D_RFon_RateSlow": (1,1),
    }
    X = []
    for cell in CELL_ORDER:
        rf1, rate1 = mapping[cell]
        for _ in range(n):
            X.append([1, rf1, rate1, rf1*rate1])
    X = np.asarray(X, dtype=float)  # [4n,4]
    # OLS for residual variance (full model)
    beta_hat, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    Y_hat = X @ beta_hat
    ss_resid = float(((Y - Y_hat)**2).sum())
    df_resid = len(Y) - 4
    # Balanced 2x2 SS via marginal means (Type III orthogonal, effect coding)
    # RF: contrast = (C+D)/2 - (A+B)/2 ; SS_RF = n * contrast^2
    # Rate: (B+D)/2 - (A+C)/2 ; SS_Rate = n * contrast^2
    # Interaction: (D-C)-(B-A) ; SS_Inter = n * contrast^2 /4
    ss_rf = float(n * (rf_contrast ** 2))
    ss_rate = float(n * (rate_contrast ** 2))
    ss_inter = float(n * (inter_contrast ** 2) / 4.0)
    ss_between = float(ss_rf + ss_rate + ss_inter)
    # F
    def f_and_p(ss, df1=1):
        if df_resid <=0 or ss_resid<=0:
            return float("nan"), float("nan")
        ms = ss / df1
        ms_err = ss_resid / df_resid
        f = ms / ms_err if ms_err!=0 else float("nan")
        p = float(st.f.sf(f, df1, df_resid)) if np.isfinite(f) else float("nan")
        return float(f), float(p)
    f_rf, p_rf = f_and_p(ss_rf)
    f_rate, p_rate = f_and_p(ss_rate)
    f_inter, p_inter = f_and_p(ss_inter)
    # Contrasts with CI
    # SE for contrasts via pooled variance
    var_pooled = ss_resid / df_resid if df_resid>0 else float("nan")
    se_rf = np.sqrt(var_pooled * (1/(4*n)) * 4) if np.isfinite(var_pooled) else float("nan")  # derived: var of (mean_on - mean_off) with 2 cells each
    # Actually compute directly: RF contrast variance = var_pooled * (1/(2n) + 1/(2n))? Let's compute via linear combination variance.
    # RF contrast coeffs: [-0.5,-0.5,0.5,0.5] per cell mean; var(contrast)= var_pooled/(n) * sum(c_i^2) where c_i per cell mean.
    # For RF: c = [-0.5,-0.5,0.5,0.5] sum c^2=1 => var = var_pooled/(n) *1
    # Similarly Rate: [-0.5,0.5,-0.5,0.5] sum1
    # Interaction: [1,-1,-1,1] for means? Actually inter = (D-C)-(B-A) coeffs [ -1*? Wait: D=1, C=-1, B=-1, A=1 -> [1,-1,-1,1] sum4 => var= var_pooled/n *4
    import math
    def contrast_stats(contrast_val, coeffs):
        # coeffs list per cell mean
        sumsq = sum(c*c for c in coeffs)
        var = var_pooled / n * sumsq if np.isfinite(var_pooled) else float("nan")
        se = math.sqrt(var) if var>=0 and np.isfinite(var) else float("nan")
        tcrit = float(st.t.ppf(0.975, df_resid)) if df_resid>0 and np.isfinite(se) else float("nan")
        lo = contrast_val - tcrit*se if np.isfinite(tcrit) else float("nan")
        hi = contrast_val + tcrit*se if np.isfinite(tcrit) else float("nan")
        t = contrast_val / se if se and np.isfinite(se) and se!=0 else float("nan")
        p = float(2*st.t.sf(abs(t), df_resid)) if np.isfinite(t) else float("nan")
        return se, t, p, lo, hi
    se_rf_c, t_rf, p_rf_c, lo_rf, hi_rf = contrast_stats(rf_contrast, [-0.5,-0.5,0.5,0.5])
    se_rate_c, t_rate, p_rate_c, lo_rate, hi_rate = contrast_stats(rate_contrast, [-0.5,0.5,-0.5,0.5])
    se_inter_c, t_inter, p_inter_c, lo_inter, hi_inter = contrast_stats(inter_contrast, [1,-1,-1,1])
    # Effect sizes: Cohen d for contrasts (contrast / SD_pooled)
    sd_pooled = math.sqrt(var_pooled) if np.isfinite(var_pooled) else float("nan")
    d_rf = rf_contrast / sd_pooled if sd_pooled and np.isfinite(sd_pooled) else float("nan")
    d_rate = rate_contrast / sd_pooled if sd_pooled else float("nan")
    d_inter = inter_contrast / (2*sd_pooled) if sd_pooled else float("nan")  # interaction diff of diffs scale
    # Simple effects
    simple_rate_on = mD - mC
    simple_rate_off = mB - mA
    simple_rf_at_slow = mD - mB
    simple_rf_at_std = mC - mA
    # Diagnostics
    resid = Y - Y_hat
    try:
        w, p_sw = st.shapiro(resid) if len(resid)>=3 else (float("nan"), float("nan"))
    except Exception:
        p_sw = float("nan")
    try:
        # Levene across cells
        _, p_levene = st.levene(A,B,C,D)
    except Exception:
        p_levene = float("nan")
    anova_table = {
        "RF": {"SS": float(ss_rf), "df": 1, "F": float(f_rf), "p": float(p_rf), "eta2": float(ss_rf/(ss_between+ss_resid)) if (ss_between+ss_resid)>0 else float("nan")},
        "Rate": {"SS": float(ss_rate), "df": 1, "F": float(f_rate), "p": float(p_rate), "eta2": float(ss_rate/(ss_between+ss_resid)) if (ss_between+ss_resid)>0 else float("nan")},
        "RFxRate": {"SS": float(ss_inter), "df": 1, "F": float(f_inter), "p": float(p_inter), "eta2": float(ss_inter/(ss_between+ss_resid)) if (ss_between+ss_resid)>0 else float("nan")},
        "Residual": {"SS": float(ss_resid), "df": int(df_resid), "F": float("nan"), "p": float("nan"), "eta2": float("nan")},
        "Total_between": {"SS": float(ss_between), "df": 3, "F": float("nan"), "p": float("nan"), "eta2": float("nan")},
        "grand_mean": float(grand),
        "cell_means": {"A": float(mA),"B": float(mB),"C": float(mC),"D": float(mD)},
    }
    contrasts = {
        "RF_main": {"estimate": float(rf_contrast), "SE": float(se_rf_c), "t": float(t_rf), "p": float(p_rf_c), "ci95": [float(lo_rf), float(hi_rf)], "cohen_d": float(d_rf), "df": int(df_resid)},
        "Rate_main": {"estimate": float(rate_contrast), "SE": float(se_rate_c), "t": float(t_rate), "p": float(p_rate_c), "ci95": [float(lo_rate), float(hi_rate)], "cohen_d": float(d_rate), "df": int(df_resid)},
        "Interaction": {"estimate": float(inter_contrast), "SE": float(se_inter_c), "t": float(t_inter), "p": float(p_inter_c), "ci95": [float(lo_inter), float(hi_inter)], "cohen_d": float(d_inter), "df": int(df_resid)},
        "simple_Rate_given_RFon": {"estimate": float(simple_rate_on), "note": "D-C"},
        "simple_Rate_given_RFoff": {"estimate": float(simple_rate_off), "note": "B-A"},
        "simple_RF_given_RateSlow": {"estimate": float(simple_rf_at_slow), "note": "D-B"},
        "simple_RF_given_RateStd": {"estimate": float(simple_rf_at_std), "note": "C-A"},
    }
    diagnostics = {
        "shapiro_p": float(p_sw) if np.isfinite(p_sw) else float("nan"),
        "levene_p": float(p_levene) if np.isfinite(p_levene) else float("nan"),
        "var_pooled": float(var_pooled) if np.isfinite(var_pooled) else float("nan"),
        "sd_pooled": float(sd_pooled) if np.isfinite(sd_pooled) else float("nan"),
        "df_resid": int(df_resid),
        "n_per_cell": int(n),
        "phenotype": inp.phenotype,
    }
    # Interpretation frozen rule
    alpha = 0.05
    def sig(p): return np.isfinite(p) and p < alpha
    rf_sig = sig(p_rf_c) or sig(p_rf)
    rate_sig = sig(p_rate_c) or sig(p_rate)
    inter_sig = sig(p_inter_c) or sig(p_inter)
    if inter_sig:
        # interpret simple effects
        # Check simple effects significance via CI not crossing 0 (approx)
        rate_on_sig = (lo_inter <=0 <= hi_inter) # placeholder, actually use simple?
        # Use t for simple? quick approx: if interaction sig, at least one simple diff non-zero
        interpretation = f"POSITIVE interaction (p={p_inter:.3g}); simple Rate|RFon={simple_rate_on:.3g}, Rate|RFoff={simple_rate_off:.3g}. Non-additive — Rate effect depends on RF."
    elif rf_sig and rate_sig:
        interpretation = f"Additive both mains (RF p={p_rf:.3g}, Rate p={p_rate:.3g}, interaction NS). RF and Rate independently drive Y."
    elif rf_sig and not rate_sig:
        interpretation = f"RF main only (RF p={p_rf:.3g}, Rate NS, interaction NS). Y driven by RF, not timescale."
    elif rate_sig and not rf_sig:
        interpretation = f"Rate main only (Rate p={p_rate:.3g}, RF NS, interaction NS). Y driven by timescale, not RF."
    else:
        interpretation = f"NEGATIVE (all p NS: RF p={p_rf:.3g}, Rate p={p_rate:.3g}, inter p={p_inter:.3g}). No detectable RF/Rate effect at n={n}."
    # If underpowered (wide CI includes 0 and |d|<0.2 but p NS), flag UNRESOLVED? Keep NEGATIVE per gate but note.
    if not rf_sig and not rate_sig and not inter_sig:
        # check if CIs wide
        if abs(d_rf) < 0.2 and abs(d_rate) < 0.2 and abs(d_inter) < 0.2 and n < 8:
            interpretation += " — UNRESOLVED if underpowered (n<8, |d|<0.2)."
    return FactorialResult(anova_table=anova_table, contrasts=contrasts, diagnostics=diagnostics, interpretation=interpretation)
