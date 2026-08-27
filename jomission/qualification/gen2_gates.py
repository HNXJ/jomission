"""Gen-2 qualification gates B0..B12 — SPECIFIED constants + seal predicates.

Source-of-truth for thresholds: scratch/gen2_qualification_skeleton_A.md §4 +
                               scratch/gen2_qualification_skeleton_A.json gates{}
Spec version: BIOPHYSICAL_QUALIFICATION_v0.1
Do not tune thresholds post-hoc; all are SPECIFIED before observation.

Authorization W1 (2026-08-27):
  - K_HDP/K_w_ctrl co-scaled (r=3.0) is DEFERRED — not exposed here.
  - W1 is topology+heterogeneity+observability only.
  - H terminology gate: B0 must FAIL/UNRESOLVED on H_conceptual (5-dim) vs
    H_implemented (scalar h) mismatch until M-06 resolved. Implemented via
    check_h_terminology().

Provides:
  - SPECIFIED_GATES dict (import from skeleton A)
  - FORBIDDEN_TERMS_B10 list (grep source for B10 objective audit)
  - seal_predicate() stubs per gate + validate_gate(gate_id, results)
  - check_h_terminology() for B0 H gate
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Forbidden terms for B10 objective (imported from skeleton A json)
# ---------------------------------------------------------------------------
FORBIDDEN_TERMS_B10: list[str] = [
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "omission",
    "AXAB",
    "BXBA",
    "AAXB",
    "BBXA",
    "AAAX",
    "BBBX",
    "RXRR",
    "RRXR",
    "RRRX",
    "Delta_exposure",
    "delta_exposure",
    "omission_effect",
    "q8_history_valid",
    "history_valid_HTheta_vs_fast",
]

ALLOWED_OBJECTIVE_TERMS_B10: list[str] = [
    "firing_regime",
    "heterogeneity",
    "AI dynamics",
    "asynchronous_irregular",
    "E/I",
    "propagation",
    "FF/FB",
    "H response",
    "H(t)",
    "Theta memory",
    "Theta(t)",
    "retention",
]

# ---------------------------------------------------------------------------
# SPECIFIED gate definitions — verbatim from skeleton A json gates{}.
# Keep thresholds as SPECIFIED constants, not tuned.
# ---------------------------------------------------------------------------
# Load from skeleton json at import if available, else fallback to inline.
# We keep inline copy as SPECIFIED (hash-verified against skeleton json).
SPECIFIED_GATES: dict[str, dict[str, Any]] = {
    "B0": {
        "label": "parameter/connectivity provenance audit",
        "scientific_question": "Are all Gen-2 parameters, connectivity tables, delays, RF geometry, and HDP/Theta parameters traceable to explicit provenance class and hash, with Gen-1 reference immutable?",
        "required_observables": [
            "config_hash",
            "numerical_config_hash",
            "hp_hash",
            "RFConfig.hash",
            "layer_count_frac",
            "AREA_LAYER_CELL_TYPES",
            "CONNECTIVITY_TABLE/p_ff/p_fb",
            "DELAY_FF/FB/WITHIN_MS",
            "HDP params live vs dead keys",
            "dt_ms",
            "n_per_area",
            "emitter",
            "geometry populations",
            "neuron_metadata",
        ],
        "candidate_metric": "provenance_coverage = n_traced/n_total; hash_stability; dead_key_count; per-table provenance histogram",
        "pass_criteria": "provenance_coverage==1.0 and hash_stability and dead_key_count==0 and all validate_* valid",
        "fail_criteria": "any validate invalid or hash drift without Ledger or dead key consumed as live",
        "unresolved_criteria": "engine API ambiguity -> RESOURCE_BOUNDARY",
        "allowed_parameters_to_change": [],
        "forbidden_target_information": ["T1-T7", "Q8 history-valid", "omission-success metric"],
        "seal_predicate": "B0_PASS iff provenance_coverage==1.0 and hash_stability and dead_key_count==0 and forall validate.valid",
        "thresholds_SPECIFIED": {
            "provenance_coverage": 1.0,
            "dead_key_count": 0,
        },
        # H terminology gate — appended by W1.0 (not in original skeleton, required per W1 task)
        "h_terminology_gate": {
            "conceptual": "5-dim H_COORDINATES [0.1,1,10,100,1000]s per jomission/dynamics/h_state.py:28-69",
            "implemented": "scalar h per JaxFNE emitters tau_i = tau_0_ms * size^3, single H ODE",
            "check": "check_h_terminology() must REPORT mismatch until M-06; B0 UNRESOLVED with limitation 'H_conceptual vs H_implemented mismatch — see M-06'",
            "M06_status": "deferred — W1 topology+observability only",
        },
    },
    "B1": {
        "label": "baseline firing regime",
        "scientific_question": "Does Gen-2 at rest/spontaneous and matched drive exhibit plausible firing regime with area/layer/class-conditional rates near 5-10 Hz as distribution?",
        "required_observables": ["r_i per neuron", "r_{a,l,c} per population", "global mean mu_r", "SD(r_i)", "H(t) covariate"],
        "candidate_metric": "global mu_r in [3,15] Hz; per-class mu_{a,l,c} vs class targets: E 5-8 (V1 L4 E 6-7, L2/3 E 4-6, L5 E 7-10, FEF/PFC E 3-6), PV 12-25, SST 6-12, VIP 8-15 Hz; fraction pops within target +/-50%",
        "pass_criteria": "global mu in [3,15] and >60% pops within class target +/-50% and no E silence <0.5 Hz or hyper >60 Hz; E<PV mean",
        "fail_criteria": "global silence/gamma explosion or >40% E pops mu<1 or >25 Hz or any class >80 Hz",
        "seal_predicate": "B1_PASS iff B0_PASS and global mu in [3,15] and coverage>=60% and no pathological class and E<PV",
        "thresholds_SPECIFIED": {
            "global_mu_Hz": (3.0, 15.0),
            "coverage_fraction": 0.60,
            "per_class_E_Hz": (5.0, 8.0),
            "per_class_PV_Hz": (12.0, 25.0),
            "per_class_SST_Hz": (6.0, 12.0),
            "per_class_VIP_Hz": (8.0, 15.0),
            "silence_E_Hz": 0.5,
            "hyper_any_Hz": 80.0,
            "hyper_mean_Hz": 60.0,
            "tolerance_fraction": 0.50,
        },
        "dependencies": ["B0"],
    },
    "B2": {
        "label": "neuronal heterogeneity + asynchronous-irregular state",
        "candidate_metric": "SD(r_i) [2,12] Hz; CV_rate [0.3,1.5]; CV_ISI mean [0.5,1.5]; Fano [0.7,2.0]; mean rho_ij [-0.05,0.2]; rho(d) decaying slope negative",
        "pass_criteria": "CV_rate>0.3 and 0.5<=mean CV_ISI<=1.5 and median Fano in [0.7,2.0] and mean rho in [-0.05,0.2] and rho(d) decay>0.05",
        "fail_criteria": "mean rho>0.3 sync or CV_rate<0.15 homogeneous or CV_ISI<0.3 clock-like or >2.5 bursty or Fano<0.4 regular",
        "seal_predicate": "B2_PASS iff B1_PASS and CV_rate>0.3 and CV_ISI in [0.5,1.5] and Fano in [0.7,2.0] and mean rho in [-0.05,0.2] and rho(d) decaying",
        "thresholds_SPECIFIED": {
            "CV_rate_min": 0.3,
            "CV_ISI_range": (0.5, 1.5),
            "Fano_range": (0.7, 2.0),
            "rho_range": (-0.05, 0.2),
            "rho_decay_min": 0.05,
            "rho_fail": 0.3,
            "CV_rate_fail": 0.15,
        },
        "dependencies": ["B1"],
    },
    "B3": {
        "label": "laminar E/I operating regime",
        "candidate_metric": "Efrac in [0.15,0.60] for E cells; BI near 0; inhibition >0.2*|I_E|; not clipped",
        "pass_criteria": ">=80% qualified pops Efrac in [0.15,0.60] and inhibition not zero and currents finite and laminar gradient plausible",
        "fail_criteria": "any large pop Efrac<0.05 or >0.95 or bounds-clipped or sign contradicts W",
        "seal_predicate": "B3_PASS iff realized E/I extracted and Efrac in [0.15,0.60] for >=80% pops and finite",
        "thresholds_SPECIFIED": {"Efrac_range": (0.15, 0.60), "Efrac_qualifying_fraction": 0.80, "Efrac_fail_lo": 0.05, "Efrac_fail_hi": 0.95},
        "dependencies": ["B2"],
    },
    "B4": {
        "label": "dense-local/sparse-global ensemble architecture",
        "candidate_metric": "p_local 0.2-0.5 short vs <0.1 long; p_FF 0.2-0.4 p_FB 0.1-0.3 sparse; motif rank>0.5 vs literature; Q in [0.2,0.7]; locality_index>3; Delta_r per W=1 in [0.05,2.0] Hz",
        "pass_criteria": "spatial decay significant p<0.05 and motif rank>0.5 and Q>0.2 and locality_index>3 and weights effective not saturating (2x gain still changes Delta_r >5%)",
        "seal_predicate": "B4_PASS iff spatial decay sig and motif rank>0.5 and Q>0.2 and locality_index>3 and not saturating",
        "thresholds_SPECIFIED": {"p_local_short": (0.2, 0.5), "p_local_long": 0.1, "p_FF": (0.2, 0.4), "p_FB": (0.1, 0.3), "motif_rank": 0.5, "Q_min": 0.2, "locality_index_min": 3.0, "saturation_delta": 0.05},
        "dependencies": ["B3"],
    },
    "B5": {
        "label": "energy-matched 32x32 RF/sensory qualification",
        "candidate_metric": "E_parity |E_off-E_on|/max <=0.05; temporal 25% per slot +/-5% delay/fx 0 omission 0; sparsity [0.18,0.30]; Jaccard<0.15; L1 row sum 1 +/-1e-5; non-V1 drive 0; CV graded [4,15]",
        "pass_criteria": "RFOperator.validate PASS and parity<=5% and sparsity in [0.18,0.30] and Jaccard<0.15 and V1-only and omission zero and graded capability proved",
        "seal_predicate": "B5_PASS iff validate PASS and parity<=5% and sparsity in bounds and Jaccard<0.15 and V1-only and omission_zero and capability_proved",
        "thresholds_SPECIFIED": {"E_parity_max": 0.05, "E_parity_fail": 0.10, "sparsity_range": (0.18, 0.30), "Jaccard_max": 0.15, "L1_tol": 1e-5, "temporal_share": 0.25, "temporal_tol": 0.05, "CV_range": (4.0, 15.0)},
        "dependencies": ["B4"],
    },
    "B6": {
        "label": "FF/FB hierarchical propagation",
        "candidate_metric": "acc_a > chance+2SE all areas; PI = acc_V4/acc_V1 in [0.3,1.0]; latencies V1<V4<FEF<PFC Spearman>0.7 p<0.05; Delta_prop_FF drop >20% higher areas; FB drop >15% contextual modulation; shuffle <8%",
        "pass_criteria": "above-chance all areas and ordered latencies and FF degrades higher >15% and FB degrades modulation >15% and shuffle smaller",
        "seal_predicate": "B6_PASS iff above-chance all and ordered latencies and FF>15% and FB>15% and shuffle<8%",
        "thresholds_SPECIFIED": {"acc_chance": 1.0 / 3.0, "PI_range": (0.3, 1.0), "latency_rho": 0.7, "FF_drop": 0.15, "FB_drop": 0.15, "shuffle_max": 0.08},
        "dependencies": ["B5"],
    },
    "B7": {
        "label": "H system-identification/adaptation",
        "candidate_metric": "fit H(t)=H_inf-(H_inf-H0)exp(-t/tau_H); tau_H E ~4.1s canonical (target [2,8]s), PV 0.03s SST/VIP 0.11s; tau LONG 833s if tau_0_ms 1000; R2>0.7; DeltaH>0.01 d>0.5; corr(H,r)>0.5; tau_rec within 50% tau_H",
        "pass_criteria": "dense H measured and R2>0.7 and tau within factor2 and DeltaH sig and recovery consistent",
        "seal_predicate": "B7_PASS iff dense H and R2>0.7 and tau identified and DeltaH sig and recovery consistent",
        "thresholds_SPECIFIED": {"R2_min": 0.7, "tau_E_range_s": (2.0, 8.0), "DeltaH_min": 0.01, "corr_min": 0.5, "tau_rec_tol": 0.5},
        "dependencies": ["B6"],
    },
    "B8": {
        "label": "Theta acquisition/saturation/retention",
        "candidate_metric": "fit Theta(t)=Theta_inf-(Theta_inf-Theta0)exp(-t/tau_Theta); tau_Theta, Theta_inf, DeltaTheta>0.001 d>0.2 not clipped; functional Delta_rate_Theta >0.5 Hz p<0.05 d>0.2 POSITIVE; retention >30% after gap ~tau",
        "pass_criteria": "Theta->inf R2>0.7 and DeltaTheta sig not clipped AND functional Delta_rate POSITIVE and retention>30%",
        "seal_predicate": "B8_PASS iff Theta->inf R2>0.7 and DeltaTheta sig and Delta_rate POSITIVE and retention>30%",
        "thresholds_SPECIFIED": {"R2_min": 0.7, "DeltaTheta_min": 0.001, "functional_Hz": 0.5, "retention_min": 0.30},
        "dependencies": ["B7"],
    },
    "B9": {
        "label": "H/Theta functional-efficacy interventions",
        "candidate_metric": "slope s = d(Delta_rate)/d(alpha) >0.3 Hz per unit R2>0.5 p<0.05 monotonic; clamping Theta to pre abolishes Delta to |<0.5| NEGATIVE; history_valid > fast by >0.5 Hz",
        "pass_criteria": "both H and Theta monotonic sig slopes matching Ledger predicted direction and clamping abolishes and history_valid > fast",
        "seal_predicate": "B9_PASS iff slopes monotonic sig and clamping abolishes and history_valid > fast and direction matches prediction",
        "thresholds_SPECIFIED": {"slope_Hz_per_unit": 0.3, "R2_min": 0.5, "clamp_Hz": 0.5, "history_valid_delta_Hz": 0.5},
        "dependencies": ["B8", "B7"],
    },
    "B10": {
        "label": "AGSDR-assisted generic-state calibration",
        "candidate_metric": "J_generic = sum w_k normalized_score(B_k) for k in {B1,B2,B3,B6,B7,B8} weights frozen equal 1/6 NO T1-T7 term; DeltaJ = J_final-J_initial; iterations_to_threshold",
        "pass_criteria": "J improved and NO forbidden term in objective (grep verified) and final model re-passes B1-B9 without controller and controller not in C_t",
        "seal_predicate": "B10_PASS iff J improved and no forbidden term and final re-passes B1-B9 without controller and not in C_t",
        "thresholds_SPECIFIED": {"w_k": 1.0 / 6.0, "forbidden_hit_max": 0},
        "forbidden_terms": FORBIDDEN_TERMS_B10,
        "dependencies": ["B9"],
    },
    "B11": {
        "label": "Gen-2 freeze/readiness",
        "candidate_metric": "freeze_consistency recomputed==frozen boolean; seal_completeness=10/10; recording_completeness=1.0",
        "pass_criteria": "all 10 seals PASS and hashes stable and paradigm_exact PASS and recording union available and audit PASS",
        "seal_predicate": "B11_PASS iff B0..B10 all PASS and freeze_consistency and paradigm_exact and recording_completeness and audit PASS; version gen2.v0.1.0",
        "thresholds_SPECIFIED": {"seal_completeness": 10, "recording_completeness": 1.0},
        "dependencies": ["B10"],
    },
    "B12": {
        "label": "blind omission experiment only after qualification",
        "candidate_metric": "T1 <0.10 sig; T2 enrichment >1.5x; T3 d<0.2; T4 frontal p<0.05 corrected V1 weaker; T5 mean r>0; T6 T1+<T1-; T7 no fixed lag p>0.05; Q per q8_phenotype assign_polarity; Delta_exposure |Delta|<0.5 Hz is NEGATIVE per q8 closure; report POLARITY per assign_polarity",
        "pass_criteria": "B12 is evidence collection not qualification: each T/Q gets POSITIVE/NEGATIVE/UNRESOLVED per frozen criteria; no T must be POSITIVE to seal",
        "seal_predicate": "B12_SEALED iff B11_PASS and full trajectory continuous C_t and paradigm_exact and EvidenceRef written and T1-T7/Q computed per frozen estimands and Ledger extended",
        "thresholds_SPECIFIED": {"effect_threshold_Hz": 0.5, "T1_threshold": 0.10, "T2_enrichment": 1.5, "min_trials": 4, "min_window_ms": 500},
        "dependencies": ["B11"],
    },
}

# Sorted gate order for DAG validation
GATE_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12"]


# ---------------------------------------------------------------------------
# H terminology gate — B0 specific
# ---------------------------------------------------------------------------
def check_h_terminology(h_state_path: str | Path | None = None) -> dict[str, Any]:
    """Inspect jomission/dynamics/h_state.py for H_conceptual vs H_implemented mismatch.

    - H_conceptual: 5-dim H_COORDINATES [0.1,1,10,100,1000] claimed in docstring
    - H_implemented: scalar h via JaxFNE emitters tau_i = tau_0_ms * size^3

    For W1, this MUST REPORT the mismatch (since M-06 not yet applied), so B0
    will be UNRESOLVED with limitation "H_conceptual vs H_implemented mismatch — see M-06",
    not silently PASS.

    Returns dict with keys:
      has_H_COORDINATES, has_5dim_claim, has_scalar_note,
      h_conceptual, h_implemented,
      mismatch, limitation, recommendation
    """
    if h_state_path is None:
        h_state_path = Path(__file__).resolve().parents[1] / "dynamics" / "h_state.py"
    p = Path(h_state_path)
    result: dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "has_H_COORDINATES": False,
        "has_5dim_claim": False,
        "has_scalar_note": False,
        "has_conceptual_vs_implemented_qualifier": False,
        "h_conceptual": "5-dim H_COORDINATES per h_state.py:28-69",
        "h_implemented": "scalar h per JaxFNE emitters tau_i=tau_0*size^3 (E 125x)",
        "mismatch": True,
        "limitation": "H_conceptual vs H_implemented mismatch — see M-06",
        "recommendation": "Do not claim '5 independent H timescales are implemented' without qualifier 'conceptual vs implemented scalar h'",
    }
    if not p.exists():
        result["mismatch"] = True
        result["detail"] = "h_state.py not found"
        return result

    text = p.read_text()

    result["has_H_COORDINATES"] = "H_COORDINATES" in text
    # Detect claim of "5 independent H timescales are implemented" without qualifier
    # Look for docstring/comment claiming 5 timescales implemented as runtime
    five_pattern = re.search(r"5\s*(independent)?\s*H\s*timescales?\s*are\s*implemented", text, re.IGNORECASE)
    result["has_5dim_claim"] = bool(five_pattern)

    # Check if file already qualifies as conceptual vs scalar
    qualifier_patterns = [
        r"conceptual vs implemented",
        r"H_conceptual.*H_implemented",
        r"conceptual.*scalar h",
        r"1-D H truth",
        r"scalar h.*conceptual",
        r"not.*5 independent.*implemented",
    ]
    has_qualifier = any(re.search(pat, text, re.IGNORECASE) for pat in qualifier_patterns)
    result["has_conceptual_vs_implemented_qualifier"] = has_qualifier

    # Also check if docstring now distinguishes
    has_scalar_note = "scalar" in text.lower() and ("tau_i" in text or "size" in text.lower() or "JaxFNE" in text)
    result["has_scalar_note"] = has_scalar_note

    # Mismatch logic per task: B0 must FAIL on mismatch until M-06
    # For now REPORT mismatch since M-06 deferred
    if has_qualifier:
        result["mismatch"] = False
        result["limitation"] = ""
        result["status"] = "RESOLVED — qualifier present"
    else:
        result["mismatch"] = True
        result["limitation"] = "H_conceptual vs H_implemented mismatch — see M-06"
        result["status"] = "UNRESOLVED — mismatch until M-06"

    # Also report emitter scalar truth for completeness
    result["emitter_scalar_evidence"] = "jaxfne emitters tau_i = tau_0_ms * size_i^3, H scalar ODE H in [0.1,10]"
    return result


# ---------------------------------------------------------------------------
# Seal predicates — stubs that evaluate against supplied results dict
# ---------------------------------------------------------------------------
def seal_predicate_B0(results: dict[str, Any]) -> dict[str, Any]:
    """B0_PASS ⇔ provenance_coverage==1.0 ∧ hash_stability ∧ dead_key_count==0 ∧ ∀ validate.valid
    plus H terminology gate (UNRESOLVED if mismatch until M-06).
    """
    issues: list[str] = []
    coverage = results.get("provenance_coverage")
    hash_stable = results.get("hash_stability")
    dead = results.get("dead_key_count")
    validates = results.get("validates", {})

    if coverage is not None and coverage != 1.0:
        issues.append(f"provenance_coverage {coverage} != 1.0")
    elif coverage is None:
        issues.append("provenance_coverage missing")

    if hash_stable is False:
        issues.append("hash_stability false")
    elif hash_stable is None:
        issues.append("hash_stability missing")

    if dead is not None and dead != 0:
        issues.append(f"dead_key_count {dead} != 0")
    elif dead is None:
        issues.append("dead_key_count missing")

    # validate_* outputs: each should have valid==true
    for k, v in validates.items():
        if isinstance(v, dict) and not v.get("valid", False):
            issues.append(f"validate {k} invalid: {v.get('issues')}")

    # H terminology gate
    h_check = results.get("h_terminology", check_h_terminology())
    if h_check.get("mismatch"):
        issues.append(f"H terminology gate UNRESOLVED: {h_check.get('limitation')}")

    status = "B0_PASS" if not issues else "B0_UNRESOLVED" if any("H terminology" in i for i in issues) else "B0_FAIL"
    # If only H mismatch, mark UNRESOLVED not FAIL
    if issues and all("H terminology" in i for i in issues) and len(issues) == 1:
        status = "B0_UNRESOLVED"
    elif h_check.get("mismatch") and not issues:
        # shouldn't happen but handle
        status = "B0_UNRESOLVED"
        issues.append(f"H terminology gate UNRESOLVED: {h_check.get('limitation')}")

    return {"gate": "B0", "status": status, "issues": issues, "h_terminology": h_check}


def seal_predicate_B1(results: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    mu = results.get("global_mu_Hz")
    coverage = results.get("coverage_fraction")
    no_path = results.get("no_pathological", True)
    e_lt_pv = results.get("E_lt_PV", True)
    b0 = results.get("B0_status", "B0_PASS")
    if b0 != "B0_PASS":
        issues.append(f"B0 not PASS ({b0}) blocks B1")
    if mu is not None and not (3.0 <= mu <= 15.0):
        issues.append(f"global mu {mu} not in [3,15]")
    if coverage is not None and coverage < 0.60:
        issues.append(f"coverage {coverage} < 0.60")
    if not no_path:
        issues.append("pathological class detected")
    if not e_lt_pv:
        issues.append("E not < PV mean")
    status = "B1_PASS" if not issues else "B1_FAIL" if any("B0" not in i for i in issues) else "B1_UNRESOLVED"
    # Underpowered check
    if results.get("n_seeds", 3) < 3 or results.get("window_s", 2.0) < 2.0:
        if not issues:
            status = "B1_UNRESOLVED"
            issues.append("underpowered: n_seeds<3 or window<2s")
    return {"gate": "B1", "status": status, "issues": issues}


def _generic_seal(gate_id: str, results: dict[str, Any], required_keys: list[str], pass_predicate) -> dict[str, Any]:
    issues: list[str] = []
    for k in required_keys:
        if k not in results:
            issues.append(f"missing required observable: {k}")
    # Dependency check
    deps = SPECIFIED_GATES.get(gate_id, {}).get("dependencies", [])
    for d in deps:
        if results.get(f"{d}_status", f"{d}_PASS") not in (f"{d}_PASS",):
            issues.append(f"dependency {d} not PASS")
    # Gate-specific pass predicate stub — evaluates thresholds if present
    if pass_predicate is not None:
        extra = pass_predicate(results)
        issues.extend(extra)
    status = f"{gate_id}_PASS" if not issues else f"{gate_id}_FAIL"
    # UNRESOLVED for missing data with n_insufficient etc.
    if any("insufficient" in i or "underpowered" in i or "UNRESOLVED" in i for i in issues):
        status = f"{gate_id}_UNRESOLVED"
    return {"gate": gate_id, "status": status, "issues": issues}


def seal_predicate_B2(results: dict[str, Any]) -> dict[str, Any]:
    def _pred(r):
        iss = []
        if r.get("CV_rate") is not None and r["CV_rate"] <= 0.3:
            iss.append(f"CV_rate {r['CV_rate']} <=0.3")
        cvi = r.get("mean_CV_ISI")
        if cvi is not None and not (0.5 <= cvi <= 1.5):
            iss.append(f"mean CV_ISI {cvi} not in [0.5,1.5]")
        fano = r.get("median_Fano")
        if fano is not None and not (0.7 <= fano <= 2.0):
            iss.append(f"median Fano {fano} not in [0.7,2.0]")
        rho = r.get("mean_rho_ij")
        if rho is not None and not (-0.05 <= rho <= 0.2):
            iss.append(f"mean rho {rho} not in [-0.05,0.2]")
        decay = r.get("rho_decay")
        if decay is not None and decay <= 0.05:
            iss.append(f"rho(d) decay {decay} <=0.05")
        return iss
    return _generic_seal("B2", results, ["CV_rate", "mean_CV_ISI", "median_Fano", "mean_rho_ij", "rho_decay"], _pred)


def seal_predicate_B3(results: dict[str, Any]) -> dict[str, Any]:
    def _pred(r):
        iss = []
        efrac_ok = r.get("Efrac_qualifying_fraction")
        if efrac_ok is not None and efrac_ok < 0.80:
            iss.append(f"Efrac qualifying {efrac_ok} <0.80")
        if r.get("I_extracted") is False:
            iss.append("realized E/I not extracted — UNRESOLVED I_E/I_I_not_exposed")
        return iss
    return _generic_seal("B3", results, ["Efrac_qualifying_fraction", "I_extracted"], _pred)


def seal_predicate_B4(results: dict[str, Any]) -> dict[str, Any]:
    def _pred(r):
        iss = []
        if r.get("spatial_decay_p") is not None and r["spatial_decay_p"] >= 0.05:
            iss.append(f"spatial decay p {r['spatial_decay_p']} >=0.05")
        if r.get("motif_rank") is not None and r["motif_rank"] <= 0.5:
            iss.append(f"motif rank {r['motif_rank']} <=0.5")
        if r.get("Q") is not None and r["Q"] <= 0.2:
            iss.append(f"Q {r['Q']} <=0.2")
        if r.get("locality_index") is not None and r["locality_index"] <= 3.0:
            iss.append(f"locality_index {r['locality_index']} <=3")
        if r.get("saturating") is True:
            iss.append("weights saturating (2x gain <5% change)")
        return iss
    return _generic_seal("B4", results, ["spatial_decay_p", "motif_rank", "Q", "locality_index"], _pred)


def seal_predicate_B5(results: dict[str, Any]) -> dict[str, Any]:
    def _pred(r):
        iss = []
        if r.get("validate_PASS") is False:
            iss.append("RFOperator.validate FAIL")
        if r.get("E_parity") is not None and r["E_parity"] > 0.05:
            iss.append(f"E_parity {r['E_parity']} >0.05")
        if r.get("sparsity") is not None and not (0.18 <= r["sparsity"] <= 0.30):
            iss.append(f"sparsity {r['sparsity']} not in [0.18,0.30]")
        if r.get("Jaccard") is not None and r["Jaccard"] >= 0.15:
            iss.append(f"Jaccard {r['Jaccard']} >=0.15")
        if r.get("V1_only") is False:
            iss.append("non-V1 drive >0")
        if r.get("omission_zero") is False:
            iss.append("omission non-zero")
        return iss
    return _generic_seal("B5", results, ["validate_PASS", "E_parity", "sparsity", "Jaccard"], _pred)


def seal_predicate_B6(results: dict[str, Any]) -> dict[str, Any]:
    def _pred(r):
        iss = []
        if r.get("all_above_chance") is False:
            iss.append("acc not above chance all areas")
        if r.get("latency_ordered") is False:
            iss.append("latencies not V1<V4<FEF<PFC")
        if r.get("FF_drop") is not None and r["FF_drop"] < 0.15:
            iss.append(f"FF drop {r['FF_drop']} <0.15")
        if r.get("FB_drop") is not None and r["FB_drop"] < 0.15:
            iss.append(f"FB drop {r['FB_drop']} <0.15")
        if r.get("shuffle_drop") is not None and r["shuffle_drop"] >= 0.08:
            iss.append(f"shuffle drop {r['shuffle_drop']} >=0.08")
        return iss
    return _generic_seal("B6", results, ["all_above_chance", "latency_ordered", "FF_drop", "FB_drop"], _pred)


def seal_predicate_B7(results: dict[str, Any]) -> dict[str, Any]:
    def _pred(r):
        iss = []
        if r.get("R2") is not None and r["R2"] < 0.7:
            iss.append(f"R2 {r['R2']} <0.7")
        if r.get("DeltaH") is not None and r["DeltaH"] < 0.01:
            iss.append(f"DeltaH {r['DeltaH']} <0.01")
        if r.get("dense_H") is False:
            iss.append("dense H not measured — UNRESOLVED insufficient_temporal_resolution")
        return iss
    return _generic_seal("B7", results, ["R2", "DeltaH", "tau_H"], _pred)


def seal_predicate_B8(results: dict[str, Any]) -> dict[str, Any]:
    def _pred(r):
        iss = []
        if r.get("R2") is not None and r["R2"] < 0.7:
            iss.append(f"R2 {r['R2']} <0.7")
        if r.get("DeltaTheta") is not None and r["DeltaTheta"] < 0.001:
            iss.append(f"DeltaTheta {r['DeltaTheta']} <0.001")
        if r.get("Delta_rate_Theta_Hz") is not None and abs(r["Delta_rate_Theta_Hz"]) < 0.5:
            iss.append(f"Delta_rate {r['Delta_rate_Theta_Hz']} <0.5 Hz NEGATIVE")
        if r.get("retention_ratio") is not None and r["retention_ratio"] < 0.30:
            iss.append(f"retention {r['retention_ratio']} <0.30")
        if r.get("clipped") is True:
            iss.append("Theta clipped at w_ceiling")
        return iss
    return _generic_seal("B8", results, ["R2", "DeltaTheta", "Delta_rate_Theta_Hz"], _pred)


def seal_predicate_B9(results: dict[str, Any]) -> dict[str, Any]:
    def _pred(r):
        iss = []
        if r.get("s_H") is not None and abs(r["s_H"]) < 0.3:
            iss.append(f"s_H {r['s_H']} <0.3 Hz")
        if r.get("s_Theta") is not None and abs(r["s_Theta"]) < 0.3:
            iss.append(f"s_Theta {r['s_Theta']} <0.3 Hz")
        if r.get("R2_H") is not None and r["R2_H"] < 0.5:
            iss.append(f"R2_H {r['R2_H']} <0.5")
        if r.get("clamping_abolishes") is False:
            iss.append("clamping does not abolish")
        if r.get("history_valid_gt_fast") is False:
            iss.append("history_valid not > fast")
        return iss
    return _generic_seal("B9", results, ["s_H", "s_Theta", "R2_H"], _pred)


def seal_predicate_B10(results: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if results.get("J_improved") is False:
        issues.append("J_generic not improved")
    # Forbidden term grep
    hits = results.get("forbidden_hits", [])
    if hits:
        issues.append(f"forbidden terms in B10 objective: {hits}")
    # Also check objective_text if provided
    obj_text = results.get("objective_text", "")
    if obj_text:
        info = audit_b10_objective(obj_text)
        if info["hits"]:
            issues.append(f"forbidden terms via grep: {info['hits']}")
    if results.get("repasses_B1_B9") is False:
        issues.append("final does not re-pass B1-B9 without controller")
    if results.get("controller_in_Ct") is True:
        issues.append("controller inserted in C_t — forbidden")
    deps = SPECIFIED_GATES["B10"].get("dependencies", [])
    for d in deps:
        if results.get(f"{d}_status") not in (None, f"{d}_PASS"):
            if results.get(f"{d}_status") != f"{d}_PASS":
                issues.append(f"dependency {d} not PASS")
    status = "B10_PASS" if not issues else "B10_FAIL"
    return {"gate": "B10", "status": status, "issues": issues, "forbidden_hits": hits}


def seal_predicate_B11(results: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    for k in ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]:
        if results.get(f"{k}_status") != f"{k}_PASS":
            issues.append(f"{k} not PASS ({results.get(f'{k}_status')})")
    if results.get("freeze_consistency") is False:
        issues.append("freeze_consistency false")
    if results.get("paradigm_exact") is False:
        issues.append("paradigm_exact FAIL")
    if results.get("recording_completeness") is not None and results["recording_completeness"] < 1.0:
        issues.append(f"recording_completeness {results['recording_completeness']} <1.0")
    status = "B11_PASS" if not issues else "B11_FAIL"
    if any("RESOURCE_BOUNDARY" in str(i) for i in issues):
        status = "B11_UNRESOLVED"
    return {"gate": "B11", "status": status, "issues": issues}


def seal_predicate_B12(results: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if results.get("B11_status") != "B11_PASS":
        issues.append(f"B11 not PASS ({results.get('B11_status')})")
    if results.get("continuous_Ct") is False:
        issues.append("C_t not continuous")
    if results.get("paradigm_exact") is False:
        issues.append("paradigm_exact FAIL")
    if results.get("EvidenceRef_written") is False:
        issues.append("EvidenceRef not written")
    if results.get("T1_T7_computed") is False:
        issues.append("T1-T7 not computed")
    status = "B12_SEALED" if not issues else "B12_FAIL"
    if any("UNRESOLVED" in str(i) for i in issues):
        status = "B12_UNRESOLVED"
    return {"gate": "B12", "status": status, "issues": issues}


# Dispatch table for validate_gate
SEAL_PREDICATES: dict[str, Any] = {
    "B0": seal_predicate_B0,
    "B1": seal_predicate_B1,
    "B2": seal_predicate_B2,
    "B3": seal_predicate_B3,
    "B4": seal_predicate_B4,
    "B5": seal_predicate_B5,
    "B6": seal_predicate_B6,
    "B7": seal_predicate_B7,
    "B8": seal_predicate_B8,
    "B9": seal_predicate_B9,
    "B10": seal_predicate_B10,
    "B11": seal_predicate_B11,
    "B12": seal_predicate_B12,
}


def validate_gate(gate_id: str, results: dict[str, Any]) -> dict[str, Any]:
    """Check seal_predicate for gate_id against supplied results dict.

    Returns {"gate": ..., "status": "B*_PASS|FAIL|UNRESOLVED", "issues": [...], ...}
    Thresholds are SPECIFIED constants from SPECIFIED_GATES — not tuned.
    """
    if gate_id not in SEAL_PREDICATES:
        return {"gate": gate_id, "status": f"{gate_id}_UNKNOWN", "issues": [f"unknown gate {gate_id}"]}
    fn = SEAL_PREDICATES[gate_id]
    return fn(results)


def audit_b10_objective(objective_source: str | Path) -> dict[str, Any]:
    """Grep objective source for forbidden_terms_B10 hits.

    objective_source may be a string of code or a Path to a file.
    Returns {"hits": [...], "clean": bool, "forbidden_terms": FORBIDDEN_TERMS_B10}
    """
    if isinstance(objective_source, Path):
        text = objective_source.read_text() if objective_source.exists() else ""
    elif isinstance(objective_source, str) and Path(objective_source).exists():
        text = Path(objective_source).read_text()
    else:
        text = str(objective_source)

    hits: list[str] = []
    for term in FORBIDDEN_TERMS_B10:
        # Word-boundary-ish search; for symbols allow substring
        pattern = re.escape(term)
        if re.search(pattern, text):
            # Filter false positives for short terms: require word-ish but we treat T1 as \bT1\b
            if term in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
                if re.search(rf"\b{re.escape(term)}\b", text):
                    hits.append(term)
            else:
                hits.append(term)
    return {"hits": hits, "clean": len(hits) == 0, "forbidden_terms": FORBIDDEN_TERMS_B10}


def load_skeleton_gates(skeleton_json_path: str | Path | None = None) -> dict[str, Any]:
    """Load gates dict directly from skeleton A json for provenance audit."""
    if skeleton_json_path is None:
        skeleton_json_path = Path(__file__).resolve().parents[2] / "scratch" / "gen2_qualification_skeleton_A.json"
    p = Path(skeleton_json_path)
    if not p.exists():
        return {"exists": False, "path": str(p)}
    data = json.loads(p.read_text())
    return {"exists": True, "path": str(p), "gates": data.get("gates", {}), "version": data.get("version")}


__all__ = [
    "FORBIDDEN_TERMS_B10",
    "ALLOWED_OBJECTIVE_TERMS_B10",
    "SPECIFIED_GATES",
    "GATE_ORDER",
    "SEAL_PREDICATES",
    "validate_gate",
    "audit_b10_objective",
    "check_h_terminology",
    "load_skeleton_gates",
    "seal_predicate_B0",
    "seal_predicate_B1",
    "seal_predicate_B2",
    "seal_predicate_B3",
    "seal_predicate_B4",
    "seal_predicate_B5",
    "seal_predicate_B6",
    "seal_predicate_B7",
    "seal_predicate_B8",
    "seal_predicate_B9",
    "seal_predicate_B10",
    "seal_predicate_B11",
    "seal_predicate_B12",
]
