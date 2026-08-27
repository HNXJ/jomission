"""B1 class-conditional firing-rate targets — SPECIFIED priors.

Source: scratch/gen2_qualification_skeleton_A.md B1 candidate_metric +
        scratch/gen2_qualification_skeleton_A.json B1.candidate_metric
Spec version: BIOPHYSICAL_QUALIFICATION_v0.1
Provenance: MODEL_ASSUMPTION / ratification-pending LITERATURE_PRIOR.
These are qualification targets (Gen-2 model property), NOT empirical constants.
Do not tune post-hoc; any change requires a Ledger entry and B0 re-audit.

NOTE: B1 per-class numbers need PI ratification before B1 seal (see skeleton A ambiguities).
Freeze these values before B1 measurement; they are priors for evidence grading.
"""

from __future__ import annotations

# SPECIFIED class-conditional targets (Hz), as distribution centres with broad tolerance.
# Tolerance for PASS is ±50% per B1 spec, with global μ ∈ [3,15] Hz and >60% coverage.
# Classes: E slow, PV fast, SST intermediate, VIP intermediate-high.
B1_CLASS_TARGETS: dict[str, dict] = {
    "E": {
        "label": "excitatory",
        "global_target_Hz": (5.0, 8.0),
        "per_layer_target_Hz": {
            "V1_L4_E": (6.0, 7.0),
            "V1_L2/3_E": (4.0, 6.0),
            "V1_L5_E": (7.0, 10.0),
            "V1_L6_E": (5.0, 8.0),
            "V4_E": (5.0, 8.0),
            "FEF_E": (3.0, 6.0),
            "PFC_E": (3.0, 6.0),
        },
        "provenance": "MODEL_ASSUMPTION — ratification-pending",
        "note": "5-10 Hz as distribution, not universal clamp; E<PV mean required",
    },
    "PV": {
        "label": "parvalbumin",
        "global_target_Hz": (12.0, 25.0),
        "per_layer_target_Hz": {},
        "provenance": "MODEL_ASSUMPTION — ratification-pending",
        "note": "fast-spiking inhibitory; must exceed E mean",
    },
    "SST": {
        "label": "somatostatin",
        "global_target_Hz": (6.0, 12.0),
        "per_layer_target_Hz": {},
        "provenance": "MODEL_ASSUMPTION — ratification-pending",
        "note": "intermediate rate",
    },
    "VIP": {
        "label": "vasoactive intestinal peptide",
        "global_target_Hz": (8.0, 15.0),
        "per_layer_target_Hz": {},
        "provenance": "MODEL_ASSUMPTION — ratification-pending",
        "note": "intermediate-high, disinhibitory role",
    },
}

# Derived convenience constants (SPECIFIED, not tuned)
B1_GLOBAL_MU_PASS_HZ = (3.0, 15.0)
B1_COVERAGE_THRESHOLD = 0.60  # fraction of pops within class target ±50%
B1_PATHOLOGICAL_SILENCE_HZ = 0.5  # E <0.5 Hz pathological
B1_PATHOLOGICAL_HYPER_HZ = 60.0  # any class mean >60 Hz pathological
B1_FAIL_HYPER_HZ = 80.0  # any area/layer mean >80 Hz FAIL

__all__ = ["B1_CLASS_TARGETS", "B1_GLOBAL_MU_PASS_HZ", "B1_COVERAGE_THRESHOLD"]
