# 05 — OMISSION & PREDICTIVE STATE (cell A_RFoff_RateRef; Q8 state replacement)

_Characterization dossier. Omission is operationalized as a zero-drive stimulus slot with preserved timing; the predictive-state question is whether the HDP carriers H and Θ store exposure/omission history. Primary authorities: sealed T1-T7 (`manifests/t1_t7_final_corrected.json`), Q8 canonical closure (`manifests/q8_canonical_closure_dt01.json`), frozen Δ_exposure (`manifests/canonical_science_seal.json`), and A-cell `recording/` artifacts. Tags: `MEASURED` · `DERIVED` · `MODEL_ASSUMPTION` · `INFERENCE` · `UNRESOLVED`._

## 1. Omission operationalization

| quantity | value | tag | source |
|---|---|---|---|
| omission slot (p2) | abs [1031, 1562] ms, rel [0, 531] ms | MODEL_ASSUMPTION | q8 closure / event_ledger |
| AXAB p2-slot drive (sum, max) | 0.0 / 0.0 (all zero) | MEASURED | `external_drive_examples.npz` [10310:15620] |
| AAAB p2-slot drive (sum, max) | 10 620 000 / 5.0 | MEASURED | same |
| AXAB vs AAAB p1 drive | identical (10 620 000) | MEASURED | same → timing preserved |
| AXAB whole-trial drive mean | 1.7225 (vs AAAB 2.2967) | MEASURED | same |

Omission = **zero visual drive, identical timing**; any activity in the omission slot is network-intrinsic (recurrent), not stimulus-driven [INFERENCE from MEASURED].

## 2. Omission phenomenology (T1-T7 sealed)

Whole-trial rates (example trials, A cell): AAAB 23.521 Hz global (V1 22.067, V4 23.482, FEF 23.062, PFC 25.474) vs AXAB 20.417 Hz global (V1 19.241, V4 20.236, FEF 20.004, PFC 22.189) — a uniform −2.6 to −3.2 Hz/area decrement [MEASURED].

| target | polarity | finding (sealed) | tag |
|---|---|---|---|
| T1 sparse omission spiking | NEGATIVE | 0/400 units omission-selective; uniform −15.6 Hz decrement, not selective | MEASURED |
| T2 higher-order bias | NEGATIVE | 0 units to enrich; frontal bias absent (T1 absent) | MEASURED |
| T3 weak V1 omission spiking | NEGATIVE | V1 −2.65 Hz (19.4 vs 22.1), strong not weak (d>0.2) | MEASURED |
| T4 frontal low-gamma enhancement | NEGATIVE | low_gamma omission < intact at all areas/positions; no frontal 20–50 Hz increase | MEASURED |
| T5 gamma–rate coupling | POSITIVE (specificity-limited) | gamma vs lower-frequency aggregate +1.04–1.13/area (r 0.96–0.99, p<1e-12), but gamma-vs-beta alone NEG (−0.006…−0.024) and within-condition r≈0 (drive-confounded) | MEASURED |
| T6 weaker coupling of T1 units | NEGATIVE | no T1 units → no distribution per frozen rule | MEASURED |
| T7 fixed between-area lead/lag | POSITIVE (absence) | 18/18 per-position \|mean\|<0.5 ms, p>0.017, has_fixed False; positive controls MAE0 within5=1.0, null p 0.247 FPR 0 | MEASURED |

The omission response is large, uniform, population-wide, and carries **no selective, no frontal-predictive, and no fixed timing structure** [INFERENCE from MEASURED].

## 3. Δ_exposure (does omission response change with exposure?)

Frozen canonical Δ_exposure rate: V1 +0.0017, V4 −0.0026, FEF +0.0011, PFC −0.0003 Hz [MEASURED from `manifests/canonical_science_seal.json`]; interpretation NEGATIVE at rate level [MODEL_ASSUMPTION threshold 0.5 Hz].

Omission-effect change pre→post (descriptive, A context p2): V1 −0.186, V4 −0.190, FEF −0.169, PFC −0.099 Hz (n=1 example); C cell (slow, n=8 paired): V1 −0.155, V4 −0.135, FEF −0.127, PFC −0.124 Hz, polarity sign-only `UNRESOLVED`. The omission effect does not attenuate with exposure → no acquired prediction of omission [INFERENCE].

## 4. Predictive-state carriers: H and Θ

| quantity | value | tag | source |
|---|---|---|---|
| A-cell H (retained 5 pts) | all 1.028966 | MEASURED | `H_t_dense.npz` |
| A-cell H run extrema (min/max/mean) | 0.9838 / 1.0901 / 1.0290 | MEASURED | `A_RFoff_RateRef_result.json` |
| A-cell Θ (retained 5 pts) | all 0.003267 | MEASURED | `H_t_dense.npz` |
| Θ vs w_floor 0.01 | below floor (0.00327) → `UNRESOLVED` | MEASURED vs MODEL_ASSUMPTION | dossier 01 §4 |
| A-cell effective H/Θ timescale | `UNRESOLVED` (dense trajectory not materialized) | — | dossier 01 §4 |

## 5. Q8 canonical state replacement (3-seed, dt0.1, 4624 ms)

Question: *does H or Θ store exposure history despite Δ rate ≈ 0?* Method: post state replaced by pre state for each carrier, omission-probe (AXAB) run with matched RNG; rate Δ per seed. Full numbers in `manifests/q8_canonical_closure_dt01.json`:

| carrier | rate Δ mean (Hz) | SD (Hz) | p | Cohen d | polarity n=3 | tag |
|---|---|---|---|---|---|---|
| H (H_post→H_pre) | +0.00054 | 0.00585 | 0.888 | 0.092 | NEGATIVE | MEASURED |
| Θ (Theta_post→Theta_pre) | +0.00126 | 0.00413 | 0.650 | 0.305 | NEGATIVE | MEASURED |
| H+Θ | −0.00036 | 0.00399 | 0.890 | −0.090 | NEGATIVE | MEASURED |
| fast_X (fast control) | −0.00072 | 0.00326 | 0.739 | −0.221 | NEGATIVE | MEASURED |
| history_valid (H+Θ vs fast) | −0.00036 | 0.00399 | 0.890 | −0.090 | NEGATIVE | MEASURED |

- Technical validity: continuous state (X, H, Θ, D, RNG, cursor) preserved except declared carrier; matched_RNG preserved; delay D = None verified [MEASURED]. Frozen-hash caveat: seed 1 and seed 2 `config_hash` differ from canonical `4f9fdeae7428199a` (declared `frozen_issues`; `frozen_ok=false`) → hash-level drift `UNRESOLVED`, rate inference unaffected.
- **Interpretation:** replacing post H, Θ, or both with their pre values does not move the omission-probe rate (|Δ| < 0.01 Hz ≪ 0.5 Hz threshold, all p > 0.6). Neither H nor Θ encodes exposure history at the rate level [INFERENCE from MEASURED].
- Field/recovery: `UNRESOLVED` — Q8 closure explicitly leaves area-resolved spectral (field) phenotype pending; window (4624 ms) now satisfies the 500 ms+ requirement but field evaluation was not part of this closure [MEASURED from q8 manifest `field_claim`].

## 6. Synthesis

The model's omission response is a large, uniform drive-offset that (i) is present in the naive state, (ii) does not habituate or become predictive across exposure (Δ_exposure ≈ 0, omission effect stable-to-deeper), (iii) is not carried by H or Θ at the rate level (Q8 NEGATIVE for all carriers), and (iv) has no fixed between-area timing structure (T7 POSITIVE for absence). There is no evidence of a predictive/anticipatory state encoding in the recorded phenotypes; the omission response is best characterized as bottom-up drive removal [INFERENCE, claim level: rate-phenotype; field level UNRESOLVED].

## 7. Number registry

All numbers appear in tables with evidence tag and source. Field-level Q8 evaluation, dense H/Θ time bases, and A-cell per-trial H series are `UNRESOLVED` and are not imputed.