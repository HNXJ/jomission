# 04 — HABITUATION & STIMULUS-SPECIFIC ADAPTATION (cell A_RFoff_RateRef, cross-cell C)

_Characterization dossier. Habituation (repetition decrement) and SSA (stimulus-specific adaptation) are treated as HYPOTHESES to test against the sealed T1-T7 evidence and the recorded A-cell (and C-cell slow-rate) data. Evidence tags: `MEASURED` (read/computed from a recorded array) · `DERIVED` (from MEASURED) · `MODEL_ASSUMPTION` (design/config-specified) · `INFERENCE` (interpretation) · `UNRESOLVED` (missing from artifacts; never imputed)._

- Sources: `manifests/t1_t7_final_corrected.json`, `manifests/t1_t7_seal.json`, `manifests/canonical_science_seal.json`, `results/rf_rate_factorial/A_RFoff_RateRef/E_A_analysis.json`, `results/rf_rate_factorial/C_RFoff_RateSlow/E_C_analysis.json`, A-cell `recording/` artifacts.
- Core principle: **repetition decrement ≠ SSA**. A population-wide decrement under repeated stimulation is, at best, habituation; calling it stimulus-specific adaptation requires a specificity control (e.g., rare-standard/deviant contrast with matched physical parameters), which this paradigm does not implement.

## 1. Hypothesis framing

| hypothesis | definition tested here | status |
|---|---|---|
| Habituation | repetition decrement of the intact-stimulus firing-rate response across the 260-trial exposure | NEGATIVE at rate level (no decrement); within-exposure trajectory `UNRESOLVED` for A |
| SSA | stimulus-specific (selective) adaptation to the repeated stimulus | NOT TESTABLE from artifacts — no oddball/equiprobable specificity control exists; no selective units found |

SSA is declared `UNRESOLVED` at the artifact level: no standard/deviant or equiprobability control is present in the 12-condition battery [MODEL_ASSUMPTION, design battery `rf_rate_factorial.v0.1.0`], so no response decrement can be attributed to stimulus specificity.

## 2. Habituation: tonic/response rate across exposure

The intact-condition (AAAB) whole-trial rate is essentially unchanged pre→post exposure (A-cell example trials, n=1):

| area | pre (Hz) | post (Hz) | Δ (Hz) | tag |
|---|---|---|---|---|
| V1 | 22.065 | 22.067 | +0.002 | MEASURED / DERIVED |
| V4 | 23.475 | 23.482 | +0.006 | MEASURED / DERIVED |
| FEF | 23.082 | 23.062 | −0.019 | MEASURED / DERIVED |
| PFC | 25.469 | 25.474 | +0.004 | MEASURED / DERIVED |

Frozen canonical Δ_exposure rate (sealed, dt1.0 supplementary replay battery): V1 +0.0017, V4 −0.0026, FEF +0.0011, PFC −0.0003 Hz [MEASURED from `manifests/canonical_science_seal.json` / `t1_t7_seal.json`]; polarity NEGATIVE at rate level (|Δ| ≪ 0.5 Hz threshold) [MODEL_ASSUMPTION threshold from design].

- **No tonic/response habituation:** the intact-stimulus response does not decrement over 1202.24 s / 260 exposure trials. Habituation hypothesis NEGATIVE at the rate level [INFERENCE from MEASURED].
- Caveat (A cell): pre/post deltas are single-replicate, example-trial descriptors; the C-cell (slow rate) analysis flags a **startup relaxation** in PRE ordered by battery position (trial-1 H rises to a peak ≈ trial 23), so early-battery pre conditions carry a relaxation offset [MEASURED from `E_C_analysis.json` `confound_flag`]. This confounds any naive pre-vs-post contrast and is `UNRESOLVED` at the A-cell level (per-trial rates not retained for A).
- A-cell per-trial rate trajectory across exposure: `UNRESOLVED` — `rate_summary.json` retains only 5 pre-trial samples; no per-exposure-trial rate array exists. Within-exposure monotonic decrement can therefore be neither confirmed nor excluded from A artifacts.

## 3. Habituation of the omission response

The omission effect is present in the **fresh (pre-exposure)** battery and does not shrink with exposure:

| contrast (p2, A context) | pre omission effect (Hz) | post omission effect (Hz) | Δ_omission_effect (Hz) | tag |
|---|---|---|---|---|
| V1 AXAB−AAAB | −2.641 | −2.827 | −0.186 | DERIVED (n=1 example) |
| V4 AXAB−AAAB | −3.056 | −3.246 | −0.190 | DERIVED (n=1 example) |
| FEF AXAB−AAAB | −2.889 | −3.058 | −0.169 | DERIVED (n=1 example) |
| PFC AXAB−AAAB | −3.186 | −3.285 | −0.099 | DERIVED (n=1 example) |

C-cell (slow rate, n=8 paired) Δ_omission_effect at p2: V1 −0.155, V4 −0.135, FEF −0.127, PFC −0.124 Hz [MEASURED from `E_C_analysis.json` `omission_vs_intact_per_position`; polarity sign-only `UNRESOLVED`].

- The omission decrement neither habituates (moves toward zero) nor is learned (does not reverse) across exposure; it slightly **deepens** (≈ −0.1 to −0.2 Hz). If exposure made the omission predictable, a predictive-state account would predict attenuation of the decrement; none is observed [INFERENCE from DERIVED].

## 4. SSA: specificity

- T1 (sealed): **0/400 units** omission-selective (omission > intact +2 Hz) pre and post; the omission decrement is a **uniform −15.6 Hz population decrement, not selective** [MEASURED from `t1_t7_final_corrected.json`]. Uniform, non-selective decrement is the opposite of an SSA signature (which requires a selective subpopulation) [INFERENCE].
- Per-class uniformity: within class, per-unit whole-trial counts are degenerate (all 242 E units identical counts; PV/SST near-uniform) [MEASURED, see dossier 02 §2b]. No firing-rate heterogeneity exists within E that could carry a stimulus-specific adapted subpopulation [INFERENCE from MEASURED].
- No SSA control condition (equiprobable standard vs deviant; matched physical stimulus, differing probability) exists in the design battery [MODEL_ASSUMPTION, `rf_rate_factorial_design.json` C3 conditions]. SSA-specificity therefore cannot be assessed: `UNRESOLVED` (not testable), and the observed decrement is best explained as a **uniform drive-offset effect** of the zero-drive omission slot, not adaptation [INFERENCE].

## 5. Hidden-state (H/Θ) evolution vs habituation

- A cell: retained H/Θ trajectories are 5 constant points (H = 1.028966, Θ = 0.003267) [MEASURED]; no per-trial H series → within-exposure H habituation `UNRESOLVED` for A.
- C cell (slow rate, τ_0 = 1000 ms): per-trial H (452 points) shows startup rise (H 1.0027 → peak 1.0179 at trial ≈23) then a slow monotone decay through exposure and post; exploratory exponential fit τ_eff ≈ 1092 s (full run) / 1124 s (exposure+post), residual σ ≈ 2.5e-4; H exposure ptp ≈ 5.3e-3, exposure→post shift −1.8e-3 [MEASURED from `E_C_analysis.json` `H_Theta_observables`]. Design reference τ_eff ≈ 833 s [MODEL_ASSUMPTION]. This is a slow **hidden-state relaxation**, and because Δ rate ≈ 0 it does not manifest as firing-rate habituation — the H decay is rate-silent [INFERENCE].
- Θ in both A and C is constant to machine precision (K_HDP = 0, RF-off) [MEASURED]; Θ = 0.003267 (A) / 0.003238 (C) sits **below w_floor = 0.01** — flagged `UNRESOLVED` (same flag as dossier 01 §4).

## 6. Cross-cell note

- Rate-level factor isolation for habituation is `UNRESOLVED`: B and D lost post recordings to the executor save crash (execution failure, not a scientific negative) per `manifests/rf_rate_factorial_repro_review.json`; only A and C carry complete lifecycles.
- The rate-slow cell C shows a resolvable slow H decay (τ ≈ 1100 s) whereas the rate-standard A-cell H trajectory was not retained at per-trial resolution — so a rate × habituation contrast is not computable [UNRESOLVED].
- Provenance caveat (carried from dossier 01 §5): A-cell executes with hp_hash `f327f9d2ad64cc88` (canonical) rather than the design's RF-off hash `bb8277e7a8e0bca2`; effective K_HDP during the run is `UNRESOLVED`, so the "RF off/on" label is not independently verified at the hp level.

## 7. Number registry

All numbers appear in the tables above with evidence tag and source. Missing quantities (A-cell per-trial rate series, SSA control condition, per-trial H for A) are `UNRESOLVED` and are not imputed.