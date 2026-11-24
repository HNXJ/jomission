# 00 — EXECUTIVE SCIENTIFIC SUMMARY — Jomission RF×Rate Characterization

_Consolidates `results/rf_rate_factorial/dossier/` reports 01–10 into a single characterization statement. All numbers below are traceable to a numbered report (`R01`–`R10`), a sealed manifest, and a generated-owner array; evidence tags follow the dossier-wide grammar: `MEASURED` · `DERIVED` · `MODEL_ASSUMPTION` · `INFERENCE` · `UNRESOLVED`. Polarities follow the frozen discipline: **NEGATIVE** = phenomenon reproduced-test and found absent/opposite; **UNRESOLVED** = not measurable/not executed and never rendered as negative._

_Scope._ This packet characterizes the **RF (receptive-field plasticity, K_HDP on `w`) × Rate (plasticity timescale, `tau_0_ms`) 2×2 factorial** on the Jomission network (`V1→V4→FEF→PFC`, 100 neurons/area, izhikevich). Full design: `manifests/rf_rate_factorial_design.json` (`rf_rate_factorial.v0.1.0`, frozen pre-execution). Consumed dossier: `01_BIOPHYSICS.md` … `10_JDNA_PSEUDOGENOME_ARCHITECTURE.md` and `dossier_receipt.json` (hash-verified: cell A input integrity `OK`, `hash_verified_all=true`).

---

## 1. What was executed

| cell (dir) | RF | Rate | K_HDP | tau_0_ms | hp_hash | lifecycle |
|---|---|---|---|---|---|---|
| A_RFoff_RateRef | off | standard (ref) | 0.0 | 5.0 | `f327f9d2ad64cc88` | **COMPLETE** |
| B_RFon_RateRef | on | standard | 0.003 | 5.0 | `f327f9d2ad64cc88` | **execution_failure → UNRESOLVED** |
| C_RFoff_RateSlow | off | slow (LONG) | 0.0 | 1000.0 (200×) | `f72a489841810a4b` | **COMPLETE** |
| D_RFon_RateSlow | on | slow | 0.003 | 1000.0 | `b326f7201c59b803` | **execution_failure → UNRESOLVED** |

- Factor isolation verified at the **config level**: B−A = RF only (same hp `f327`), C−A = Rate only (RF off, hp `f327→f72a`), D−A = RF+Rate (R08 §2, `MEASURED`).
- **B and D ran the full trajectory to terminal step 20900480 (post 95/96) but crashed at the save step** (`TypeError` at `spikes_example`): no `result.json`/`EvidenceRef.json`/`artifact_hashes.json` and no post recordings. Per frozen rule this is an **execution failure → UNRESOLVED, NOT a scientific negative** (R08 §5, R09 §3, `rf_rate_factorial_repro_review.json`).
- A and C full-lifecycle artifacts are hash-verified against sealed manifests (R08 §5; `A_hashes`/`C_hashes` match).

**Consequence:** the **full RF×Rate 2-way factorial (rate-level Δ_exposure, field low-gamma, recovery, interaction) is UNRESOLVED** — D is missing and B/D lost post phenotype recordings. Only the **partial H-phenotype** and A/C single-cell characterizations are measurable (R08 §3–4).

---

## 2. Major positive findings

1. **T5 — gamma↔rate coupling exceeds the lower-frequency aggregate (POSITIVE, specificity-limited).** Across the hierarchy, gamma (30–80 Hz) vs a lower-frequency aggregate couples with per-area r 0.96–0.99 (p<1e-12, `MEASURED`, sealed T1-T7). **Specificity caveat:** gamma-vs-beta alone is NEGATIVE (−0.006…−0.024) and within-condition r≈0 (drive-confounded); the frozen aggregate comparison is POSITIVE, specificity NEG (R05 §2, `manifests/t1_t7_final_corrected.json`).
2. **T7 — no strong fixed between-area lead/lag (POSITIVE for the absence hypothesis).** 18/18 per-position |mean|<0.5 ms, p>0.017, `has_fixed=False`; positive controls MAE0/within5 = 1.0, null-p 0.247, FPR 0 (R05 §2, sealed T1-T7).
3. **Rate-only LONG intervention achieved exposure-scale H drift (mechanism works).** At reference rate H is stationary in exposure (cell B: exposure_H_std 7.14e-5, τ_eff,E ≈ 4.1 s design / 3.53 s fitted). Under the 200× `tau_0_ms`=1000 ms intervention (cell C) H decays across the full 1202.24 s exposure with τ_eff ≈ 1093 s measured (design 833 s, ratio 1.31, 236 samples/e-fold; `EXPLORATORY` fit) — the intended ×3-rejected / ×200-adopted move from second-scale saturation to exposure-scale drift (R07 §1,4–5).
4. **RF raises the H operating point; LONG lowers it (measured H-mean contrasts, n=1/cell, no SE).** β_RF_h = **+0.033** (B−A: H_mean 1.06209 vs 1.02897); β_τ_h = **−0.015** (C−A: H_mean 1.01390 vs 1.02897). These are `MEASURED` 2-point magnitudes, **descriptive only, no inferential p-value**; interaction β_RF×τ_h = **UNRESOLVED (D missing)** (R08 §3, `rf_rate_factorial_estimator.json`).
5. **The implemented plasticity timescale hierarchy is now correctly identified.** Declared `tau_theta_s`=1000 s is a **dead key never consumed** by the engine; the realized hierarchy is `τ_X (ms) << τ_H (0.03–4 s) << τ_w (≈1–2.5 s)`, satisfying the weak `≫τ_X` clause but violating the exposure-scale design at reference rate — the audit that motivated the LONG level (R07 §1).

---

## 3. Major negative findings

1. **No habituation at the rate level (Δ rate NEG).** The intact-stimulus response does not decrement over 260 exposure trials: frozen canonical Δ_exposure rate V1 +0.0017, V4 −0.0026, FEF +0.0011, PFC −0.0003 Hz, all ≪ 0.5 Hz threshold (R04 §2, `manifests/canonical_science_seal.json`).
2. **Q8 — no predictive/history encoding in H or Θ at the rate level (NEGATIVE).** Replacing post→pre state for each carrier (3-seed, dt0.1, 4624 ms) leaves the omission-probe rate unchanged: |Δ| < 0.01 Hz for H, Θ, H+Θ, fast_X (all p>0.6, Cohen d ≤0.31) — **field/recovery UNRESOLVED** (R05 §5, R09 §2, `q8_canonical_closure_dt01.json`).
3. **T1 — no sparse omission spiking.** 0/400 units omission-selective; the omission response is a uniform −15.6 Hz population decrement, not selective (R04 §4, sealed T1-T7).
4. **T2 — no higher-order frontal bias** (0 units to enrich; T1 absent) **NEG**.
5. **T3 — V1 omission response is strong, not weak** (V1 −2.65 Hz, d>0.2) — the "weak V1 omission spiking" hypothesis **NEG**.
6. **T4 — no frontal low-gamma omission enhancement**; omission low-gamma < intact at all areas/positions (uniform suppression) **NEG**.
7. **T6 — weaker coupling of omission-selective units** fails because there are no omission-selective units (T1 absent) **NEG**.
8. **The omission effect does not become predictive across exposure.** It neither habituates nor reverses; it slightly deepens (≈ −0.1…−0.2 Hz, A n=1; C n=8 paired −0.12…−0.16 Hz) (R04 §3, R05 §3).

---

## 4. T1–T7 state (sealed)

| Target | Hypothesis | Polarity | Finding (abridged) | Tag |
|---|---|---|---|---|
| T1 | sparse omission spiking | **NEG** | 0/400 selective; uniform −15.6 Hz decrement | MEASURED |
| T2 | higher-order/frontal bias | **NEG** | 0 units to enrich; frontal bias absent (T1 absent) | MEASURED |
| T3 | weak V1 omission spiking | **NEG** | V1 −2.65 Hz, strong not weak (d>0.2) | MEASURED |
| T4 | frontal low-gamma enhancement | **NEG** | low_gamma omission < intact all areas/pos; no frontal 20–50 Hz increase | MEASURED |
| T5 | gamma↔rate > lower-frequency aggregate | **POS (specificity-limited)** | gamma r 0.96–0.99 p<1e-12 vs lower +1.04–1.13; gamma-vs-beta NEG, within-condition r≈0 (drive-confounded) | MEASURED |
| T6 | weaker coupling of T1 units | **NEG** | no T1 units → no distribution per frozen rule | MEASURED |
| T7 | fixed between-area lead/lag | **POS (absence)** | 18/18 \|mean\|<0.5 ms, p>0.017, has_fixed False; positive controls clean | MEASURED |

Source: `manifests/t1_t7_final_corrected.json` (sealed), mirrored in R05 §2.

---

## 5. Q1–Q15 state

The workspace does not carry a machine-readable Q1–Q15 enumerator; the only structured record is the **pilot** `manifests/mini_full_seal.json`, which marks every question `UNRESOLVED` pending the full 260×1202 s / field-level battery. The present RF×Rate packet authoritatively adjudicates the rate-level subset it covers; the remainder stay `UNRESOLVED` (not executed / not measurable from retained artifacts). **State legend: NEG = tested and absent/opposite; UNRESOLVED = not adjudicated, never negative.**

| Q | State (this packet) | Basis |
|---|---|---|
| Q1 | **NEG (rate-level)** | Δ rate across exposure ≈ 0 → no firing-rate habituation (R04 §2, R05 §3) |
| Q2 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q3 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q4 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q5 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q6 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q7 | **NEG (pilot)** | pilot Δ_exposure −5.07 Hz not significant at pilot scale; superseded direction pending FULL |
| Q8 | **NEG (rate) / UNRESOLVED (field)** | H/Θ state replacement rate Δ ≈ 0, all p>0.6 (canonical n=3); field/recovery pending (R05 §5, R09 §2) |
| Q9 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q10 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q11 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q12 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q13 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q14 | UNRESOLVED | requires full multi-replicate/field battery; not executed |
| Q15 | UNRESOLVED | requires full multi-replicate/field battery; not executed |

_Design completion rule:_ a NEGATIVE result is still COMPLETE if C1–C7 hold; failures here are evidence about the model (R08, `rf_rate_factorial_design.json` `completion_predicate`).

---

## 6. RF×Rate factorial result

- **Full 2-way ANOVA (Y ~ RF + Rate + RF:Rate) on rate/field phenotypes: UNRESOLVED** — D missing (interaction), B/D post recordings lost. `INCOMPLETE ≠ NEGATIVE` (R08 §4).
- **Partial H-phenotype (MEASURED, n=1/cell):**
  | contrast | estimate | tag |
  |---|---|---|
  | β_RF_h (B−A) | **+0.033** | MEASURED (2-point) |
  | β_τ_h (C−A) | **−0.015** | MEASURED (2-point) |
  | β_RF×τ_h (interaction) | **UNRESOLVED** (D missing) | UNRESOLVED |
- **Rate-level Δ_exposure / field low-gamma / recovery / slot-resolved rate Δ: UNRESOLVED** (B,D unrecoverable; A/C slot-resolved primary estimands NOT_RECOVERABLE from ledger) (R08 §4, R09 §3).

---

## 7. Limitations (drives the failure atlas, R11)

1. **B/D save crash** → full factorial and post phenotypes `UNRESOLVED` (never NEGATIVE).
2. **A-cell RF-off identity not verified at hp level.** Executed hp_hash is the canonical RF-ON `f327f9d2ad64cc88`, not the design's RF-off `bb8277e7a8e0bca2`; no `K_HDP=0` override appears in the visible executor; effective K_HDP during the run is `UNRESOLVED` (R01 §5, R04 §6, R06 §8).
3. **Spike timestamps not retained** → no ISI/temporal CV, no slot-resolved (PSTH) omission effect, no temporal-synchrony check; per-unit whole-trial counts show strong within-class degeneracy (all 242 E units identical counts) (R01 §3, R02 §2b, R03 §6).
4. **Field proxy uncalibrated** (`physical_amplitude_calibrated=False`) → all power arbitrary units, only relative/log claims; `ex_AAAB` field is byte-identical to `pre_0000` (pre-exposure duplicate), so pre-vs-post field contrast collapses (R03 header, §9).
5. **H/Θ dense trajectory not materialized** → effective H/Θ timescale and per-trial H series for A `UNRESOLVED`; recorded Θ = 0.003267 sits **below w_floor = 0.01** (bound violation flagged `UNRESOLVED`) (R01 §4, R04 §5, R07 §6).
6. **T4/T5 spectral contrasts single-replicate (n=1)** → underpowered, polarity `UNRESOLVED`/descriptive only (R03 §3–6). Band-convention discrepancy (companion 20–50 Hz vs frozen 30–50 Hz) flagged `UNRESOLVED` at provenance (R03 §4).
7. **No SSA control condition** in the battery → SSA `UNRESOLVED`/NOT TESTABLE; observed decrement is a uniform drive-offset, not adaptation (R04 §1,§4).
8. **Q8 frozen-hash drift** — seed 1/2 config_hash differ from canonical (`frozen_ok=false`); rate inference unaffected, hash-level drift `UNRESOLVED` (R05 §5).
9. **Caveat C-cell startup relaxation** in PRE ordered by battery position confounds naive pre-vs-post contrasts (R04 §2).
10. **All rate/field contrasts single-replicate or underpowered at the factorial level** — inferential statistics require n≥4 per design; the H-mean contrasts carry no SE (R08 §6, R03 §6).

---

## 8. Bottom line

- **The model reproduces an omission response** — a large, uniform, population-wide drive-offset (−2.6…−3.2 Hz/area) with **no selective, no frontal-predictive, and no fixed-timing structure**; it does not habituate or become predictive across exposure, and neither H nor Θ stores exposure history at the rate level (R05).
- **RF and Rate both modulate the plasticity fixed-point** (RF on raises H +0.033; LONG rate lowers H −0.015, and the rate-only knob successfully moves H from second-scale saturation to exposure-scale drift), but the **interaction and all phenotype-level factorial verdicts are UNRESOLVED** by the B/D save crash.
- **The negative/habitual T1–T4/T6 findings, the specificity-limited T5, the absence-type T7, and Q8-rate NEGATIVE** are the robust, sealed results of this packet; every `UNRESOLVED` (B/D crash, field level, interaction, SSA) is a measurement/execution gap, not evidence of absence.
