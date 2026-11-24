# 08 — RF × RATE FACTORIAL (2×2 estimator, partial H-phenotype)

_Characterization dossier. This report records the executed 2×2 factorial (RF off/on × Rate standard/slow) estimator and its **partial** H-phenotype, and explicitly separates what is `MEASURED` from what is `UNRESOLVED` due to the B/D save crash. Sources: `manifests/rf_rate_factorial_estimator.json`, `manifests/rf_rate_factorial_repro_review.json`, `manifests/four_cell_lifecycle_seal.json`, `jomission/ablations/rf_rate_factorial.py` (frozen design), `E_A/E_B/E_C_analysis.json`._

---

## 1. Design recap (frozen, `rf_rate_factorial.py`)

| cell | design | RF | Rate | K_HDP | tau_0_ms | hp_hash | status |
|---|---|---|---|---|---|---|---|
| A_RFoff_RateRef | A_RFoff_RateStd | off | standard | 0.0 | 5.0 | f327f9d2ad64cc88 | **COMPLETE** |
| B_RFon_RateRef | C_RFon_RateStd | on | standard | 0.003 | 5.0 | f327f9d2ad64cc88 | execution_failure → **UNRESOLVED** |
| C_RFoff_RateSlow | B_RFoff_RateSlow | off | slow | 0.0 | 1000.0 | f72a489841810a4b | **COMPLETE** |
| D_RFon_RateSlow | D_RFon_RateSlow | on | slow | 0.003 | 1000.0 | b326f7201c59b803 | execution_failure → **UNRESOLVED** |

Directory names (A/B/C/D_Rate*Ref) differ from frozen manifest cell names; factor-level identity is established from executor assertions + hp_hash (per `E_*_analysis.json`) [MEASURED]. RF off = K_HDP 0.0 (mechanism toggle, moves w* to baseline — intentional); Rate slow = `tau_0_ms` 1000 (200×, rate-only, preserves fixed points/bounds).

## 2. Factor isolation (config level) — VERIFIED

| contrast | change | isolation | tag |
|---|---|---|---|
| B−A | RF off→on, hp same f327 | RF only | MEASURED (config-level) |
| C−A | rf off, hp f327→f72a | Rate only | MEASURED (config-level) |
| D−A | rf off→on + hp change | RF+Rate | MEASURED (config-level) |

## 3. H_mean cell means & partial contrasts (from `rf_rate_factorial_estimator.json`)

| cell | H_mean (global) | tag |
|---|---|---|
| A (RFoff, RateStd) | 1.0289663 | MEASURED |
| B (RFon, RateStd) | 1.06209 | MEASURED |
| C (RFoff, RateSlow) | 1.013898 | MEASURED |
| D (RFon, RateSlow) | **UNRESOLVED** (smoke-only, save crash) | UNRESOLVED |

| contrast (H_mean) | estimate | interpretation | tag |
|---|---|---|---|
| **β_RF_h** (B−A) | **+0.03312** | RF on **raises** H_mean by ~+0.033 | MEASURED (2-point contrast) |
| **β_τ_h** (C−A) | **−0.01507** | LONG (slow) rate **lowers** H_mean by ~−0.015 | MEASURED (2-point contrast) |
| **β_RF×τ_h** (interaction) | **UNRESOLVED** (D missing) | — | UNRESOLVED |

## 4. Rate-level & full factorial — UNRESOLVED

| estimand | status | reason | tag |
|---|---|---|---|
| `rate_level_delta_exposure` | **UNRESOLVED** | B,D lost post recordings to save crash; A,C partial | UNRESOLVED |
| full 2-way ANOVA (RF×Rate) | **UNRESOLVED** | missing D (interaction) and B post (rate Δ) | UNRESOLVED |
| `field_low_gamma`, recovery, slot-resolved rate Δ | UNRESOLVED / NOT_RECORDED | B,D post unrecoverable; A/C slot-resolved primary estimands NOT_RECOVERABLE from ledger | UNRESOLVED |

**Completion:** the FULL factorial completion predicate (C1–C7) is **INCOMPLETE** — B and D lack `result.json`/`EvidenceRef.json`/`artifact_hashes.json` + post recordings. **INCOMPLETE ≠ NEGATIVE**: missing execution is a design-execution failure, not a scientific negative result (per `repro_review.json`).

## 5. Execution-failure detail (B/D) — UNRESOLVED, NOT NEGATIVE

From `rf_rate_factorial_repro_review.json` + `four_cell_lifecycle_seal.json`:

| cell | last_global_step | last_phase | terminal_reached | result_json | classification |
|---|---|---|---|---|---|
| B | 20900480 (post 95/96) | post | true | absent | execution_failure → UNRESOLVED |
| D | 20900480 (post 95/96) | post | true | absent | execution_failure → UNRESOLVED |

Both B and D ran the **full trajectory to terminal step 20900480** but the executor crashed at the save step (`TypeError` at `spikes_example`), so `H/Θ` and post rate recordings were not retained. A and C full-lifecycle artifacts are hash-verified (`A_hashes`/`C_hashes` match sealed manifests).

## 6. Cross-reference to H-phenotype mechanism (see dossier 07)

- B (RFon, RateStd): H mean 1.06209, stationary across exposure (std 7e-5) — reference saturation.
- C (RFoff, RateSlow): H mean 1.013898, slow-decaying (τ_eff≈1093 s measured) — LONG drift.
- The H_mean contrasts are consistent with the mechanism in dossier 07 (RF raises the H operating point ~+0.03; LONG rate lowers the mean H toward its slower-decaying trajectory ~−0.015). These are **2-point, n=1-per-cell contrasts with no SE** — they are `MEASURED` magnitudes but carry no inferential p-value; formal ANOVA inference is `UNRESOLVED`.

## 7. Number registry

All numbers in tables carry an evidence tag and source; `UNRESOLVED` fields are never rendered as negative, and the missing D interaction cell is explicitly `UNRESOLVED`, not evidence of no interaction.
