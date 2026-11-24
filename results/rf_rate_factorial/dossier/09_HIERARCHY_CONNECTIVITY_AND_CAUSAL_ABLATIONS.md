# 09 — HIERARCHY, CONNECTIVITY & CAUSAL ABLATIONS

_Characterization dossier. This report documents (1) the implemented feed-forward area hierarchy and connectivity, and (2) the causal-ablation evidence — the Q8 state-replacement counterfactuals and the (incomplete) RF/Rate factorial ablations — with strict polarity discipline. Q8 rate-level result is **NEGATIVE**; field-level is **UNRESOLVED**; the factorial phenotype ablations are **NOT_EXECUTED → UNRESOLVED** (never rendered as negative). Sources: `manifests/q8_evaluation_seal.json`, `manifests/q8_canonical_closure_dt01.json`, `results/q8_evaluation/q8_B_independent_3seed_dt0p1.json(+.csv)`, `jomission/analysis/q8_phenotype.py`, `jomission/simulation/state_replacement.py`, `manifests/rf_rate_factorial_repro_review.json`._

---

## 1. Area hierarchy & connectivity (implemented)

| quantity | value | provenance | tag |
|---|---|---|---|
| network realization | `V1→V4→FEF→PFC 100/area izhikevich edge_list` | `EvidenceRef.json` (all cells) | MEASURED |
| feed-forward drive routing | V1→V4→FEF→PFC through the network | design/network realization | MODEL_ASSUMPTION (network realization) |
| neurons / area | 100 (V1 E64 PV11 SST9 VIP16; V4 E62; FEF E60; PFC E56 …) | `dossier_receipt.json` / `RF_operator.npz` | MEASURED |
| RF drive target | V1 L4 E(9)+PV(3) only; V4/FEF/PFC receive no RF drive | dossier 06 | MEASURED |
| Δ_exposure hierarchy | V1→V4→FEF→PFC om.-vs-intact field coupling (see dossier 03/06) | field manifests | MEASURED (proxy_readout) |

The hierarchy is a fixed, feed-forward `V1→V4→FEF→PFC` edge-list realization shared by all four factorial cells; there is no measured recurrent inter-area coupling estimate in this packet (`UNRESOLVED` if claimed beyond the realization).

## 2. Q8 causal ablations — counterfactual state replacement

Q8 question: **Does H or Θ store exposure history despite rate Δ≈0?** Carriers are replaced from post→pre state (`ContinuationState` carriers, matched RNG/inputs, T1–T7 intact). Canonical result: dt 0.1, **n=3 seeds**, 12-condition battery per `q8_B_independent_3seed_dt0p1`.

| counterfactual | carrier | rate Δ (Hz, mean±SD) | rate p | rate d | **rate polarity** | field | recovery | **overall** |
|---|---|---|---|---|---|---|---|---|
| H_post→H_pre | H | **0.0005 ± 0.0058** | 0.888 | 0.09 | **NEGATIVE** | UNRESOLVED | UNRESOLVED | NEGATIVE |
| Θ_post→Θ_pre | Θ(w) | 0.0013 ± 0.0041 | 0.650 | 0.31 | NEGATIVE | UNRESOLVED | UNRESOLVED | NEGATIVE |
| H+Θ_post→H+Θ_pre | H+Θ | −0.0004 ± 0.0040 | 0.890 | −0.09 | NEGATIVE | UNRESOLVED | UNRESOLVED | NEGATIVE |
| fast_X_post→X_pre | X_fast | −0.0007 ± 0.0033 | 0.739 | −0.22 | NEGATIVE | UNRESOLVED | UNRESOLVED | NEGATIVE |
| history_valid HΘ vs fast | history_valid | −0.0004 ± 0.0040 | 0.890 | −0.09 | NEGATIVE | UNRESOLVED | UNRESOLVED | NEGATIVE |

All rate Δ magnitudes are **well below the frozen 0.5 Hz threshold** with p≥0.05 → **rate polarity NEGATIVE** (per `q8_phenotype.assign_polarity`). **Field and recovery polarities are `UNRESOLVED`**: each artifact is a single-condition (AXAB) probe with n=1 < 4, failing the frozen `min_trials=4` / `min_window_ms=500` gates; the 12-condition, n≥4 field/recovery battery is pending. Therefore **overall Q8 = rate NEGATIVE, field UNRESOLVED** — no carrier stores exposure history at the **rate** level under this probe, and field-level history is not adjudicated.

**Pilot caveat:** the earlier dt=2.0 single-probe run (`q8_evaluation_seal.json`) showed an `H_post→H_pre` rate Δ of −2.7 Hz (above 0.5 Hz) flagged `POSITIVE_hint` but `UNRESOLVED` at n=1. That pilot is **superseded by the canonical dt=0.1 n=3 result** (rate NEGATIVE); the discrepancy is a power/config artifact, not an established positive. It is recorded here for transparency, tagged UNRESOLVED/pilot.

## 3. RF / Rate factorial ablations — NOT_EXECUTED → UNRESOLVED

The full factorial ablations at the phenotype level (rate-level `Δ_exposure`, field low-gamma, recovery, and the 2-way ANOVA) were **not completed**: cells B and D crashed at save (full trajectory ran to terminal step 20900480, but post recordings were not retained). Consequently:

- **RF ablation** at phenotype level: UNRESOLVED (NOT NEGATIVE).
- **Rate ablation** at phenotype level: UNRESOLVED (NOT NEGATIVE).
- **Interaction (RF×Rate)** at phenotype level: UNRESOLVED (D missing).
- Only the **H_mean partial contrasts** are `MEASURED` (β_RF_h=+0.033, β_τ_h=−0.015; see dossier 08) — these are n=1-per-cell pairwise magnitudes, not causal ablation verdicts.

**Rule (frozen):** `ablations NOT_EXECUTED → UNRESOLVED`, never `NEGATIVE`. Missing execution is a design-execution failure, not a scientific negative.

## 4. Number registry

All rate deltas, p-values, and d appear in tables with evidence tag and source. Field/recovery cells are `UNRESOLVED` (underpowered/pending), never rendered as negative. The causal-interpretation boundary is explicit: rate-level Q8 is NEGATIVE; history storage at field/recovery level is `UNRESOLVED`.
