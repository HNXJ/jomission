# FIGURE INDEX — Jomission RF×Rate Characterization

_Index of the quantitative **figure/panel content** defined by the characterization dossier. **No raster image files (png/pdf/svg) are emitted by this packet**; each figure is a named, reproducible panel whose data are backed by a sealed **generated-owner array** (`.npz`/`.json` under a cell's artifact root) and presented by a numbered report (R01–R10). Generated-owner convention: `results/rf_rate_factorial/<CELL>/` is the artifact root recorded in each cell's `EvidenceRef.json` (`generated_owner`)._

Legend: **gen-owner** = generated-owner array(s) backing the figure; **report** = dossier file that presents the numbers.

---

## Primary cell figures (cell A_RFoff_RateRef unless noted)

| Figure | Title | gen-owner array(s) | report |
|---|---|---|---|
| **F1** | Firing-rate & HDP state: global/per-area rates, H/Θ trajectories, recording/timing | `A_RFoff_RateRef/recording/rate_summary.json`, `recording/spikes_example_AAAB.npz`, `_AXAB.npz`, `_BBBA.npz`, `recording/H_t.json`, `recording/H_t_dense.npz`, `A_RFoff_RateRef_result.json`, `recording/continuation_state.json`, `recording/event_ledger.json` | 01_BIOPHYSICS |
| **F2** | E/I balance & microcircuit: E-vs-I rates, per-class counts, layer rates | `A_RFoff_RateRef/recording/spikes_example_*.npz`, `checkpoints/ckpt_trial_0010.json` (geometry.neuron_rows), `recording/rate_summary.json` | 02_EI_BALANCE_MICROCIRCUIT |
| **F3** | Oscillations & field: Welch PSD peaks, band power, omission log-ratios, TFR, rate↔field coupling, cross-area corr, depth profile | `A_RFoff_RateRef/recording/field_proxy_*.npz`, `recording/field_proxy_manifest.json`, `recording/spikes_example_*.npz` | 03_OSCILLATIONS_AND_FIELD |
| **F4** | Habituation & SSA: pre/post rate, omission-effect stability, hidden-state H/Θ vs habituation | `A_RFoff_RateRef/recording/rate_summary.json`, `E_A_analysis.json`; `C_RFoff_RateSlow/E_C_analysis.json`; sealed `manifests/t1_t7_final_corrected.json`, `manifests/canonical_science_seal.json` | 04_HABITUATION_SSA |
| **F5** | Omission & predictive state: omission operationalization, T1–T7, Δ_exposure, H/Θ carriers, Q8 state replacement | `A_RFoff_RateRef/recording/external_drive_examples.npz`, `E_A_analysis.json`, `recording/H_t_dense.npz`; `manifests/t1_t7_final_corrected.json`, `manifests/canonical_science_seal.json`, `manifests/q8_canonical_closure_dt01.json` | 05_OMISSION_AND_PREDICTIVE_STATE |
| **F6** | Retinotopy & RF: operator weights, target composition, stimuli, drive, engine validation | `A_RFoff_RateRef/recording/RF_operator.npz`, `recording/visual_field_patterns.npz`, `recording/external_drive_examples.npz`, `checkpoints/ckpt_trial_0010.json`; `jomission/network/rf.py`, `tests/test_rf.py` | 06_RETINOTOPY_AND_RECEPTIVE_FIELDS |
| **F7** | Plasticity H/Θ dynamics & timescales: dead-key audit, effective τ, reference vs LONG H drift | `B_RFon_RateRef/E_B_analysis.json`, `C_RFoff_RateSlow/E_C_analysis.json`, `A_RFoff_RateRef/recording/H_t.json`, `recording/continuation_state.json`; `jomission/dynamics/hdp.py`, `configs/default.json` | 07_PLASTICITY_H_THETA_DYNAMICS |
| **F8** | RF×Rate factorial: factor isolation, H_mean cell means, partial contrasts, completion status | `manifests/rf_rate_factorial_estimator.json`, `E_A_analysis.json`, `E_B_analysis.json`, `E_C_analysis.json`, `manifests/rf_rate_factorial_repro_review.json` | 08_RF_RATE_FACTORIAL |
| **F9** | Hierarchy, connectivity & causal ablations: network realization, Q8 counterfactuals, factorial ablation status | `A_RFoff_RateRef/EvidenceRef.json`; `manifests/q8_canonical_closure_dt01.json`, `manifests/q8_evaluation_seal.json`, `manifests/rf_rate_factorial_repro_review.json`; `results/q8_evaluation/q8_B_independent_3seed_dt0p1.json` | 09_HIERARCHY_CONNECTIVITY_AND_CAUSAL_ABLATIONS |
| **F10** | JDNA pseudogenome architecture: identity hashes, artifact manifest, hdp_params genome, state carriers, schedule | `A_RFoff_RateRef/EvidenceRef.json`, `artifact_hashes.json`, `recording/continuation_state.json`, `A_RFoff_RateRef_result.json`; `jomission/simulation/state_replacement.py` | 10_JDNA_PSEUDOGENOME_ARCHITECTURE |

---

## Summary/index figures (dossier-level)

| Figure | Title | gen-owner array(s) | report |
|---|---|---|---|
| **F11** | Executive synthesis: positives/negatives, T1–T7, Q1–Q15, RF×Rate result, limitations | `manifests/rf_rate_factorial_estimator.json`, `manifests/rf_rate_factorial_repro_review.json`, `manifests/t1_t7_final_corrected.json`, `manifests/q8_canonical_closure_dt01.json`, `manifests/canonical_science_seal.json`, `dossier_receipt.json` | 00_EXECUTIVE_SCIENTIFIC_SUMMARY |
| **F12** | Failure atlas: reproduced/contradicted/unresolved phenomena + mechanistic constraints | aggregate of all F1–F10 gen-owner arrays above; sealed manifests | 11_MODEL_FAILURES_LIMITATIONS |

---

## Provenance notes

- **Generated-owner arrays** are frozen artifacts registered in each cell's `artifact_hashes.json`; cell A's are re-verified in `dossier_receipt.json` (`hash_verified_all: true`, all 20 consumed artifacts match sealed `artifact_hashes.json`) (R01 §header, R10 §2).
- **Field figures (F3, and field panels of F4/F5/F7/F12)** draw on the uncalibrated `proxy_readout` field (`physical_amplitude_calibrated=False`); all spectral power is arbitrary units → **relative/log claims only** (R03 §header). The `ex_AAAB` field is byte-identical to `pre_0000` (a pre-exposure duplicate) (R03 header).
- **H/Θ figures (F1, F4, F7)** reflect per-trial-mean sampling only (H/Θ dense early-time traces not materialized); effective timescale and per-trial H series are `UNRESOLVED` (R01 §4, R07 §6).
- **Factorial figures (F8, F11, F12)** reflect the partial H-phenotype only; D missing and B/D post recordings lost → interaction and phenotype-level panels `UNRESOLVED` (R08).
- No figure above claims content not present in a generated-owner array; every `UNRESOLVED` field is reported as unresolved and never imputed.
