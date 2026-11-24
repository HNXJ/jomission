# 06 — RETINOTOPY & RECEPTIVE FIELDS (RF operator, cell A_RFoff_RateRef)

_Characterization dossier. The RF operator (`jomission/network/rf.py`, `tests/test_rf.py`) defines the visual→V1 drive geometry; this report distinguishes what the operator **specifies** (`MODEL_ASSUMPTION`/`LITERATURE_PRIOR`/`DERIVED`) from what is **measured** on recorded artifacts (`MEASURED`), and flags what is NOT measured (`UNRESOLVED`). Source: `results/rf_rate_factorial/A_RFoff_RateRef/recording/RF_operator.npz`, `visual_field_patterns.npz`, `external_drive_examples.npz`, `checkpoints/ckpt_trial_0010.json` geometry._

## 1. Operator specification (design/config-level)

| quantity | value | provenance | tag |
|---|---|---|---|
| lattice | 32×32 (n_pixels 1024) | config | MODEL_ASSUMPTION |
| field extent | 8° at 0.25°/px | 0.25 = 8/32 | DERIVED |
| σ (RF Gaussian) | 1.8 px | macaque V1 literature prior | LITERATURE_PRIOR |
| spacing (10×10 tiling) | 3.2 px = 32/10 for 100 V1 units | tiling | DERIVED |
| RF overlap | 0.4538 = exp(−3.2²/(4·1.8²)) | derived | DERIVED |
| stimulus blobs A / B | centers (8,8) / (24,24), stimulus σ 2.5 px | config | MODEL_ASSUMPTION |
| sparsity threshold / band | 0.2 / [0.18, 0.30] | config | MODEL_ASSUMPTION |
| Jaccard A/B | < 0.15 | config | MODEL_ASSUMPTION |
| tier / base amplitude | graded / 5.0 | config | MODEL_ASSUMPTION |
| targeting | V1 only, L4 E+PV | config | MODEL_ASSUMPTION |
| omission | zero drive (timing preserved) | config | MODEL_ASSUMPTION |
| version | rf.v0.1.0 | — | MODEL_ASSUMPTION |

All values above are carried in the recorded `RF_operator.npz` `config` string [MEASURED — the artifact stores them; their scientific provenance is as tagged].

## 2. Weights — MEASURED from RF_operator.npz

| quantity | value | tag |
|---|---|---|
| weights shape | (400, 1024) | MEASURED |
| weights mean / max / nonzero frac | 0.000244 / 0.08724 / 0.03722 | MEASURED |
| weights_target shape | (12, 1024) | MEASURED |
| weights_target mean / max / nonzero frac | 0.000977 / 0.06506 / 0.16073 | MEASURED |
| per-target-row L1 sums | min = max = 1.0 | MEASURED |
| non-target V1 rows nonzero frac | 0.1473 | MEASURED |
| non-V1 rows | zero (V1 targeting; validated also by test_rf.py) | MEASURED / INFERENCE |
| n_v1 / n_target | 100 / 12 | MEASURED |

All 100 V1 units carry a Gaussian RF; only the 12 drive targets (V1 L4 E/PV) receive scheduled drive [MEASURED].

## 3. Target composition (checkpoint geometry + target_indices)

| target | area | layer | type | count | tag |
|---|---|---|---|---|---|
| 38–46 | V1 | L4 | E | 9 | MEASURED |
| 47–49 | V1 | L4 | PV | 3 | MEASURED |
| V1 total | V1 | all | E64 PV11 SST9 VIP16 | 100 | MEASURED |

Drive is injected exclusively into V1 L4 E(9)+PV(3); V4/FEF/PFC receive no RF drive (feed-forward drive routes V1→V4→FEF→PFC through the network) [MODEL_ASSUMPTION network realization `V1->V4->FEF->PFC 100/area izhikevich edge_list`].

## 4. Stimuli — MEASURED from visual_field_patterns.npz

| quantity | value | tag |
|---|---|---|
| A peak location / value | (8,8) / 1.0 | MEASURED |
| B peak location / value | (24,24) / 1.0 | MEASURED |
| A / B pixel mean | 0.038326 / 0.038253 | MEASURED |
| A≡B pixel fraction | 0.03027 (≈97% of pixels differ) | MEASURED |
| omitted pattern | all zero | MEASURED |

A/B drive separation (computed from recorded weights): population sparsity A=B=0.25 at 0.2×max (within design band 0.18–0.30); Jaccard A/B = 0.0 (< 0.15); target-drive Pearson correlation A vs B = −0.406 (distinct, non-overlapping) [DERIVED from MEASURED].

## 5. Drive — MEASURED from external_drive_examples.npz

| quantity | AAAB | AXAB | tag |
|---|---|---|---|
| whole-trial drive mean / max | 2.2967 / 5.0 | 1.7225 / 5.0 | MEASURED |
| p2-slot [1031,1562] ms drive sum | 10 620 000 | 0 (omission zero drive) | MEASURED |
| p1-slot drive sum | 10 620 000 | 10 620 000 (timing preserved) | MEASURED |

Omission = zero drive with identical onset/duration; the ~25% drop in AXAB whole-trial drive mean (2.2967 → 1.7225) is the missing p2 slot [DERIVED].

## 6. Engine integration & validation (test_rf.py)

- `to_stimulus_schedule` realizes graded per-unit RF drive through existing JaxFNE `StimulusSchedule` `target_indices`; per-unit amplitude heterogeneity via per-unit events — no engine modification [MEASURED from code + `prove_jaxfne_target_indices_capability`].
- `apply_rf_to_configuration` adds RF metadata → config_hash distinct from canonical `4f9fdeae7428199a` (RF-cell hashes e.g. `e5e331a140ebd37e` for B/D) [MEASURED, test_rf.py `test_config_hash_distinct_from_canonical`].
- Validation asserts: L1 row sums = 1, V1-only targeting, weight sparsity 0.05–0.35 at 1e-4, population sparsity A/B in band, Jaccard < 0.15, omission drive zero, deterministic weights (seed-independent geometry) [MEASURED from test_rf.py].

## 7. Receptive-field measurement — UNRESOLVED

The operator defines the **input** drive; it is a MODEL_ASSUMPTION of the paradigm. No experiment measured emergent RF properties of the simulated neurons: no reverse-correlation/spatial-probe/contrast-response mapping was run, and no per-unit spatial tuning curve exists in the recording union. Claims such as "V1 units in the simulated network are retinotopically tuned with σ≈1.8 px" are therefore `UNRESOLVED` and NOT claimed. The recorded spike data reflect the network's response to the operator, not a measured receptive field.

## 8. Provenance caveat (cross-cell)

Executed hp_hash for cell A is `f327f9d2ad64cc88` (canonical) rather than the design RF-off hash `bb8277e7a8e0bca2`; effective K_HDP during the run is `UNRESOLVED` (see dossier 01 §5). Consequently the "RF on/off" factor is realized at the config/drive level (RF metadata + retinotopic schedule), but the K_HDP intervention status of cell A is not independently verified. This caveat applies to any RF-factor contrast built on these cells.

## 9. Number registry

All numbers appear in tables with evidence tag and source. Emergent receptive-field tuning, per-unit spatial tuning curves, and reverse-correlation estimates are `UNRESOLVED` and are not imputed.