# Item 3 seal proposal (DRAFT, not sealed)

Status: **approved as drafted by the owner, 2026-11-05 (chat: "Approve as drafted").** What is sealed is the structure: definitions, sources, fit/held-out split, leakage controls, negative controls, uncertainty and multiplicity handling. Every numeric acceptance threshold stays `UNSET`; no run that needs one may start until the owner sets it, and setting it is an amendment committed before the run. Score changes are made by the reviewers only. Seal commit: see git log of this file.

Labels. **[S]** value reported by a source, with location. **[D]** definition proposed here. **[H]** hypothesis. **[T]** acceptance threshold (all `UNSET`). Sources: **M** manuscript (PDF page), **D** deck v4 (slide), **T25**, **T26** (PDF page). Every [S] value below is machine-checked in `item02_digest_receipt.txt` (97 of 97 rows matched).

Source cautions (H3). M reports PC1Y 71.53% + PC2Y 28.35% = 99.88%, cumulative stated 99.87% (rounding); D reports simulated PCA as 96/2% (sums to 98%, other PCs exist). D "motif r of the area means" has no stated aggregation (see M1).

## Gate: item 2 completed

Missing checks added and run: synaptic Q tables (T25 Table 2 p10, T26 Table 2 p22), τ_E, τ_I, delays, N, mean-field T = 15 ms (T26 p6), τ_d 2 ms, linear model (Eq. 10 matrix, Q_SST→E 4, Q_SST→PV 2.05, τ 20 ms, 4 ms lead). Result 97 of 97 matched. Finding: **the two papers use different synapse sets** (T25: Q_PV 6, Q_SST 5, E→E 1.25, E→PV 3.75, E→SST 2.5, τ_E (1, 2, 5), τ_I 5, no delay; T26: Q_{PV,SST}→{E,PV} 5, E→E 2, E→{PV,SST} 3, τ_E 2, τ_I 8, delay 2 ms, p 0.07). A run uses one paper's neuron and synapse sets together; the choice is item 2's remaining decision (UNSET). Item 2 score is unchanged pending your approval.

## Metrics

### M1 Motif r

| Field | Proposal |
|---|---|
| Definition [D] | Per area a: `r_a = mean over f of Pearson r over the 21 bins` between simulated and observed relative power `RP(c,f) = P(c,f)/max_c P(c,f)` (M eq. 1, p8), the same form as M's r_PRED (p15). Bands per M p16. 60 Hz interpolated as in M p8 |
| Aggregation [D] | Primary: mean of r_a over held-out areas. D's "motif r of the area means" (D8) has no stated definition: `UNSET` until recovered from D's analysis code; reported as secondary only |
| Empirical reference [S] | D8: 0.44 (area means, physiological refit). D9: held-out r 0.30 (range −0.15 to 0.52; MST lowest, PFC highest), SST-only fit 0.35. Observed between-area conservation of the motif 0.76 (M p18); own-area anatomy→LFP 0.88, cross-area 0.70 (M p23). The ceiling for simulated vs observed r is not reported: `UNSET` |
| Uncertainty [D] | Area-level: bootstrap over the 11 areas; seed-level: simulated seeds averaged within area first. Null: shuffle laminar bins within area, 5,000 permutations (M p13-14). Bootstrap count `UNSET` |
| Acceptance [T] | `UNSET`. Form: lower bound of the held-out mean exceeds the permutation-null 95th percentile and exceeds the SST-only comparator (D9, 0.35) |
| Role | Held-out acceptance. Training-area r is a fit objective only |

### M2 Peak depths

| Field | Proposal |
|---|---|
| Coordinates [S] | Depth as fraction of cortex from mid-L4 (0), supragranular negative; 21 bins, −0.6 to 0.4, step 0.05, 60%/40% split (M p12). Simulated column uses the same 21 bins (D3) |
| Bands | Primary: M bands (alpha-beta 8-30, gamma 31-150 Hz; low gamma 31-90, high gamma 91-150; M p16). Hypothesis bands [H]: gamma 35-90, alpha-beta 10-25 Hz, secondary only |
| Peak location [D] | Argmax over bins of the band-mean RP (as the band lines in M Fig 1D). Edge-bin handling declared before running (D6: edge bins wrong) |
| Empirical [S] | Text only: gamma peaks in L3, alpha-beta in L5/6 (M p18); gamma above alpha-beta in 10 of 11 observed areas, 5 of 11 simulated (D9). Numeric peak bins are in figures only: `UNSET` (observed data not in hand) |
| Alignment caution | M aligns anatomy and LFP by assuming the alpha-beta/gamma crossover is L4 (M p12). The simulated crossover must therefore not be used to align the simulation; bins are fixed by the anatomy |
| Uncertainty [D] | Bootstrap of peak bin over probes (observed) and over seeds (simulated); interval in bins. Count `UNSET` |
| Tolerance [T] | `UNSET`. Form: simulated peak inside the observed peak's bootstrap interval, or within k bins (k `UNSET`); and per-area order gamma peak superficial to alpha-beta peak, counted against 10/11 observed |
| Role | Held-out; never in the fit objective |

### M3 PCA split

| Field | Proposal |
|---|---|
| Preprocessing [S] | Mean over areas of the 21 x P matrices; z-score along the feature dimension (frequencies), not bins (M p13); PCA; keep PCs to at least 90% cumulative variance (M p13); bootstrap ratio threshold 2.5, 5,000 resamples (M p14). Simulated Y0 uses the same pipeline; training-areas-only z-scoring [D] |
| Components, variance [S] | Observed Y: PC1 71.53%, PC2 28.35%, cumulative 99.87% (M p21). Observed X: 62.47%, 28.46%, 90.93% (M p19). D10: simulated 96/2% vs observed 72/28%; irregular-firing variant 59/16% (D11) |
| Loadings [S] | PC1Y: delta/theta and gamma positive, peak L3. PC2Y: alpha-beta and low gamma, positive from L3 through L5/6 (M p21) |
| Split criterion [D] | (i) number of PCs for 90%; (ii) PC1:PC2 shares; (iii) loading sign pattern; (iv) Procrustes distance of score profiles to the observed prototype. M's critical d (X 0.756, Y 0.752, M p14) was derived for observed areas; using it on simulations needs your approval |
| Uncertainty [D] | Bootstrap over areas for the shares; seeds averaged within area |
| Acceptance [T] | `UNSET` |
| Role | Held-out; never in the fit objective |

### M4 Leave-one-area-out r

Two analyses in the sources, kept apart:

| Analysis | Held-out unit | Aggregation | Reference [S] |
|---|---|---|---|
| A. Circuit selection (D9) | area: drives/gains chosen on 10 areas, motif r measured on the 11th | mean of 11 folds; fold agreement | held-out r 0.30 (−0.15 to 0.52); one circuit wins 10 of 11 folds |
| B. PLSR on simulated LFP (D10, M p15) | area: PLSR fit on 10 areas, predicts the 11th | per area: mean over frequencies of Pearson r over bins (M p15); then mean across areas; per band; two-sided Wilcoxon signed-rank vs 0 across areas | M: mean 0.71; alpha-beta 0.71-0.88, high gamma 0.68-0.83, low gamma 0.48-0.85, delta 0.50-0.59, theta 0.46-0.69 (ranges as printed, unit of the range not stated: `UNSET`); D: 0.70 |

Components fixed at 4 as in M p15 [D]; not re-selected on held-out folds. Expected generalization [S]: LOAO r 0.71 sits near the cross-area r 0.70, below own-area 0.88 (M p23). Analysis B alone does not test agreement with observation (it can score high for any smooth function of density); it is reported together with M1. Acceptance [T]: `UNSET`.

### M5 Q²

| Field | Proposal |
|---|---|
| Definition [S] | `Q² = 1 − Σ(Y_test − Y_pred,out)² / Σ(Y_test − Ȳ_train)²`, Ȳ the training-set mean (M eq. 2, p15). Summation index set not printed in the extracted text: proposed bins within frequency, then averaged over frequencies and bands; `UNSET` until confirmed |
| Cross-validation [S] | Leave-one-area-out, 11 folds, 4 PLSR components (M p15); training-only z-scoring [D] |
| Empirical [S] | M mean 0.42; alpha-beta 0.40-0.75, high gamma 0.37-0.66, low gamma −0.10 to 0.67 (M p21). D10: 0.37 refit, 0.16 SST-only, 0.42 observed |
| Null baseline | Q² = 0 (predict the training mean); permutation null: shuffle laminar bins in anatomy, 5,000 permutations, Fisher combined probability (M p15) [S]; comparators 0.16 (SST-only, D10) |
| Acceptance [T] | `UNSET`. Form: above the permutation-null 95th percentile and above 0 with the interval excluding 0 |

## Fit targets versus held-out acceptance

| Class | Quantities | Rule |
|---|---|---|
| Fit (selects drives, gains, wiring inside the sealed grid) | F1 physiological regime: all four types active after onset, per-type rates in a band (band `UNSET`; D8 rates 4.9 / 6.7 / 6.4 / 11.2 Hz are a result, not a band). F2 module level: ~35 Hz with PV leading SST ~7 ms (T26 p6; T25 p4); linear model 4 ms (T26). F3 training-area motif r | Chosen on training areas and tuning seeds only |
| Held-out acceptance | M1-M5 on the held-out area and evaluation seeds; SZ knockdown direction (M predicts high-gamma loss, beta trend p 0.054, M p24; D12: PLSR 120 Hz −0.40% vs −0.13%, direct knockdown lowers low gamma) | Never enter the fit objective; no metric used in fitting also serves as acceptance for the same area |

## Independent validation and leakage controls

- Available: 11 areas with matched anatomy and LFP (M p2, p5). LFP from Mendoza-Halliday's 14 regions (M p3, p6) has no matching anatomy in hand, so it cannot validate the density-to-motif map (`UNSET` as validation data).
- Nested leave-one-area-out: for held-out area a, (1) choose circuit, drives and gains from the sealed grid using areas other than a; (2) evaluate a on evaluation seeds not used in step 1; (3) a's observed LFP is not read before step 2.
- PCA, PLSR and z-scoring parameters come from training areas only. In M, z-scoring is applied to the mean over all 11 areas (p13); whether it is refit per fold is not stated, so the source may leak. [D] fix in simulation.
- Tuning seeds and evaluation seeds disjoint; counts `UNSET` (D used 2 seeds, 16 configs). The number of configurations tried (D5: 288 scanned, 113 kept) is recorded and enters the multiplicity family.
- Densities build the column and the PLSR inputs, so the simulation can carry the densities through trivially; see C1(ii).
- Readout fixed before running (D7: Gaussian proxy r 0.35 vs synaptic bipolar 0.17); no switching after results.
- After approval: spec hash recorded; no change after the first evaluation-seed run (new criteria start a new lineage edge).

## Negative controls

| Principal claim | Negative control | Expected if the claim is false |
|---|---|---|
| C1 Column reproduces the motif (M1, M2) | (i) bin-shuffled null, 5,000 permutations [S]. (ii) density-scrambled column: type fractions shuffled across bins, totals fixed. (iii) own-area vs other-area anatomy: specificity only, because cross-area r is already 0.70 (M p23), so it is not a null | r and peak order fall to the null; (iii) own − other > 0, size `UNSET` |
| C2 SST/PV ratio sets depth separation [H] | Flat and reversed SST/PV profile at fixed inhibitory count (item 17) | Deep alpha-beta peak persists, hypothesis unsupported |
| C3 Adaptation term contributes | H off (b = 0) and the same column with spike-shuffled adaptation input | No change in band power; term not needed |
| C4 Gamma arises from the E-PV loop with SST→PV asymmetry (T25, T26) | PV silenced; SST→PV removed; PV→SST and SST→SST added | Rhythm persists unchanged (sources: PV suppression slows then destabilizes, SST suppression abolishes it) |
| C5 Two-component split (M3) | Bin-shuffled Y | Split indistinguishable from the shuffled one |
| C6 Structure-function map (M4, M5) | Permutation null [S]; training-mean predictor (Q² = 0); SST-only comparator (D10) | Not above the null |
| C7 Knockdown direction | Sham knockdown of equal magnitude on random type/layer | Same direction as the real knockdown |

## Uncertainty and multiplicity

- Unit of inference: area (n = 11), seeds averaged within area first. Effect size, interval, n and test reported for each metric.
- Primary family: M1-M5, one pre-declared test each. Secondary: six bands, per-frequency curves (descriptive, M shows ±2 SEM), knockdown. Correction not found in M (band tests are Wilcoxon, overall Fisher; M p15-16): [D] Holm over the primary family, Benjamini-Hochberg over bands. α and q `UNSET`.
- 150 correlated frequency points are not tested individually.
- Selection among tried configurations is counted in the family; no optional stopping.

## UNSET register

| Quantity | Needed from |
|---|---|
| All acceptance thresholds (M1-M5, SZ direction) | owner |
| Numeric observed peak bins and their intervals | observed data (not in hand) |
| D's motif-r aggregation; Q² summation index; unit of M's printed ranges | D analysis code; M supplement |
| Bootstrap counts, seed counts, α, q, edge-bin rule, rate band | owner / sealed spec |
| Use of M's Procrustes cutoffs on simulations | owner |
| Neuron and synapse set (T25 or T26) | owner |
| Location of the NumPy analysis port (D1) | item 14 |

## Approval

Approve, amend or reject this table. On approval: fill the `UNSET` rows you choose to set, commit the sealed spec, then item 4 may start. Items 4, 11, 15 and 20 in the plan are amended to match (below).
