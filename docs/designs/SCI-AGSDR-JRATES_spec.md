# SCI-AGSDR-CALIBRATION: J_rates rerun spec (sealed on commit of this file)

Supersedes: docs/designs/SCI-AGSDR-JRATES_spec_DRAFT.md (kept as history).
Authority: owner authorization (2026-11-04) to seal this spec and run the rerun. Rulings R1 to R5 (2026-11-04) are fixed below.
Scope: produces per-population mean rates and J_rates for the 24 stage-0 candidates. It does not select, does not run stage 1, does not change any gate, weight or sampling rule, and does not edit the stage-0 receipts or the B3 rerun receipts.

## Question and estimand
- Frozen J_rates (manifests/agsdr_local_objectives.json, lines 170 to 186): per-class rate distributions; distance from class targets; ordering E < PV.
- Population p = (area, layer, cell_type); the canonical network has 68 populations (areas FEF, PFC, V1, V4; layers L1, L2/3, L4, L5, L6; cell types E, PV, SST, VIP).
- mu_p = mean over neurons in p of r_i, with r_i = spike count / 2 s.

## Target bands (soft, +/-50 percent; inclusive edges)
- Target [lo, hi] gives band [0.5 lo, 1.5 hi].
- E, V1 L4: [6, 7] -> [3, 10.5]. E, V1 L2/3: [4, 6] -> [2, 9]. E, V1 L5: [7, 10] -> [3.5, 15].
- E, FEF or PFC, any layer (R2): [3, 6] -> [1.5, 9].
- E, all other populations (R1): class global [5, 8] -> [2.5, 12].
- PV [12, 25] -> [6, 37.5]; SST [6, 12] -> [3, 18]; VIP [8, 15] -> [4, 22.5].

## Scores (R3, R4)
- Coverage = fraction of the 68 populations with mu_p inside its band. Soft target 0.60 (frozen).
- Ordering: over each (area, layer) that has both an E and a PV population (R3), a violation is mu_E >= mu_PV. Ordering penalty = violations / number of such pairs.
- J_rates = 0.5 (1 - coverage) + 0.5 (ordering penalty), in [0, 1] (R4).
- VIP mean > 0.5 Hz is recorded and not gating (R5).

## Inputs (frozen)
- Candidate receipts docs/receipts/agsdr_local/candidate_AGS_S0_*_seed0.json (values, builder_args, metrics).
- Runner scripts/agsdr_local_sweep.py: build, simulate and score functions, unchanged.
- Duration 2000 ms, dt 0.1 ms, seed 0, n = 400.

## Controls (must pass; a failure stops the run and is reported, no retuning)
1. Reproduction: total_spikes exact; bar_r and mean_rho within 1e-9 of the stored receipt.
2. Class means: for each cell type, the mean of r_i over the class equals the stored class_mean_Hz within 1e-9.
3. Population consistency: the mean of the 68 population means is not used as bar_r; bar_r is the mean over neurons and must equal the stored bar_r within 1e-9.

## Outputs (new, additive)
- docs/receipts/agsdr_jrates_rerun/candidate_AGS_S0_XX_jrates.json: reproduction, 68 population mean rates with n, coverage, ordering counts, J_rates.
- docs/receipts/agsdr_jrates_rerun/summary.json: all 24, controls result, spec sha256, script sha256.

## Stop rules
- Any control fails: stop after the failing candidate, write its receipt, run no further candidates.
- Any ambiguity found during the run that the rulings do not cover: stop and report; do not choose silently.

## Not decided here
- The six-term selection rule, the shortfall (OPEN-26), J_CL1C, and the H08 amendment are separate. J_rates does not select on its own.
