# SCI-AGSDR-CALIBRATION: J_rates spec (DRAFT, not sealed, not in effect)

Status: DRAFT for owner ruling. Nothing here is executed. Sealing requires (1) the owner's rulings on the items marked RULING below, and (2) an owner authorization for a rerun that saves per-population mean rates (none of the stage-0 or B3 receipts keeps them).

## Question and estimand
- Frozen source: manifests/agsdr_local_objectives.json, J_rates (lines 170 to 186): per-class rate distributions P(r_E), P(r_PV), P(r_SST), P(r_VIP); "not |bar_r - 7.5| scalar".
- Estimand: for each candidate, the distance of the population mean rates mu_p from class targets, over the populations p = (area, layer, cell_type) of the canonical network (68 populations; areas FEF, PFC, V1, V4; layers L1, L2/3, L4, L5, L6; cell types E, PV, SST, VIP).
- The frozen evaluation computes mu_{a,l,c} from per-neuron rates r_i. The rate of each neuron is spike count / 2 s (runner convention).

## Targets (frozen values, with +/-50 percent soft band)
- Band for a target [lo, hi]: [0.5 lo, 1.5 hi] (frozen: "±50% soft").
- Class global targets: E [5, 8] Hz -> [2.5, 12]; PV [12, 25] -> [6, 37.5]; SST [6, 12] -> [3, 18]; VIP [8, 15] -> [4, 22.5].
- Per-layer E targets (frozen, partial): V1 L4 6 to 7; V1 L2/3 4 to 6; V1 L5 7 to 10; FEF and PFC 3 to 6 (the layer scope is not stated).
- RULING R1: per-layer E targets not listed (V4 all layers; V1 L1 and L6; FEF and PFC layers other than as ruled). Proposal: use the class global target for these populations. Alternative: stop and report a missing target.
- RULING R2: the "FEF/PFC 3 to 6" target: all layers of FEF and PFC (proposal), or only the layers the frozen text names (none named)?

## Coverage and ordering (frozen: coverage >= 0.60 soft PASS; E < PV ordering required)
- Coverage = fraction of the 68 populations with mu_p inside its band. Soft target 0.60.
- Ordering: for each (area, layer) that has both E and PV populations, a violation is mu_E >= mu_PV. Ordering penalty = violations / number of such pairs.
- RULING R3: ordering is per (area, layer) (proposal) or global mean E < global mean PV (frozen text gives no scope).

## Normalized J_rates (needed for the dominance rule, OPEN-23)
- Frozen: normalized J_k in [0, 1], 0 inside the target neighbourhood, 1 at the hard boundary. The formula is not given.
- Proposal: J_rates = w_c (1 - coverage) + w_o (ordering penalty), with w_c + w_o = 1, both in [0, 1], so J_rates in [0, 1].
- RULING R4: weights w_c and w_o. Proposal w_c = w_o = 0.5 (equal, no basis for another choice in the frozen text); the owner may choose otherwise, and the choice is frozen before any run.

## Hard item stated in the frozen text (conflict)
- The frozen text says "VIP finite >0.5 hard, >1 Hz soft desirable". The runner has no VIP mean gate; H10b is an E-class gate. RULING R5: VIP mean > 0.5 Hz is a hard gate (add to the hard list, by amendment) or soft only (proposal: soft, recorded, because the sealed design's nine-gate list does not include it).

## Data required
- Per-population mean rate mu_p for all 68 populations, per candidate, 24 candidates, seed 0.
- Not retained: receipts keep class means and population min and max only. The B3 rerun (docs/receipts/agsdr_b3_rerun) keeps per-population currents, not rates. So J_rates needs a rerun of the 24 candidates that saves mu_p. That is a new execution and needs its own authorization.

## Controls (before the rerun is accepted)
1. Reproduction: total spikes exact and per-class mean rates equal the stored class_mean_Hz (1e-9), for each candidate.
2. Mean rates of the population list must average to the stored bar_r (1e-9).
3. The rerun must not change any other stored value; compare to the stored metrics.

## Stop rules
- Any control fails: stop, write a failure receipt, change nothing.
- A missing target (R1 not ruled) or an undefined weight (R4 not ruled): stop before computing J_rates.

## Not decided here
- Selection, stage 1, the H08 status (amendment 3 draft), and the dimension set are not decided by this spec. J_rates is one of the six J entries in the owner's Q5 ruling.
