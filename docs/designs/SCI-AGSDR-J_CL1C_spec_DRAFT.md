# SCI-AGSDR-CALIBRATION: J_CL1C spec (DRAFT, not sealed, not in effect)

Status: DRAFT. Nothing is executed. Sealing needs the owner rulings listed below and an owner authorization for the interventional simulations (see Cost).

## Question and estimand
- Frozen source: manifests/agsdr_local_objectives.json, J_CL1C (lines 246 to 300). Three functional perturbations of the stage-0 candidates, each paired with a baseline on the same Poisson seed, 1000 ms transient.
- Per perturbation, the frozen deltas: PV perturbation (PV drive +50 percent): delta_r_PV, delta_rho, delta_r_E. SST perturbation (SST drive +50 percent): delta_r_SST, delta_r_E, delta_rho. VIP perturbation (VIP drive +50 percent plus +2.0 direct current): delta_r_VIP, delta_r_SST, delta_r_E.
- Frozen soft thresholds per delta: PV recruitment delta_r_PV > +20 Hz (soft; > +15 hard by the frozen text); delta_rho < -0.05 (soft pass), < -0.10 (strong); delta_r_E < 0 preferred; SST recruitment delta_r_SST > +8 Hz; delta_r_E < -0.5 Hz (soft pass); delta_rho < -0.03 bonus; VIP recruitment delta_r_VIP > +1.5 Hz; delta_r_SST < -1.0 Hz (soft pass); E_up +0.1 bonus.

## Cost
- Per candidate: 1 baseline and 3 perturbed runs of 1000 ms, each with its own seed. For 24 candidates: 96 interventional simulations of 1000 ms. This is a new execution and needs its own authorization.

## Source conflicts (to rule before sealing)
- C1 structural gains: the frozen entry gives fixed gains (PV->E 1.30, E->PV 1.70, SST->E 0.60, VIP->SST 5.00, pseudogenome v0). Each candidate has its own motif gains (stage-0 values). Which gains apply?
- C2 drive values: the frozen entry gives PV 3.01 -> 4.52, SST 2.06 -> 3.09, VIP 1.79 -> 2.69. The current builder sets drives from tonic_drive_scale (candidates range about 0.5 to 0.6) and PV_DRIVE_SCALE_DEFAULT 1.7. The stated values are not the candidate-specific builder values.
- C3 hard versus soft: the frozen text calls PV recruitment ("delta_r_PV > +15 Hz") hard, and the J_CL1C block says "not just hard". The runner has no such gate.
- C4 evidence: the frozen current_evidence cites scratch/review_CL1v2_functional.md and scratch/gen2_C014_receipt.md. Neither file is in the repository, so those values are not verifiable here.
- C5 effect size: "effect size d" is named, but a single seed gives no variance. Which seeds, and what d?
- C6 normalization: J_CL1C must be in [0, 1] for the epsilon rule. The frozen text gives "vector of three delta violations" with weight 1/3 within J_CL1C, and does not define normalization.

## Proposed definitions (for ruling)
- P1 gains: each candidate's own stage-0 motif gains (its own model), so that J_CL1C is a property of that candidate. Alternative: the frozen fixed gains for all candidates.
- P2 drives: the candidate's own builder drives (its tonic_drive_scale and the builder's per-class drive scales), with the perturbation as +50 percent of that drive. Alternative: the frozen stated values.
- P3 violations: for each threshold, a violation v = max(0, threshold - observed) / band width (for "delta must be below x") or the analogous shortfall (for "must exceed x"), clipped to [0, 1]. Each perturbation gives one violation value; the VIP perturbation adds the SST-down violation.
- P4 J_CL1C = mean of the violation values (weight 1/3 per perturbation, equal weights within each perturbation). Bounded in [0, 1].
- P5 seeds: seed 0 only, as in stage 0. Effect size is not computed (no variance from one seed); the spec reports deltas, not d.

## Controls (before the rerun is accepted)
1. Baseline reproduction: each candidate's 1000 ms baseline is rerun on seed 0 and must reproduce its stored spikes or the stored 2000 ms receipt's first 1000 ms rate within 1e-9 (the stage-0 run used 2000 ms; the first half is the same trajectory only if the runner is deterministic; this must be checked).
2. Zero-perturbation control: a perturbation with drive scale 1.0 must give delta = 0 exactly.

## Outputs (new, additive)
- docs/receipts/agsdr_jcl1c_rerun/candidate_AGS_S0_XX_jcl1c.json and summary.json.

## Stop rules
- Any control fails: stop, report, no retuning.
- Any of C1 to C6 unruled: stop before running.

## Not decided here
- Selection, the shortfall (OPEN-26), J_rates (computed), and the H08 amendment are separate.
