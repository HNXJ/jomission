# SCI-AGSDR-CALIBRATION: J_CL1C rerun spec (sealed on commit of this file)

Supersedes: docs/designs/SCI-AGSDR-J_CL1C_spec_DRAFT.md (kept as history).
Authority: owner rulings C1 to C6 and authorization to run 96 interventional simulations (2026-11-04).
Scope: computes J_CL1C for the 24 stage-0 candidates. It does not select, does not run stage 1, does not change a gate, weight, sampling rule or candidate parameter, and does not edit stage-0 or B3/J_rates receipts.

## Rulings applied (2026-11-04)
- C1: each candidate's own stage-0 motif gains (its own model).
- C2: intervention = x1.5 on each neuron's own builder drive (the emitter's per-neuron drive array), for the target class.
- C3: all J_CL1C thresholds are soft (recorded, not gating).
- C4: no scratch review file is used; the frozen "current_evidence" values are not used.
- C5: raw deltas reported; no effect size d (one seed).
- C6: each perturbation's violation is a clipped shortfall in [0, 1]; J_CL1C is the equal-weight mean of the three perturbation violations.

## Perturbations (frozen intervention, rulings C1 to C2)
- PV: drive x1.5 on PV neurons.
- SST: drive x1.5 on SST neurons.
- VIP: drive x1.5 on VIP neurons, plus +2.0 added to the drive of VIP neurons (frozen: "additive +2.0 direct current").
- Each perturbation: baseline and perturbed run, same Poisson background (rate, amplitude and seed as the candidate's builder args; Poisson seed 0 + 7919 as in the runner), same engine seed 0, duration 1000 ms, dt 0.1 ms.

## Measured deltas (perturbed minus baseline)
- delta_r_X: change of the class mean rate of class X (Hz; spikes / 1 s), for X in E, PV, SST, VIP.
- delta_rho: change of the mean pairwise correlation of 100 ms binned counts (10 bins) over 400 neuron pairs, drawn with the runner's pair rule (seed LHS_SEED + 0), zero-variance pairs excluded; the same pairs in baseline and perturbed.

## Thresholds and shortfalls (frozen soft thresholds; bonuses excluded)
- Shortfall for a requirement "x >= t" (t > 0): clip((t - x) / t, 0, 1).
- Shortfall for a requirement "x <= t" (t < 0): clip((x - t) / |t|, 0, 1).
- Shortfall for a requirement "x <= 0" (no threshold other than zero): not used (E_down preferred is a bonus; see below).
- PV violation = mean of: shortfall(delta_r_PV >= +20 Hz); shortfall(delta_rho <= -0.05).
- SST violation = mean of: shortfall(delta_r_SST >= +8 Hz); shortfall(delta_r_E <= -0.5 Hz).
- VIP violation = mean of: shortfall(delta_r_VIP >= +1.5 Hz); shortfall(delta_r_SST <= -1.0 Hz).
- Excluded as bonuses (reported, not in J_CL1C): delta_rho <= -0.10 (strong); delta_r_E < 0 for PV; delta_rho <= -0.03 for SST; delta_r_E >= +0.1 for VIP (E_up).
- Implementation choice (declared here, not ruled): the violation of a perturbation is the equal-weight mean of its listed shortfalls. The owner may revise; this is reversible before the run.

## J_CL1C
- J_CL1C = (PV violation + SST violation + VIP violation) / 3, in [0, 1] (C6).

## Controls (must pass; failure stops and is reported)
1. Determinism (smoke, AGS_S0_06): two identical baseline runs give identical spikes.
2. Zero-perturbation: the perturbation pipeline with x1.0 and no addition reproduces the baseline spikes exactly (AGS_S0_06).
3. Target-only: the perturbed drive array differs from the baseline only on the target class (checked for each run).

## Outputs (new, additive)
- docs/receipts/agsdr_jcl1c_rerun/candidate_AGS_S0_XX_jcl1c.json: baseline and perturbed class rates, deltas, rho, shortfalls, violations, J_CL1C, bonuses (reported), controls.
- docs/receipts/agsdr_jcl1c_rerun/summary.json: all 24, spec sha256, script sha256, controls.

## Stop rules
- A control fails: stop, report, change nothing.
- A missing input (e.g., the emitter drive array not found): stop before running.

## Not decided here
- Selection, the shortfall (OPEN-26), the H08 amendment, and the dimension set are separate.
