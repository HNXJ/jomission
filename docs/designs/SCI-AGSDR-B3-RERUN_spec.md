# SCI-AGSDR-CALIBRATION: B3 rerun spec (sealed before execution)

Status: sealed on commit of this file (the sealing commit is recorded in the run receipts).
Authority: owner Q7 (2026-11-04): "Authorize a rerun of the 24 stage-0 sims that saves per-population ratios".
Scope: this spec only. It does not select candidates, does not run stage 1, does not change any gate, weight, sampling rule or candidate parameter, and does not touch the original stage-0 receipts.

## Question and estimand
- Question: what is soft term B3 (fraction of populations with tau-weighted realized E/I ratio in [0.15, 0.60]) under the frozen J_EI definition W*r*tau, for each stage-0 candidate?
- Estimand: per-population ratio p_p = I_E,p / (I_E,p + I_I,p) with I_E,p = sum over edges with E presynaptic and post in population p of tau_E * |w| * r_pre, and I_I,p likewise for non-E presynaptic with tau_I. Populations with I_E,p + I_I,p = 0 are excluded, as in the runner.
- B3 = fraction of included populations with 0.15 <= p_p <= 0.60. Soft term B3_term = max(0, 0.8 - B3) / 0.8 (runner convention).
- Population = (area, layer, cell_type), as in the runner.

## Inputs (frozen)
- The 24 candidate receipts docs/receipts/agsdr_local/candidate_AGS_S0_*_seed0.json: values, builder_args, metrics.
- Runner scripts/agsdr_local_sweep.py (its build, simulate and score functions are used unchanged).
- Tau values: docs/receipts/agsdr_local/stage0_tau_check.json (E-pre 2.0 ms, I-pre 5.0 ms, single-valued).
- Duration 2000 ms, dt 0.1 ms, seed 0, n = 400 (runner constants).

## Controls (must pass before the rerun is accepted)
1. Reproduction: each rerun candidate must reproduce its stored total_spikes exactly and its stored bar_r and mean_rho to 1e-9. Failure of any candidate stops the rerun and is reported; no retuning.
2. Tau-free control: the tau-free per-population fraction in [0.15, 0.60] (the runner's proxy) must equal the stored population_Efrac_frac_in_0p15_0p60 for each candidate. This checks the per-population accumulation against the runner.
3. Tau check: tau values per presynaptic class are single-valued in each rebuilt model.

## Outputs (new, additive)
- docs/receipts/agsdr_b3_rerun/candidate_AGS_S0_XX_b3.json: per candidate: reproduction checks, per-population sums (tau-free and tau-weighted), counts, B3 (tau-weighted and tau-free), B3_term.
- docs/receipts/agsdr_b3_rerun/summary.json: all 24, controls result, and the spec sha256 and the sealing commit.
- The original stage-0 receipts, stage0_summary.json, and stage0_correction_v1.json are not edited.

## Stop rules
- Any control 1 to 3 fails: stop, write the failure receipt, do not proceed.
- Selection, stage 1, and any gate change are out of scope; their results are not produced here.

## Not decided here
- H05 denominator, H08 status, overflow, dimension set and the epsilon predicate are settled by the owner's Q1 to Q6 rulings and recorded elsewhere; this spec does not use them.
