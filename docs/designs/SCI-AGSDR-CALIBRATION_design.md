# SCI-AGSDR-CALIBRATION: sealed design (pre-execution)

Status: DESIGN SEALED BEFORE EXECUTION. Run only after the TODO item is updated to AUTHORIZED and the B10 objective audit is committed.
Authority: owner instruction 2026-11-04 (unblock SCI-AGSDR-CALIBRATION now). Frozen spec: manifests/agsdr_local_freeze.json, manifests/agsdr_local_harness.json, manifests/agsdr_local_objectives.json, manifests/agsdr_local_seeds.json.
Known limitation (stated risk): the 2026-09-19 order warned that calibrating on the PV-silent / SST-dominant regime may absorb substrate defects. Results must be read against that risk.

## Free parameters (frozen ranges, freeze five_dimensions_orthogonal)
Mapping chosen by owner 2026-11-04 (centred mapping). Multiplicative W_ij = W_base_ij * g_k(mask), with g = 1 for every motif outside the masks.
- g_rec in [0.25, 0.45]: mask {E->E}. Note: the range does not contain the baseline 1.0; flagged as an inconsistency in the frozen spec, used as written.
- g_fastEI in [1.4, 2.0] for E->PV, and [1.0, 1.6] for PV->E.
- g_dend in [0.5, 0.9] for E->SST; and [0.42, 0.78] for SST->E, SST->PV; and [0.56, 1.04] for SST->VIP.
- g_disinh in [3.5, 6.5]: mask {VIP->SST}.
- g_background: tonic scale [0.45, 0.75]; Poisson rate [1000, 3000] Hz; Poisson amplitude [1, 3]. Constraint mu_total in [3.4, 4.0].
Frozen (not free): E typed mixture M2, SST/VIP/PV intrinsics, jitter, spatial, lognormal, RF, cell counts, delays, H/Theta, dt, p_ff, p_fb, all other motifs.
Builder seams: background rate/amplitude/tonic scale are builder arguments (jomission/network/builder.py build_jomission_network). Motif gains are applied post-construct by _apply_motif_gains (builder.py:623) from the MOTIF_GAIN table; the candidate applies the four mask scales multiplicatively to the table values.

## Objectives and gates (frozen)
Hard (any FAIL rejects a candidate): finite; 1 < bar_r < 30 Hz; mean rho < 0.60; CV_rate > 0.15; P(CV_E > 0.5) >= 0.15; max CV_E >= 1.5; Efrac in (0.05, 0.95); no class mean > 80 Hz; E > 0.5 Hz. Mean CV_ISI is soft only.
Soft distributional (Pareto, w_k = 1/6, tie-break only): B1 mu in [3, 15] Hz; B2 CV_rate > 0.3, mean CV in [0.5, 1.5], Fano in [0.7, 2.0], rho in [-0.05, 0.2]; B3 Efrac in [0.15, 0.60] for at least 80% of populations.
Selection statistic: cross-seed mean, not best seed. Single-seed claims forbidden.

## Seeds and fidelity (frozen)
S0 = {0}, 24 candidates. S1 = {0, 1}, 8 candidates. S2 = {0, 1, 2, 3}, 3 candidates. Window 2000 ms, dt 0.1 ms, n = 400. Total 52 simulations.

## Procedure (departures from the frozen spec, declared here)
1. Stage 0: 24 candidates from a fixed Latin hypercube over the free box (seeded, declared). The frozen spec's AGSDR proposal step (Bayesian or evolutionary) is NOT used; this is a declared deviation. No Jacobian is estimated.
2. Hard feasibility first on S0 single seed. Keep feasible candidates; rank by Pareto over soft distributional deviations with w = 1/6.
3. Stage 1: the top 8 feasible candidates, rerun on S1 = {0, 1}; rank by cross-seed means.
4. Stage 2: the top 3, rerun on S2 = {0, 1, 2, 3}; report pooled distributions, mean, 95% CI (t_3), and spread.
5. Final theta* = argmin of stage-2 Pareto ranking. Advance at most one theta* to a ledger edge; no retuning after results.
6. Required before stage 0: the B10 objective audit (jomission/qualification/gen2_gates.py audit_b10_objective, 0 hits) committed as an artifact.

## Outputs
Per candidate and seed: docs/receipts/agsdr_local/candidate_<id>_seed<s>.json (receipts, not results/ until owner decides on sealing). A trace file docs/receipts/agsdr_local/trace.jsonl. Summary receipt docs/receipts/agsdr_local/summary.json.
Execution tier: tier 3 scientific execution, under TODO SCI-AGSDR-CALIBRATION.

## Stop rules
- Stage 0 yields no feasible candidate: stop; record the null.
- Any B10 audit hit: stop before stage 0.
- Any failure of an environment or engine check: stop and report; no silent retry.
