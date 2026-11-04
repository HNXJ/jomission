# H balance rule vs baseline (pre-registered, sealed on commit of this file)

Question: does the activity-balance H rule (`jomission/dynamics/activity_balance.py`), acting through the additive drive bias, restrain rate and synchrony when the canonical 400-neuron network is driven hard, while leaving 1 to 10 Hz neurons near H = 1?

Estimand: differences (balance minus baseline) in class mean rates, bar_r, per-neuron rate SD, mean pairwise rho, population synchrony (Fano of 10 ms population count), per condition and seed; and the H distribution by rate bin.

## Configuration under test (frozen before the full run)
- Rule: `activity_balance`, parameters = `DEFAULT_PARAMS` of the module (r_lo 1 Hz, r_hi 10 Hz, tau_r 1000 ms, tau_H 1000 ms, kappa 0.05 1/(Hz s), kappa_I 5.0). Proposed and untuned; no parameter changes after this commit.
- Engine: `RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, hdp_params={"hdp_rule": "activity_balance", "hdp_rule_params": {}})`.
- Baseline: same network and seeds, `RuntimeConfig(recurrent_backend="edge_list")`.
- Network, conditions, seeds, duration, background, metrics code: identical to `docs/designs/H_ACTIVE_TEST_spec.md` (A: tonic 0.6; B: tonic 0.9; seeds 0, 1; 2000 ms; dt 0.1 ms; 8 runs). Metrics are imported from `scripts/h_active_test.py`.
- Additional record per balance run: b_final (mean, SD, min, max) and max |b - kappa_I (H - 1)|.

## Prior information (disclosed)
A smoke run (A, seed 0, 2000 ms; scratch, not a receipt) preceded sealing: baseline 7555 spikes, balance 5632, H range 0.62 to 1.04, H mean 0.90 (> 10 Hz), 1.03 (< 1 Hz), 1.00 (1 to 10 Hz). Hypotheses below for A seed 0 are therefore not independent; the other 6 runs are.

## Hypotheses (each can fail)
- H1: bar_r(balance) < bar_r(baseline) in every run (8 pairs, both conditions).
- H2: H mean in the > 10 Hz bin < 1.0 < H mean in the < 1 Hz bin, in every balance run (bins with n = 0 are reported, test undefined).
- H3: per-neuron rate SD (balance) < baseline in every run.
- H4: pop_synchrony_fano_10ms and mean_rho (balance) <= baseline in every run (hypersynchrony restraint). The rule has no synchrony term; a failure is a result.
- Report effect size as per-run difference and relative change; n = 2 seeds per condition (network fixed), so no inferential test; spread across seeds only.

## Controls (all must pass before metric differences are interpreted)
1. Null wiring: balance rule with kappa = 0 reproduces baseline spikes exactly (200 ms, seed 0).
2. Baseline deterministic: repeated baseline run identical (200 ms).
3. Engagement: full-length balance run differs from baseline and H SD > 0.
4. Rule unit tests (`tests/test_activity_balance.py`) pass.

## Known limitation (disclosed)
H is carried in float32 near 1.0; increments below about 6e-8 per step are lost for H while the bias b integrates them (2.93M of 8.0M neuron-steps in the smoke). b departs from kappa_I (H - 1) by up to 5.6e-3 at 2000 ms (|b| up to about 1.9). Reported, not a gate. `JAX_ENABLE_X64=1` does not change the trace dtype.

## Stop rules
- Any control fails: report, do not interpret the metric differences.
- Rate-bin thresholds 1 and 10 Hz are the owner's band; not tuned.

## Not decided here
- Whether the rule belongs in the calibrated model (OPEN-27); parameter tuning; the synchrony term.
