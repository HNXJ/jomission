# H active configuration vs baseline (pre-registered, sealed on commit of this file)

Question: does the engine's H controller, active on the canonical 400-neuron network, keep H near 1.0 under normal activity, and does it restrain activity and synchrony when the network is driven hard?

Estimand: differences (H minus baseline) in class mean rates, bar_r, mean pairwise correlation (rho), population synchrony, and the H distribution, per condition and seed.

## Configuration under test (frozen before running)
- H configuration: jaxfne `DEFAULT_HDP_DESYNC` (K_HDP 0.01, tau_0_ms 5.0, K_ctrl 0.15, alpha 0.05, gamma 0.5, barrier 0.01/0.01, C_spike 0.0), passed as `RuntimeConfig(enable_hdp=True, hdp_params=DEFAULT_HDP_DESYNC)`.
- Baseline: the same network and seeds with `RuntimeConfig(recurrent_backend="edge_list")` (HDP off).
- Network: `build_jomission_model(n_per_area=100, seed=0)` with builder defaults. Background: builder defaults (rate 2988 Hz, amplitude 2.997).

## Conditions
- A (normal): builder defaults.
- B (hyperactive stress): tonic_drive_scale x 1.5 (0.6 -> 0.9), everything else identical.
- Seeds: 0 and 1 (engine seed; the Poisson stream seed is seed + 7919).
- Duration 2000 ms, dt 0.1 ms. Total runs: 2 conditions x 2 seeds x 2 arms = 8.

## Metrics (per run)
- Class mean rates (E, PV, SST, VIP), bar_r, per-neuron rate SD.
- Mean pairwise rho: 400 pairs, 100 ms bins, the runner's pair rule.
- Population synchrony: variance over 10 ms bins of the summed population spike count, divided by its mean (Fano of the population rate).
- H (H arm only): per class mean, SD, min, max of H at the end of the run; fraction of neurons with H outside [0.8, 1.2]; correlation of H with each neuron's rate; H means in rate bins (< 1 Hz, 1 to 10 Hz, > 10 Hz).

## Controls
1. H engagement: in the H arm, H must move away from 1.0 (SD > 0) or the arm is reported as inert.
2. Baseline reproduction: the baseline arm must be deterministic (repeat one baseline run, identical spikes).
3. The two arms must differ in spikes at the start (not identical); identical output is reported as inert.

## Stop rules
- Any control fails: report, do not interpret the metric differences.
- The thresholds 1 to 10 Hz for the rate bins are the user's stated band; they are reported, not tuned.

## Not decided here
- Whether this configuration is part of the calibrated model (OPEN-27). This test reports the behaviour only.
- The user's described rule (a dead band with stronger action outside 1 to 10 Hz) is not the shipped preset; it is a separate candidate if this test shows the shipped preset does not behave that way.
