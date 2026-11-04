# Mechanism of the sync-term failure: relaxation oscillation? (pre-registered, sealed on commit of this file)

Question: is the failure of `activity_balance_sync` (rates collapse to 1.3 to 5.4 Hz, Fano and rho rise; `docs/receipts/h_sync_test`) a relaxation oscillation of the delayed negative feedback loop, or a different mechanism?

Competing explanations
- **O (oscillation)**: loop `rate -> slow trace P_s (tau_r) -> sigma -> H (tau_H) -> drive_bias -> rate` has delay and gain enough to oscillate; the collapse is the low phase of a slow cycle.
- **R (ratio blowup)**: `sigma` divides by the lagged `P_s`; after any dip `P_s` is small, so a small fast-rate recovery gives a huge `sigma`, which clamps the network low (monostable collapse, no oscillation).
- **X**: neither.

## Design
- Network, conditions, seeds as `H_ACTIVE_TEST_spec.md` (A tonic 0.6, B tonic 0.9; seeds 0, 1). New runs, **15 000 ms**, dt 0.1 ms. Analysis window t >= 3000 ms (12 s), after the slowest traces (tau 3000 ms) have charged.
- Series: population rate (Hz) in 50 ms bins over the window (240 points).
- Arms per (condition, seed): baseline (HDP off); balance-only (`activity_balance`, defaults); sync (`activity_balance_sync`, defaults of the sealed sync test) at time scale s in {0.3, 1, 3} with `tau_r = tau_H = 1000 s` ms (all else unchanged); `fixref` probe (`activity_balance_sync_fixref`, s = 1, `p_ref` = baseline window population rate of the same condition and seed).
- Runs: 4 x 6 = 24, 15 s each.

## Detector (fixed before running)
Statistic S = max over band 0.2 to 5 Hz of the Hann-windowed periodogram of the linearly detrended series, divided by its band mean; f_peak = argmax; period = 1/f_peak. Null: 1000 AR(1) surrogates matched to the series' variance and bias-corrected lag-1 autocorrelation phi = phi_hat + 2(1 + 3 phi_hat)/L (clipped to [0, 0.99]); p = (1 + #{S_sur >= S}) / 1001; oscillation iff p < 0.05. Surrogate seed 12345. Periods above about 4 s (window 12 s) are poorly resolved; reported with the value, not discarded.
Window rate = mean of the series. Amplitude p2p_rel = (q95 - q05) / mean.

## Hypotheses (each can fail)
- **HM1 (oscillation exists)**: sync at s = 1 is detected in all 4 runs and balance-only in at most 1 of 4.
- **HM2 (delay scaling)**: the detected period increases strictly with s (0.3 < 1 < 3) with detection at all three scales in at least 3 of 4 (condition, seed) cases, and the median log-log slope of period on s across detected cases lies in [0.5, 1.5].
- **HM3 (ratio blowup)**: window rate ratio sync(s=1)/balance <= 0.5 and fixref/balance >= 0.8 in all 4 runs. Reported alongside: collapse independent of the denominator iff fixref/balance <= 0.5 in all 4 runs.

## Decision rule (classification)
1. HM1 and HM2 true: RELAXATION_OSCILLATION_SUPPORTED.
2. else HM1 false and HM3 (ratio blowup) true: RATIO_BLOWUP_MONOSTABLE_COLLAPSE.
3. else HM1 false: NO_OSCILLATION_DETECTED_OTHER_MECHANISM.
4. else (HM1 true, HM2 false): OSCILLATION_PRESENT_BUT_PERIOD_DOES_NOT_SCALE.
Any other pattern is reported as observed; no post hoc reclassification. n = 2 seeds per condition: descriptive; the surrogate p is per run, uncorrected (4 runs per hypothesis, "all runs" rule).

## Controls (all must pass before interpreting)
1. Detector positive control: synthetic sawtooth (period 1.5 s) plus AR(1) noise of equal SD, 100 draws: detection rate >= 0.9 (200 surrogates per draw).
2. Detector negative control: pure AR(1) noise (phi 0.5 and 0.9), 100 draws each: false-positive rate <= 0.10.
3. Probe unit tests (`tests/test_activity_balance_sync_fixref.py`): equals sync when `p_ref` = mean slow rate; differs and scales as 1/`p_ref`; finite at near-silence. Rule tests for balance and sync pass.
Stop rule: any control fails: report, do not interpret.

## Pre-seal revision (disclosed)
The first detector control run (uncorrected phi_hat) gave a false-positive rate 0.12 at phi 0.9 (limit 0.10); measured phi_hat bias after detrending at L = 240 was -0.026 (phi 0.5) and -0.030 (phi 0.9). The correction above was added before sealing and controls were rerun. No run data was used.

## Limits (disclosed)
- Single gain kappa_s = 0.5, single g; the oscillation threshold in gain was not scanned.
- Time scaling moves `tau_r` and `tau_H` together; it does not separate their contributions.
- Spectral detection on 12 s cannot resolve periods much beyond 4 s; scale-3 periods may saturate.
- `p_ref` is taken from the baseline arm (a probe choice, not a candidate rule).
- Floating-point H quantization as in `H_BALANCE_TEST_spec.md`.

## Not decided here
Whether any synchrony term enters the model (OPEN-27); a redesigned term.
