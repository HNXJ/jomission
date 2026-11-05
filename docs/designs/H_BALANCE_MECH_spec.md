# Why the balance-only rule raises rho and Fano (pre-registered; sealed on commit of this file)

Todo H-BALANCE-MECH (plan P3; docs/PROGRESS_REVIEW_2026-11-05.md section 2). Checklist item 19.
Known: under `activity_balance` (defaults) H4 fails in 4 of 4 runs. In B, rho rises from 0.018 (baseline) to 0.188, against 0.067 for a rate-matched uniform drive cut (H9). The single ceiling r_hi = 10 Hz cuts PV by 53% and E by 27% (B, seed 0).

Competing mechanisms
- M1 (inhibitory tracking): the rule suppresses the most active interneurons; inhibition then tracks shared excitation less well, so correlations rise. Prediction: exempting interneurons, or giving them class-appropriate ceilings, removes most of the rise.
- M2 (homogenisation): compressing within-class rate heterogeneity of E cells raises correlations, whatever the interneurons do. Prediction: the rise persists with H on E cells only, with E rate SD compressed.

## Design
- Network, conditions, seeds and duration as `H_BALANCE_TEST_spec.md`: canonical 400-neuron model, A (tonic 0.6) and B (tonic 0.9), seeds 0 and 1, 2000 ms, dt 0.1 ms.
- Rule `activity_balance` unchanged; only the band parameters differ (per-neuron arrays; the step broadcasts them).
- Arms per (condition, seed):
  - baseline: HDP off;
  - ref: balance defaults (r_lo 1, r_hi 10 for all cells);
  - e_only: E cells band (1, 10); interneurons band (0, 1e6), so their delta is 0, H stays 1 and b stays 0;
  - class_rhi: r_lo 1 for all; r_hi E 10, PV 25, SST 12, VIP 15 (upper edges of the B1 bands; E keeps 10).
- Runs: 4 arms x 4 = 16. Runner `scripts/h_balance_mech.py`, one process per (condition, seed).
- Measures as in `H_SYNC_TEST_spec.md` (mean_rho in 100 ms bins, rho in 10 ms bins, population Fano 10 ms, class rates), plus within-class rate SD.

## Statistic
Rise fraction R_arm = (rho_arm - rho_baseline) / (rho_ref - rho_baseline), mean_rho in 100 ms bins (the H4 statistic). Discrimination uses condition B, where rho_ref - rho_baseline is large (about 0.17); A is reported (its rise is about 0.02 to 0.04 and R would be noise-dominated).

## Hypotheses and decision rule
1. Control fails: CONTROL_FAILED_NO_INTERPRETATION.
2. M1: R_e_only <= 0.5 and R_class_rhi <= 0.5 in both B runs: M1_INHIBITORY_TRACKING_SUPPORTED.
3. M2: R_e_only > 0.5 and E rate SD (e_only) < E rate SD (baseline) in both B runs: M2_HOMOGENISATION_SUPPORTED.
4. Otherwise: NEITHER_M1_NOR_M2, reported as observed.
n = 2 seeds: descriptive. A seed 0 was seen in an earlier smoke run of the balance test; B runs were not seen with these arms.

## Controls
- C1 (positive control, exact): baseline and ref arms reproduce `docs/receipts/h_balance_test` (bar_r, mean_rho, Fano) exactly in all 4 runs.
- C2: in e_only, every interneuron ends with H exactly 1; E cells are engaged (H SD > 0) (200 ms run).
- C3: class_rhi arrays hold the declared value per class.

## Limits (disclosed)
- The class_rhi ceilings come from the B1 viability bands; they are not tuned.
- The e_only arm also changes the E-to-interneuron drive balance; it separates "interneurons controlled or not", not every pathway.
- Float32 H quantization as in `H_BALANCE_TEST_spec.md`.
- Tonic network of the H tests, not the AGSDR candidates.

## Not decided here
Entry of any rule into the calibrated model (Q7).
