# H balance + synchrony term (pre-registered, sealed on commit of this file)

Status: sealed on commit. Owner command (2026-11-04): seal with the defaults and run. Parameters are proposed and untuned and are fixed at this commit. Supersedes `H_SYNC_TEST_spec_DRAFT.md` (removed).

## Question
Does adding a synchrony-restraint term to the activity-balance rule (OPEN-27 successor; baseline result in `docs/receipts/h_balance_test`) reduce rho and population Fano relative to baseline, without losing the rate restraint? And is any rho change beyond what rate compression alone produces?

Motivation (receipts, n = 2 seeds): balance-only cut bar_r (A 9.4 to 7.0 Hz, B 22.7 to 15.1 Hz) but raised mean rho (B 0.02 to 0.19) and population Fano (B 7.5 to 20.4). H4 failed in 4/4 runs.

## Estimand
Per condition and seed, differences in bar_r, per-neuron rate SD, class rates, mean rho (100 ms bins), population Fano (10 ms), and H by rate bin for: (i) sync arm minus baseline, (ii) sync arm minus balance-only arm, (iii) balance-only arm minus rate-matched baseline (confound control, below).

## Mechanism (proposed)
Rule `activity_balance_sync` (`jomission/dynamics/activity_balance_sync.py`), per-neuron aux with 2 coordinates: `rate` (slow trace, tau_r = 1000 ms, as in the balance rule) and `rate_fast` (tau_f = 10 ms). Population traces are their means:
```
tau_f dr_fast_i/dt = r_inst_i - r_fast_i          tau_f = 10 ms
P_f = mean_j r_fast_j        P_s = mean_j r_j     (slow trace; the population rate trace at tau_s = tau_r = 1000 ms)
sigma(t)     = max(0, P_f - (1 + g) * P_s) / max(P_s, eps)    (relative burst excess, dimensionless)
```
H update, neuron i (adds to the existing balance rule):
```
tau_H dH_i/dt = (1 - H_i) - kappa * delta_i * (tau_H/1000) - kappa_s * sigma * c_i * (tau_H/1000)
c_i(t)        = r_fast_i / max(P_f, eps)      (share of the burst; mean_i c_i = 1)
db_i/dt       = kappa_I * dH_i/dt
```
So bursts (population rate transiently above its own slow baseline) push H below 1 in proportion to each neuron's share of the burst, and drive_bias goes negative. Distribution-free: no band; depends only on the population trace.

Proposed defaults: g = 0.5, kappa_s = 0.5 (1/s per unit sigma), eps = 1e-3 Hz; balance parameters unchanged from `DEFAULT_PARAMS`.

## Decisions (defaults taken by owner command)
- **D1 signal**: population burst excess (above) (chosen) vs local (per-area, per-layer) burst excess (not run); population chosen because the failed metric is population Fano and node-scope rules cannot see structure without extra aux.
- **D2 weighting c_i**: share of the burst (above); uniform c_i = 1 not run.
- **D3 actuation**: additive drive_bias only (channel B ruling).
- **D4 gain**: single kappa_s = 0.5 fixed at sealing; no grid.
- **D5 metric target**: mean rho at 100 ms bins (primary, H6) plus Fano; mean rho at 10 ms bins recorded as secondary (not tested).

## Arms
1. Baseline (HDP off).
2. Balance-only (existing rule, `DEFAULT_PARAMS`; re-run, must reproduce the receipt spikes exactly, else report).
3. Balance + sync (this spec).
4. Rate-matched baseline (HDP off, condition B only): tonic_drive_scale tuned so bar_r matches arm 2 within 3% on a tuning seed (seed 2), then run on seeds 0, 1. Separates "rho rises because rates are compressed" from "rule fails to restrain synchrony". Target = balance-only bar_r (condition B, seed 2). Bisection of tonic_drive_scale on [0.6, 0.9], 8 iterations, best iterate kept, all recorded.

Conditions A (tonic 0.6), B (tonic 0.9); seeds 0, 1 (tuning seed 2 only, arm 4); 2000 ms, dt 0.1 ms; network `build_jomission_model(n_per_area=100, seed=0)`; background as in H_ACTIVE_TEST. Runs: 2 conditions x 2 seeds x 3 arms + 2 (arm 4) = 14, plus 9 tuning runs. Code: `scripts/h_sync_test.py`; metrics from `scripts/h_active_test.py`.

## Hypotheses (each can fail)
- H5: population Fano(sync) < Fano(balance-only) in every run.
- H6: mean rho(sync) < mean rho(balance-only) in every run.
- H7: bar_r(sync) < bar_r(baseline) in every run (rate restraint kept).
- H8: Fano(sync) <= Fano(baseline) in condition A runs (full hypersynchrony restraint; stricter).
- H9 (confound): min over seeds of rho(balance-only, B) minus max over seeds of rho(rate-matched, B) > |rho_rm(seed 0) - rho_rm(seed 1)|. If not, the balance-only rho rise is rate compression, not a rule defect.
- H5 to H8 hold only if true in every run in their scope; verdicts are computed in `summary.json`.
- Report per-run differences and relative changes; n = 2 seeds, no inferential test; spread across seeds only. Three-way arm comparison, no multiplicity correction claimed.

## Controls (all must pass before interpreting metrics)
1. Null: kappa_s = 0 reproduces the balance-only spikes exactly (200 ms, seed 0).
2. Baseline determinism (200 ms repeat). Also: arm 2 reproduces the h_balance_test receipt metrics exactly (bar_r, rho, Fano); otherwise reported and H5/H6 flagged.
3. Positive control (`tests/test_activity_balance_sync.py`): a synthetic population trace with P_f = 3 P_s gives sigma > 0 and dH < 0, d_b < 0; P_f = P_s gives sigma = 0 and no sync contribution. Negative control: sigma = 0 inputs leave dH equal to the balance-only rule.
4. Engagement: sync arm differs from balance-only arm; H SD > 0.
5. Rate matching (arm 4): achieved bar_r within 3% of arm 2 on the tuning seed; otherwise report arm 4 as failed and H9 undefined.
6. Unit tests of both rules pass (run inside `control`).

## Known limitations
- float32 H quantization (see H_BALANCE_TEST spec): b departs from kappa_I (H - 1) by at most about 6e-3; reported, not gated.
- Population trace uses all 400 neurons; ignores laminar structure (D1).
- `activity_balance.py` docstring changed after the balance receipts (behaviour identical); its hash differs from `h_balance_test/summary.json`; arm 2 reproduction is checked by metrics.
- aux carries three coordinates replicated per neuron (memory 3N floats; negligible).
- Proposed gains untuned; a null here shows only that this form and gain do not restrain synchrony.

## Stop rules
- Any control fails: report, do not interpret metric differences.
- Shortfall of rate matching: report arm 4 as failed.
- No change of parameters, metrics or arms after sealing.

## Not decided here
Calibrated-model status (OPEN-27); inclusion in AGSDR stage 1 (blocked).
