# H balance + synchrony term (DRAFT; not sealed; no run authorized)

Status: draft for owner review. Items marked **[OWNER]** are decisions; defaults are proposals. Sealing = commit of the non-DRAFT file after owner answers. Parameters are proposed and untuned and are fixed at sealing.

## Question
Does adding a synchrony-restraint term to the activity-balance rule (OPEN-27 successor; baseline result in `docs/receipts/h_balance_test`) reduce rho and population Fano relative to baseline, without losing the rate restraint? And is any rho change beyond what rate compression alone produces?

Motivation (receipts, n = 2 seeds): balance-only cut bar_r (A 9.4 to 7.0 Hz, B 22.7 to 15.1 Hz) but raised mean rho (B 0.02 to 0.19) and population Fano (B 7.5 to 20.4). H4 failed in 4/4 runs.

## Estimand
Per condition and seed, differences in bar_r, per-neuron rate SD, class rates, mean rho (100 ms bins), population Fano (10 ms), and H by rate bin for: (i) sync arm minus baseline, (ii) sync arm minus balance-only arm, (iii) balance-only arm minus rate-matched baseline (confound control, below).

## Mechanism (proposed)
Population rate and its fast and slow traces (Hz), shared across neurons, carried as per-neuron aux columns (replicated; `aux_layout="per_neuron"`, 3 coordinates: `rate`, `pop_fast`, `pop_slow`):
```
P_inst(t)    = sum_j spikes_j(t) * 1000 / (dt * N)
tau_f dP_f/dt = P_inst - P_f          tau_f = 10 ms
tau_s dP_s/dt = P_inst - P_s          tau_s = 1000 ms
sigma(t)     = max(0, P_f - (1 + g) * P_s) / max(P_s, eps)    (relative burst excess, dimensionless)
```
H update, neuron i (adds to the existing balance rule):
```
tau_H dH_i/dt = (1 - H_i) - kappa * delta_i * (tau_H/1000) - kappa_s * sigma * c_i * (tau_H/1000)
c_i(t)        = fast-rate coincidence weight of neuron i = r_fast_i / max(mean_j r_fast_j, eps)    [OPEN: see D2]
db_i/dt       = kappa_I * dH_i/dt
```
So bursts (population rate transiently above its own slow baseline) push H below 1 in proportion to each neuron's share of the burst, and drive_bias goes negative. Distribution-free: no band; depends only on the population trace.

Proposed defaults: g = 0.5, kappa_s = 0.5 (1/s per unit sigma), eps = 1e-3 Hz; balance parameters unchanged from `DEFAULT_PARAMS`.

## Owner decisions
- **D1 [OWNER] signal**: population burst excess (above) vs local (per-area, per-layer) burst excess. Default: population, because the failed metric is population Fano and node-scope rules cannot see structure without extra aux.
- **D2 [OWNER] weighting c_i**: share of the burst (above) vs uniform c_i = 1. Default: share; uniform is the simpler control arm if you want both.
- **D3 [OWNER] actuation**: additive drive_bias only (default, consistent with the ruling of channel B) vs also edge-weight scaling.
- **D4 [OWNER] gain**: single kappa_s fixed at sealing (default) vs a pre-declared 3-point grid (0.15, 0.5, 1.5) with the pick rule "smallest kappa_s meeting H5 and H6", chosen on seed 2 and tested on seeds 0, 1.
- **D5 [OWNER] metric target**: keep mean rho at 100 ms bins plus Fano (default), or add rho at 10 ms bins (more sensitive to fast synchrony; added as secondary metric either way).

## Arms
1. Baseline (HDP off).
2. Balance-only (existing rule, `DEFAULT_PARAMS`; re-run, must reproduce the receipt spikes exactly, else report).
3. Balance + sync (this spec).
4. Rate-matched baseline (HDP off, condition B only): tonic_drive_scale tuned so bar_r matches arm 2 within 3% on a tuning seed (seed 2), then run on seeds 0, 1. Separates "rho rises because rates are compressed" from "rule fails to restrain synchrony". Tuning by bisection on seed 2, 8 iterations max, recorded.

Conditions A (tonic 0.6), B (tonic 0.9); seeds 0, 1 (tuning seed 2 only for arm 4 and D4); 2000 ms, dt 0.1 ms; network `build_jomission_model(n_per_area=100, seed=0)`; background as in H_ACTIVE_TEST. Runs: 2 conditions x 2 seeds x 3 arms + 2 (arm 4) = 14, plus tuning runs.

## Hypotheses (each can fail)
- H5: population Fano(sync) < Fano(balance-only) in every run.
- H6: mean rho(sync) < mean rho(balance-only) in every run.
- H7: bar_r(sync) < bar_r(baseline) in every run (rate restraint kept).
- H8: Fano(sync) <= Fano(baseline) in condition A runs (full hypersynchrony restraint; stricter).
- H9 (confound): mean rho(balance-only, B) > mean rho(rate-matched baseline, B) by more than the seed spread. If not, the balance-only rho rise is rate compression, not a rule defect.
- Report per-run differences and relative changes; n = 2 seeds, no inferential test; spread across seeds only. Three-way arm comparison, no multiplicity correction claimed.

## Controls (all must pass before interpreting metrics)
1. Null: kappa_s = 0 reproduces the balance-only spikes exactly (200 ms, seed 0).
2. Baseline determinism (200 ms repeat).
3. Positive control (unit test): a synthetic population trace with P_f = 3 P_s gives sigma > 0 and dH < 0, d_b < 0; P_f = P_s gives sigma = 0 and no sync contribution. Negative control: sigma = 0 inputs leave dH equal to the balance-only rule.
4. Engagement: sync arm differs from balance-only arm; H SD > 0.
5. Rate matching (arm 4): achieved bar_r within 3% of arm 2 on the tuning seed; otherwise report arm 4 as failed and H9 undefined.
6. Unit tests of the rule pass.

## Known limitations
- float32 H quantization (see H_BALANCE_TEST spec): b departs from kappa_I (H - 1) by at most about 6e-3; reported, not gated.
- Population trace uses all 400 neurons; ignores laminar structure (D1).
- aux carries three coordinates replicated per neuron (memory 3N floats; negligible).
- Proposed gains untuned; a null here shows only that this form and gain do not restrain synchrony.

## Stop rules
- Any control fails: report, do not interpret metric differences.
- Shortfall of rate matching: report arm 4 as failed.
- No change of parameters, metrics or arms after sealing.

## Not decided here
Calibrated-model status (OPEN-27); the pick among D4 grid values beyond the stated rule; inclusion in AGSDR stage 1 (blocked).
