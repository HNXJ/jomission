# Corrected synchrony term (DRAFT; not sealed; no run authorized)

Status: draft from owner answers Q1 to Q4 (2026-11-04). Sealing = commit of the non-DRAFT file on the owner's go. Parameters other than the declared grid are fixed at sealing.

## Question
Does a synchrony term with a bounded, constant reference (no start-up blowup) restrain Fano and rho beyond what rate compression alone does, without losing rate restraint?

## Mechanism (owner Q2)
Rule `activity_balance_sync2` = `activity_balance_sync` with one change:
```
sigma = max(0, P_f - (1 + g) P_s) / r_ref,     r_ref = r_hi = 10 Hz (constant, rule parameter)
```
All else unchanged (g = 0.5, tau_f = 10 ms, c_i share weights, drive_bias channel, balance part at DEFAULT_PARAMS). Predicted removal of the start-up blowup: peak sigma at t = 0 about P_f / r_ref = 4 instead of 98.5.

## Gain (owner Q3): declared grid
kappa_s in {0.15, 0.5, 1.5}. Selection on **tuning seed 2, condition B**, window t in [0, 4000) ms: pick the smallest kappa_s meeting all of the criterion terms in HC1 to HC3 below (computed on seed 2 against its rate-matched baseline). If none meets them: stop, report "no grid value passes", run nothing on test seeds. The picked kappa_s is then fixed.

## Design
- Network, conditions A (tonic 0.6) and B (tonic 0.9), background as `H_ACTIVE_TEST_spec.md`; duration 4000 ms; window from t = 0; dt 0.1 ms; `record_weight_trace` False.
- Test seeds 0, 1, 3 (untouched by selection), conditions A and B: 6 (condition, seed) cells.
- Arms per cell: baseline (HDP off); balance-only; sync2 at the picked kappa_s; rate-matched baseline (HDP off, tonic tuned so bar_r matches sync2 within 3%).
- Rate matching: bisection of tonic on [0.3, condition tonic], 8 iterations, on tuning seed 2 per condition, target = sync2 bar_r on seed 2 of that condition; the tonic is then used on the test seeds. (Match is checked again on the test seeds and reported.)
- Metrics: bar_r, class rates, per-neuron rate SD, mean rho (100 ms bins), population Fano (10 ms), peak sigma in [0, 500) ms, rate in [0, 1000) ms (for HC4).

## Criterion (owner Q4) and hypotheses (each can fail; "all" = every test cell in scope)
- **HC1**: Fano(sync2) < Fano(rate-matched), all 6.
- **HC2**: rho(sync2) < rho(rate-matched), all 6.
- **HC3**: bar_r(sync2) <= 0.8 x bar_r(baseline) in condition B, all 3.
- **HC4** (start-up fixed): rate in [0, 1000) ms of sync2 >= 0.5 x balance-only rate in the same window, all 6.
- Supported iff HC1 to HC4 all true. Descriptive, no inferential test; seeds vary the Poisson/engine stream only (network and initial burst fixed), so effective n is small (disclosed from the start-up test).

## Controls (all must pass before interpreting)
1. Null: kappa_s = 0 reproduces balance-only spikes exactly (4000 ms, A seed 0).
2. Determinism: repeated sync2 run identical.
3. Unit tests of sync2 (equals sync when r_ref = P_s; sigma bounded by (P_f)/r_ref at P_s = 0; finite at silence) and of the balance and sync rules pass.
4. Rate matching: achieved bar_r within 3% on tuning seed (else the rate-matched arm is reported failed, HC1 and HC2 undefined).
5. Detector-free: no oscillation claims are made.

## Limits
- Grid of three; the optimum may lie between points.
- r_ref = r_hi is a choice (owner Q2), not tuned.
- Equal-rate comparison uses tonic scaling as the matching knob; other knobs (e.g. background rate) could give different rho at equal rate.
- Float32 H quantization (see `H_BALANCE_TEST_spec.md`).

## Stop rules
Any control fails: report, do not interpret. Grid selection fails: stop and report.

## Not decided here
OPEN-27 membership of any synchrony term; a local (per-area) burst signal.
