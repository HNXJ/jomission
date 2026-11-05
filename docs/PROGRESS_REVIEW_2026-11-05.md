# Progress review: mathematics and physics of the results so far, and the plan (2026-11-05)

Status tags: OBSERVED (receipt value), DERIVED (computed here from receipt values; arithmetic shown), INFERRED (mechanism proposed, not tested). Head at writing: db15d60. Checklist 55/100 (checklist-score.md v4).

## 1. The balance controller is a leaky proportional loop

Rule (`jomission/dynamics/activity_balance.py`), per neuron i, t in ms:
- r_i' = (s_i - r_i)/tau_r; delta_i = max(0, r_i - r_hi) - max(0, r_lo - r_i)
- H_i' = (1 - H_i)/tau_H - kappa delta_i/1000; b_i = kappa_I (H_i - 1)

Fixed point: H* = 1 - kappa tau_H delta*/1000, so b* = -kappa_I kappa (tau_H/1000) delta* = -0.25 delta* (defaults kappa 0.05, kappa_I 5, tau_H 1000).
- Check (DERIVED vs OBSERVED, B seed 0): b_min -3.09 implies delta 12.4 Hz, i.e. the most-suppressed cell sits at about 22.4 Hz; H_min 0.381 vs predicted 1 - 0.05 x 12.4 = 0.38. The fixed-point relation holds to float32 resolution (|b - kappa_I(H-1)| <= 5.7e-3).
- Leak means residual error: delta* = delta_0 / (1 + 0.25 g), g = dr/db (Hz per unit bias, closed network). From class means, baseline vs balance, condition B (rough: class mean, not per cell; cells below r_hi carry no error), DERIVED: g_E 4.2, g_PV 8.2, g_SST 4.2 Hz/unit.
- Loop: two first-order stages (tau_r = tau_H = 1 s) and a fast network. Characteristic (tau s + 1)^2 + L = 0, L = 0.25 g. Poles (-1 +- i sqrt(L))/tau. DERIVED: L_PV 2.0, zeta 0.57, ring period 4.4 s; L_E 1.0, zeta 0.70, period 6.2 s. Stable, underdamped. INFERRED link: the mech detector flagged the balance arm in 2 of 4 runs (docs/receipts/h_sync_mech/summary.json, HM1 balance column); a 4 to 6 s damped ring after start-up is within its band. Not tested.

Consequence: the controller does what a leaky P-loop does. H1 to H3 (rates down, H anti-correlated with rate, corr(H, rate) = -0.92, rate SD down) are its direct algebra, not a network discovery.

## 2. Why correlations rise under balance (H4 fails)

OBSERVED (B, seed 0; h_sync_test receipts), class mean rates Hz, rho, population Fano (10 ms):

| arm | E | PV | SST | VIP | rho | Fano |
|---|---|---|---|---|---|---|
| baseline | 21.7 | 47.0 | 25.7 | 5.1 | 0.018 | 7.6 |
| balance | 15.7 | 22.2 | 17.6 | 5.1 | 0.188 | 20.6 |
| rate-matched (uniform tonic) | 15.5 | 26.5 | 17.3 | 3.0 | 0.067 | 10.1 |

- The same population rate reached two ways gives rho 0.067 vs 0.188 (H9). So the rho rise is not rate compression.
- One uniform set point r_hi = 10 Hz cuts each cell by its excess: PV -53%, E -27% (B). The rule removes the high-rate tail of each class. PV mean H 0.73 vs E 0.89 (OBSERVED, H_by_class).
- Bandwidth: the controller's corner is 1/(2 pi tau_H) = 0.16 Hz; rho and Fano are measured at 10 ms (about 16 to 100 Hz). A DC-bias channel cannot cancel fast common fluctuations. It only moves operating points.
- INFERRED mechanism (two candidates, not yet discriminated): (M1) loss of inhibitory tracking: decorrelation in balanced networks comes from inhibition following shared excitation (Renart 2010; Tetzlaff 2012); cutting the most active interneurons lowers the feedback gain. (M2) Homogenisation: compressing within-class rate heterogeneity removes a decorrelating spread of operating points.
- Single-neuron correlation transfer (de la Rocha 2007) predicts rho falls with rate. The observed rise goes the other way, so the cause is in the network, not the single cell (INFERRED).

## 3. Why the synchrony term collapsed the network

sigma = max(0, P_f - (1+g) P_s)/P_s with P_s the slow trace. At start-up P_s(t) ~ P t/tau_r, so sigma ~ tau_r/t - 1.5 (DERIVED).
- sigma = 98.5 (largest observed) corresponds to t = 10 ms: the first burst.
- Integrated H drop ~ kappa_s (tau_r/1000) ln(t_1/t_0) = 0.5 ln(100) = 2.3 for t from 10 ms to 1 s (DERIVED). This exceeds the H range from 1 to the floor 0.1, so H pins at the floor and b = kappa_I (0.1 - 1) = -4.5. Rates collapse. This is the log-divergent 1/t integral of a ratio against a lagged denominator, matching HS2/HS3 (start-up burst drives most of the collapse) and the null oscillation result (HM1 false).
- Fixed denominator (sync2, p_ref = 10): no blow-up, but no grid value beats the rate-matched baseline. Same bandwidth limit as section 2.
- Line closed by owner (Q6). Lesson (general): any rule dividing by a slow trace needs a floor or a warm-up, and must be checked at t -> 0.

## 4. The E-I regime problem is a multistability problem

OBSERVED:
- SCI-EI-INTERACTION-1: no viable cell; PV about 0.06 Hz, VIP about 0.05 Hz against bands 12-25 and 8-15.
- SCI-EI-REGIME-1: realized PV drive 5.99 clears PV rheobase 4.0, yet PV silent.
- V2.1-coupled-background: at lambda 0.65625, PV window rates depend on history: continued from the lambda-1 end state, 14.65 to 14.69 Hz (stable); continued from the lambda-0 end state, 17.05, 15.36, 2.01, then 0.07 to 0.24 Hz; fresh start, decay to 0.71 Hz. Same parameters, different end states.
- With tonic drive (H test network), PV is active (A 14.7 Hz, B 47 Hz).
- NATIVE-STATE-1: "class-specific state-space barriers".
- Builder motif gains: VIP->SST 5.0, SST->VIP 0.8, SST->PV 0.6, PV->E 1.3, E->PV 1.7. No long-range projection targets PV, SST or VIP.

DERIVED/INFERRED reading: a silent PV population with suprathreshold external drive requires net inhibition onto PV; history dependence at fixed parameters is the signature of coexisting attractors (a fold pair). The VIP-SST pair is a strong mutual-inhibition flip-flop. Candidate (INFERRED): the canonical network sits on the SST-dominant branch (VIP off, SST on, PV suppressed); the viable band lies on another branch, reached only from some initial states. If so, "no viable cell" in a fresh-start grid search is a statement about basins, not about existence. Since inter-area input reaches only E, long-range drive cannot recruit interneurons directly.

## 5. AGSDR: arithmetic of the shortfall

OBSERVED: 2 of 24 stage-0 candidates pass H05 (all-E CV denominator: 38/242, 40/242); 8 stage-1 slots; H07/H11 can only lower the count.
DERIVED: with feasible fraction p = 1/12, P(at least 8 feasible) = 0.55 at N = 96, 0.79 at 120, 0.92 at 144. Filling 8 slots needs about 6x the stage-0 sample, or a design change. This is a science trade-off for the owner. It is moot until OPEN-25 (viable regime) is settled.

## 6. Plan (ranked by information per cost; each item has a check that can fail)

| # | Work | Closing check (known positive / negative) | Blocked by | Checklist |
|---|---|---|---|---|
| P0 | OPEN-28 reruns (running, detached, one pytest at a time, worktree at db15d60) | each of 14 files: pass, fail with cause, or timeout at 2400 s; test_hdp_attractor memory | none | 48 |
| P1 | Class-level mean-field of the canonical network: 4 classes, f-I from docs/receipts/fi_single_neuron_v1.json, effective weights from builder; fixed points, Jacobian, continuation in (I_PV, I_VIP) | positive: reproduce tonic-network class rates (A baseline) within a declared tolerance; negative: predict the PV-silent state of INTERACTION-1; then: does any branch satisfy the B1 bands inside the allowed box? | none (analysis only) | 3, 13 (via OPEN-25) |
| P2 | Hysteresis test in the spiking model: up and down sweeps of lambda with state carried, 3 seeds | fold pair present iff up and down branches differ over a lambda interval at all seeds; null: sweep with state reset | sealed spec | 3 |
| P3 | Balance-rule mechanism test (sealed spec): arms E-only H; class-specific r_hi (PV 25, SST 12, VIP 15, E 10) | M1 predicts rho rise vanishes with interneurons exempt; M2 predicts it persists while within-class rate SD stays compressed | sealed spec | 19 |
| P4 | H07/H11 implement, seal, run on 24 candidates (Q12 authorised) | 4 controls; final hard-feasible set (at most 2 of 24) | none | 13 |
| P5 | Documentation gains: OPEN-08 receipts or removal; handoff with R1 to R6; fresh-clone reproduction; CI on head | each by its OPEN row | P0 for CI | 22, 26, 28, 50 |
| - | Owner decisions | AGSDR sample size vs design (section 5); goals ratification (item 33); tier 3 (item 1); Pages environment (OPEN-07) | owner | 1, 33, 44 |

Order: P0 runs now. P1 next: it is analysis only, cheap, and it decides whether the viable regime exists, the one fact that gates stage 1, AGSDR and items 3, 13 and 14. P2 and P3 need sealed specs. P4 can run alongside them.
