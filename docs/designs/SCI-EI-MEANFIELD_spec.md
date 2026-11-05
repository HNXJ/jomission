# Mean-field of the calibrated network (plan item P1, original check; pre-registered, sealed on commit of this file)

Plan P1 (docs/PROGRESS_REVIEW_2026-11-05.md section 6) asks for a mean-field model with a known positive and a known negative check: it must reproduce the class rates and predict the PV-silent state. Q20 accepted the VIP probe in its place; this spec builds the original check anyway, so P1 is closed on its own terms.

## Facts found before sealing (observed 2026-11-05, AGS_S0_06, seed 0, 2000 ms; not part of the test)
- Kernel (jaxfne `_hdp_registrable_kernel.py`, `edge_current = w * syn_state`, `syn_next = syn_state * decay + spikes[pre]`): an exponential synapse, so the mean recurrent input to cell i is mu_i = sum_j w_ij tau_j r_j / 1000 (r in Hz, tau in ms).
- At the measured rates, mu_i lies in [-0.024, 0.044] against an emitter drive of 1.8 to 3.0.
- Zeroing every recurrent weight changes class rates by at most 0.08 Hz (E 9.70 to 9.76, PV 14.38 to 14.30, SST 11.54 to 11.56, VIP 1.70 to 1.68). Weights x20 raise PV to 20.8 Hz and x100 to 39.3 Hz, so the weight field is live (positive control for the zeroing).

## Model (neuron-resolved mean-field)
r_i = phi_i(mu_i), with mu = W r, W_ij = sum over edges j->i of w tau_j / 1000.
- phi_i is measured open loop: the same built model with all recurrent weights zeroed, Poisson background and drive unchanged. Each cell gets a constant offset dI in {-0.10, -0.05, 0, 0.05, 0.10} added to its emitter drive. Rate is counted over 500 to 2000 ms. phi_i is piecewise linear in dI, extrapolated linearly at the ends.
- Fixed points: damped iteration r <- 0.5 r + 0.5 phi(W r) from three starts (zero; phi(0); 3 x phi(0)) until max |delta r| < 1e-3 Hz or 2000 iterations. Class rate = mean over cells of the class.
- Stability and coupling: spectral radius rho of diag(phi_i') W at the fixed point (the loop gain of the linearised recurrence). rho < 1 means a stable, weakly coupled fixed point.

## Conditions (candidate AGS_S0_06, seed 0; drive changes as in the VIP and hysteresis probes)
- C_a: VIP drive f = 1, g = 1. Simulated rates: docs/receipts/vip_drive_probe/AGS_S0_06_seed0.json, row f = 1.
- C_b: f = 2.25, g = 1. Simulated: docs/receipts/agsdr_stage0_a4/AGS_S0_06.json.
- C_c: f = 2.25, global drive g = 0.5. Simulated: docs/receipts/hysteresis_probe/AGS_S0_06_seed0.json, axis G, arm up, value 0.5. This is the PV-silent state, PV 0.1 Hz.

## Checks (each can fail)
- K1 (positive): mean-field class rates agree with simulation within max(2 Hz, 25 percent) for every class in C_a and C_b.
- K2 (negative, PV-silent): in C_c the mean-field predicts PV < 1 Hz. This substitutes for the PV-silent state of SCI-EI-INTERACTION-1, which is a different lineage model. C_c is the PV-silent state of the calibrated candidate itself.
- K3 (negative, VIP): VIP < 3 Hz in C_a, and VIP in [8, 15] in C_b.
- K4 (uniqueness): all three starts converge to the same fixed point (max class difference < 0.01 Hz) in every condition.
Verdict: MEANFIELD_REPRODUCES if K1 to K4 hold; otherwise MEANFIELD_FAILS_<checks>. Reported alongside: rho per condition, and the open-loop prediction phi(0) (no coupling) as a null against which the coupling contribution is read.

## Limits
- Mean recurrent input only: recurrent fluctuations and correlations are not modelled. The Poisson background noise is inside phi.
- One candidate and one seed; the dI grid is narrow because mu is small (checked against the fixed point: |mu_i| must stay within the grid, else reported).
- Runner `scripts/ei_meanfield.py`; receipts `docs/receipts/ei_meanfield/`.
