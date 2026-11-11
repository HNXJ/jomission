# Proposal only: spectrolaminar circuit + two-timescale HDP for omission

Status: **UNSEALED / NOT AUTHORIZED FOR RUNS**. Separate lineage from the LANSA preprint; no claim of physiological omission prediction. Preserve the omission firewall and canonical slot timing (531 ms; omitted stimulus = zero sensory drive during an intact slot).

## Hypothesis and estimand

Fast adaptive state captures local stimulus history and omission-onset prediction error; slow state captures context/repetition across trials. Test whether omission responses exceed a rate- and drive-matched sensory-withdrawal null, and whether the interaction of timescales predicts laminar sign/latency structure.

For population u, propose H=(h_f,h_s), with

  tau_f dh_f/dt = -h_f + F_f(r_u, s_u),
  tau_s dh_s/dt = -h_s + F_s(r_u, s_u),  tau_s >> tau_f,
  I_u(t) = I_u^0(t) + g_f h_f(t) + g_s h_s(t).

These are candidate equations, not an existing JaxFNE rule. Choose timescales from a sealed grid relative to 531-ms slots and multi-trial blocks, not from the existing `HDPConfig.tau_theta_s=1000` s without verification. Keep slow plastic Theta distinct from fast hidden H; the repository's `jomission/dynamics/hdp.py` is a specific homeostatic 2-coordinate instance, not proof of fast/slow AdEx execution.

## Minimal experiment

Compare frozen circuit with (i) HDP off, (ii) fast only, (iii) slow only, (iv) both; matched input schedule and seeds. Primary estimand: omission-minus-matched-zero-drive response in a prespecified post-omission window, stratified by layer and E/PV/SST. Secondary: gamma/alpha-beta power, onset latency, phase, H/Theta trajectories. Controls: expected zero-drive, unexpected omission, timing-jitter, shuffled sequence history, gain-matched rate control, and input-free spontaneous periods. Explicitly distinguish omission from time deletion.

## Gates

- Verify JaxFNE rule registration, state shape/locality, dH and per-spike jH semantics, units, sign, saturation, and feedback by primitive tests before coupling to the stack.
- Verify tau_f and tau_s with impulse decay, two-pulse facilitation/depression, block carryover and reset controls; no vacuous zero-drive passes.
- Smoke one seed, then seal targets and run independent seeds. Hold out sequences and recording areas for generalization.
- No changes to LANSA E0/T1/T8 seals or preprint conclusions; the current motif model fails the full map scorer and cannot be presented as a validated physiological substrate.

Acceptance: a prespecified omission-specific contrast survives matched-withdrawal and shuffled-history nulls with uncertainty and seed/area accounting; otherwise report a negative result. Owner approval required for scientific estimands and gate changes before runs.
