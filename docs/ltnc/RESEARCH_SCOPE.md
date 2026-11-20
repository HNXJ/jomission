# LTNC — predictive-routing canonical microcircuit (research workspace)

Owner direction: 2026-11-13. LTNC = Large Transformer Neural Circuit.
Home: HNXJ/jomission (hypotheses, toy models, experiments, failed/negative results). Downstream: HNXJ/jaxfne (only independently checked reusable mechanisms, APIs, and clean documentation).
Status: research design only. This does not modify the frozen omission lineage, its next_authorized_task, or any existing gate. LTNC is a separate unsealed research lane until the project authority records an executable task.

## Scientific question

Can a compact laminar predictive-coding circuit perform transformer-like contextual routing through a learned multiplicative interaction, and which of oscillatory phase, receptor/neuromodulatory state, or physically calibrated extracellular feedback provides additional causal explanatory power?

References to verify before paper-faithful replication:
- Bastos et al. 2012, *Canonical Microcircuits for Predictive Coding* (predictive-coding hypothesis, not an established wiring identity).
- Shah & Yamins 2026, *The transformer as a candidate canonical cortical microcircuit*, bioRxiv v1, https://doi.org/10.64898/2026.11.01.763457. Full equations, author FML/FOL definitions, code, and target figure not yet verified. The attention equations below are proposals, not paper quotations.

## Minimum analytically tractable model

Two interacting cortical areas, each reduced initially to prediction and error populations, with optional E/PV/SST/VIP subdivisions only if needed for a discriminating observable.

  e_i = x_i - xhat_i
  tau_i dx_i/dt = -x_i + f_i(u_i + sum_j W_ij r_ij e_j + I_field_i)
  r_ij = softmax_j[(q_i dot k_j)/sqrt(d) + beta_phi cos(theta_i-theta_j) + beta_h h_ij]
  dtheta_i/dt = omega_i + sum_j kappa_ij sin(theta_j(t-d_ij)-theta_i)
  tau_h dh_ij/dt = -h_ij + H_ij(x,e,m)

Symbols: i,j populations; x_i and xhat_i dimensionless normalized activity and prediction; e_i dimensionless error; u_i dimensionless drive; tau_i,tau_h,d_ij in seconds; W_ij dimensionless; r_ij nonnegative normalized route weights; f_i dimensionless response map; q_i,k_j dimensionless learned embeddings of dimension d; beta_phi,beta_h dimensionless; theta in radians; omega rad/s; kappa 1/s; h_ij dimensionless receptor/modulatory state; m external neuromodulatory input. I_field_i is dimensionless only after an explicit, calibrated physical-to-population transduction map; otherwise set it to zero. The update law H_ij is unspecified pending a defined mechanism. Softmax neighborhood and structural edge mask must be declared; forbidden edges cannot acquire nonzero route weights.

## Research stages and falsifiers

R0 — Source: obtain full paper/supplement/code; identify exact FML/FOL, minimal author task, parameter count, training protocol, figure estimand and selection history. Missing source => no paper-faithful claim.
R1 — Multiplicative routing: compare fixed additive, matched nonlinear additive, and learned multiplicative routing with equal wiring, trainable parameter budget and held-out relational trials. Include disabled routing and shuffled relational labels. Report effect size, seed variability, selection/testing split, computational cost.
R2 — Predictive coding: compare error-dependent routes with routes fed raw activity under matched architecture; inspect whether error propagation and predictions are identifiable separately.
R3 — Oscillations: beta_phi=0, phase shuffle preserving power, matched oscillatory power with destroyed coherence, and delay perturbation. Separate phase dependence from firing-rate/gain changes. Analytically inspect fixed points and local stability before small JAX experiments.
R4 — Neurochemistry: replace abstract h only with a defined receptor or neuromodulator model with source-supported kinetics and spatial ownership. Contrast fast synaptic gating against slow diffuse modulation. No invented fast address-specific neurotransmitter router.
R5 — Electrodynamics: derive Q -> Phi under declared geometry, conductivity, reference and boundaries; only then add compartmental ephaptic feedback. Uniform-Phi, zero-gradient, no-feedback and geometry-matched controls. Relative LFP proxies cannot establish physical coupling.
R6 — Observable-specific reduction: assess what simpler states/operators can be removed while preserving routing accuracy, phase response and field observables within preregistered tolerances.

## Separation and promotion contract

- jomission owns research protocols, experimental implementations, negative results and exploratory figures.
- jaxfne owns only general, scientifically qualified reusable execution primitives and clean docs. A task-specific success does not license a new generic API.
- A promotion requires explicit equations/units, consumer-path tests, numerical reference and independent review; physical/biological claims additionally require calibrated or empirical evidence.
- Existing jomission omission science and its sealed artifacts are unaffected; do not alter manifests/todo.json or manifests/current_state.json without reconciling the active lineage and authority.
- The existing jaxfne exploratory LTNC plan at artifacts/etudes/ltnc/RESEARCH_PLAN.md on _temp_ltnc is historical/provisional. Do not merge it as a separate research authority; after source verification, replace it with a short pointer or a qualified reusable-method document.

## Next authorized research action

Read the full Shah–Yamins preprint and author source, write a source-exact R0 extraction with provenance, then derive the smallest two-area or one-column toy that preserves the target computation. Until R0, numerical reproduction is UNKNOWN. No code execution, publication or release authorized by this document.
