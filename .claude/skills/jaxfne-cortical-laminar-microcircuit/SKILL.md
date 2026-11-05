---
name: jaxfne-cortical-laminar-microcircuit
description: Theory, math and biophysics for building a laminar Izhikevich-TFNE cortical column in JaxFNE (E, PV, SST, VIP) that reproduces the spectrolaminar motif (deep alpha-beta, superficial gamma) from cell densities, with an H-state adaptation and balance term implemented through HDP. Use when designing, tuning, running or reviewing the laminar column, its wiring, its LFP motif analysis, or the Tahvili E-PV-SST gamma mechanism it borrows. Not for the omission paradigm itself (use the omission lineage specs) or for gate execution rules (use jomission-gate-runner).
---

# JaxFNE cortical laminar microcircuit

Sources (read from `docs/project-sources/laminar_motif/` if copied there; otherwise the owner's uploads):
**M** Mulvey et al., *Conserved Laminar-Cellular Architecture Predicts the Spectrolaminar Motif* (manuscript);
**T25** Tahvili et al. 2025, Cell Reports 44:116131; **T26** Tahvili et al. 2026, PLoS Comput Biol 22 (June 2026, DOI suffix pcbi.1014378);
**D** `Spectrolaminar_motif_jaxfne_v4` deck (local pilot v4, 2026-10-20).
Tags: [O] read from a source this session, [K] general knowledge not in the sources (verify before use), [H] hypothesis.

## 1. What is to be reproduced (M, D)

- Motif [O, M]: gamma (31-150 Hz) peaks in L2/3, alpha-beta (8-30 Hz) peaks in L5/6, L4 is the crossover. Conserved across 11 areas (mean r 0.76). Band conservation: high gamma 0.89, beta 0.78, alpha 0.63, low gamma 0.63, delta 0.52, theta 0.49.
- Relative power [O, M eq. 1]: `RP(c,f) = P(c,f) / max_c P(c,f)`, multitaper, 2 Hz smoothing, 1 s window (500 ms pre/post), 60 Hz interpolated out.
- Depth space [O, M]: piecewise-linear to -0.6:0.4 of cortical depth about mid-L4, step 0.05, 21 bins. X is 21x4x11 (bins, cell types PV/CB/CR/NG, areas); Y is 21x150x11.
- Densities [O, M]: PV bimodal L3 and L4 (L4 larger); CB and CR superficial unimodal; NG peak superficial L3, small peak L5/6. Conservation across areas: CR 0.81, CB 0.70, PV 0.64, NG 0.53.
- PCA [O, M]: anatomy PC1 superficial CB/CR (62.47%), PC2 L3/4 PV (28.46%); function PC1 delta/theta/gamma in L3 (71.53%), PC2 alpha-beta + low gamma, positive through L5/6 (28.35%).
- PLSR [O, M]: X_P has 11 columns = 4 types + 6 pairwise products (PV·CB, PV·CR, PV·NG, CB·CR, CB·NG, CR·NG) + 1 E/I term `z(NG) / sum z(inhibitory)`. 4 components, leave-one-area-out: mean Q² 0.42, mean r 0.71; beta coefficients across areas r 0.992. `Q² = 1 - Σ(Y_test - Y_pred)² / Σ(Y_test - Ȳ_train)²`. Beta (alpha-beta) has the highest positive loading on E/I, peaking deep; gamma loads on PV and NG, negatively on CB·CR.
- Schizophrenia perturbation [O, M]: Hedges g knock-down of PV/CB/CR; model predicts high-gamma loss, trend (p=0.054) to more beta.
- Deck baseline to beat [O, D]: 1050 Izhikevich neurons per column, fractions 483 E / 205 PV / 167 SST / 195 VIP (mean anatomy), 21 full-width bins, fractions within 0.021 of LANSA; explicit wiring only; drives E 4.0, PV 4.0, SST 0.5, VIP 25. Physiological refit rates E 4.9, PV 6.7, SST 6.4, VIP 11.2 Hz. Result: motif r of area means 0.44; held-out r 0.30 (-0.15 to 0.52); gamma above alpha-beta in 5 of 11 simulated areas (10 of 11 observed); conservation r 0.63 (obs 0.76); LOAO r 0.70 (0.71); Q² 0.37 (0.42); PCA 96/2% vs observed 72/28%. Edge/interior power 0.48 vs observed 0.86. Open: edge bins, alpha-beta depth, irregular firing (E CV 0.14, volleys). Preset drives sit below rheobase for PV (4.0) and VIP (22.6): issue P-043.
- Type map [O, M/D]: PV = PV; CB ~ SST (co-localize in primates); CR ~ VIP; NG = pyramidal (E). The map is an assumption, not an identity.

## 2. Hypothesis under test [H, owner 2026-11-05]

PV and SST density peak superficially while the SST/PV ratio peaks deep. Then gamma (35-90 Hz) originates mainly superficially (PV-led E-PV loop, T25/T26) and alpha-beta (10-25 Hz) originates deep (SST-dominated, slower, adapting). Checks before building on it:
- Compute PV, CB, CB/PV from the densities per bin; M reports PV peaking near L3/4 and CB/CR near the pia, so "PV density peaks superficially" may need restating.
- Falsifier: flatten or reverse the SST/PV ratio profile across depth with total inhibitory count fixed; the deep alpha-beta peak must move or vanish. If the motif survives, the ratio is not the cause.
- T25/T26 show SST/PV ratio up lowers frequency (about 40 to 15 Hz) and raises amplitude: direction consistent with the hypothesis [O].

## 3. Tahvili E-PV-SST mechanism (T25, T26)

- AdEx [O, T26 eq. 15-16]: `C V' = gL (EL - V) + gL Δ exp((V - VT)/Δ) - w + I`; `τw w' = a (V - EL) - w + b Σ δ(t - tj)`; spike at VT + 5Δ; reset Vreset; refractory Tref. PV has no adaptation.
- Parameters [O]. Shared: VT -50 mV, Tref 5 ms, τw 500 ms, C 200 pF, gL 10 nS, Vreset -65 mV, a = (4, 0, 4) nS for (E, PV, SST).
  T25: Δ (2, 0.5, 1.5) mV, b (130, 0, 25) pA, EL (-60, -60, -55) mV. T26: Δ (2, 0.5, 2) mV, b (50, 0, 80) pA, EL (-65, -65, -55) mV.
  Differences (owner-confirmed 2026-11-05): SST Δ 1.5 vs 2 mV and SST b 25 vs 80 pA; E EL -60 vs -65 mV and E b 130 vs 50 pA. Same in both: E Δ, all PV values, SST EL (-55 mV), a, VT, Vreset, C, gL, τw, Tref. Pick one set per run and record which.
- Synapses [O]: conductance-based, Erev_E 0 mV, Erev_I -80 mV in both. The papers use different sets (receipt: `docs/laminar_motif/item02_digest_receipt.txt`):
  - T25 (Table 2, p10): τ_E (1, 2, 5) ms for (PV, SST, E) [owner-confirmed], τ_I 5 ms, no synaptic delay in the base model (p17). Q (nS): PV→E and PV→PV 6; SST→E and SST→PV 5; E→E 1.25; E→PV 3.75; E→SST 2.5; Ext 1.25 [printed "QE→Ext"]. N = 8000 E, 800 PV, 600 SST (NI = 1400).
  - T26 (Table 2, p22): τ_d 2 ms, τ_E 2 ms, τ_I 8 ms. Q (nS): {PV, SST}→{E, PV} 5; E→E 2; E→{PV, SST} 3; Ext→{E, PV} 2.5 (Poisson 4 Hz); PV→SST = SST→SST = 0. N = 8000 E, 750 PV, 750 SST (NI = 1500), p = 0.07.
  - A run uses one paper's neuron set with the same paper's synapse set, recorded together.
- Connectivity [O]: strong SST→PV, weak or absent PV→SST and SST→SST. This asymmetry, not delay or adaptation, produces ~35 Hz with PV leading SST by ~7 ms (T25 6-7 ms, 80 degrees; T26 7-8.5 ms).
- Roles [O]: PV sets frequency and stability; SST sets amplitude and needs precise spike timing; suppressing either slows or abolishes the rhythm.
- Mean field [O, T26]: `T ν_i' = -ν_i + F_i(ν(t - τd), w_i)`, T = 15 ms, τd = 2 ms, i in {E, PV, SST}; high-conductance state, effective time constant `Tm = Cm / μG`.
- Linear reduction [O, T26, Eq. 10 matrix and parameters verified in the receipt]: `ν' = (1/τ) A ν`, τ = 20 ms, Q_SST→E = 4, Q_SST→PV = 2.05, PV leads SST by 4 ms; rows post, columns pre, PV→SST = SST→SST = 0; oscillation needs eigenvalues `λ = ±iω` plus a real root; frequency set mainly by the time constant (ω/2πτ). Linear systems have no limit cycle: sustained rhythm in the full model comes from the nonlinearity.
- VIP [O]: neither paper simulates it; both name it as the extension (VIP mainly inhibits SST). A VIP→SST disinhibition term is this lineage's addition, not a replicated result.

## 4. Izhikevich-TFNE column

- Neuron [K, Izhikevich 2003]: `v' = 0.04 v² + 5 v + 140 - u + I`, `u' = a (b v - u)`, spike at v >= 30: `v ← c`, `u ← u + d`. Starting points: RS (E) a .02 b .2 c -65 d 8; FS (PV) a .1 b .2 c -65 d 2; LTS (SST) a .02 b .25 c -65 d 2. VIP: bursting or irregular, chosen by rheobase and rate target. Verify units and dt against the JaxFNE model before use.
- Map T25 AdEx roles to Izhikevich classes: PV fast, non-adapting; E and SST adapting, SST stronger per spike (T26 b 80 vs 50 pA). Cross-check with `a, d` after the H term below replaces `u`.
- TFNE form [O, memory tfne-algebra]: `x : A : y`, `A -> (s, h0, I)`, `(h_t, x_t; s) -> (h_{t+1}, y_t)`; `H` is the H-state tensor, `h` mutable state, `s` static. Factor at specification time, flatten at execution. JaxFNE is the only simulator.
- H-state with two terms (owner 2026-11-05):
  - Adaptation `h_w`: an AdEx-like slow variable realized by HDP, `dΘ = P(H, X, Θ)`. Spike increment `b`, subthreshold coupling `a (V - EL)`, time constant `τw` (T25/T26: 500 ms). Per type: PV none; E and SST on.
  - Balance `h_b`: E/I balance, `z(E) / Σ z(inhibitory)` per bin (the M proportion term), bounded, slower than the neuronal state.
  - HDP carries this (owner 2026-11-05; JaxFNE 0.5.8 read [O]). HDP is the general `(H, X, B, Θ) --P--> (dH, dΘ)` with H an arbitrary relative biophysical state (RBS); adaptation variables are named RBS antecedents (`docs/theory/doctrine/rbs_rbd_hdp.md`). `jaxfne.hdp_rule.register_hdp_rule(HDPRuleDescriptor, step_fn)`: the step receives `H, v, u, spikes, w, ...` and returns `dH` (rate), `jH` (per-spike jump, no dt), `d_theta`/`j_theta`. Vector H via `h_shape`; plastic targets `edge_weight` and `drive_bias` (per-neuron additive current). H→x coupling is a typed gain map (protocol H1b, closed, `docs/theory/memory/protocol_h_h1b_h_to_x_gain.md`).
  - AdEx-like `w` as a registered rule: `dH = (a (v - EL) - H)/τw` rate, `jH = b` per spike (`H` in pA, or relative with `H* = 0`, the deviation convention), fed to the neuron as `drive_bias ≈ -H`, or through the H1b gain. Which feedback path (`drive_bias` vs H1b) and the unit convention are open design choices for items 8-9. [derived from the API; not yet run]
  - `drive_bias` read from `jaxfne/_hdp_registrable_kernel.py` (0.5.8) [O]: native current is `drive + sched + bias + syn + noise_coef * noise`, so a negative `bias` is an inhibitory current. `bias` is a kernel-carry coordinate: `bias ← clip(bias + dt·d_theta + j_theta, b_min, b_max)`, default bounds ±50 native units; `j_theta` has no dt factor. It is a persistent plastic parameter (carried across continuations), not a transient, so the verification in item 9 must show: (a) a spike gives a jump of the intended sign and size in native units; (b) decay time constant equals τw to within dt discretization; (c) subthreshold coupling `a (v - EL)` has the right sign; (d) the bias returns to its initial value after activity stops and starts from zero in each run (no carry-over plasticity); (e) no clipping at ±50 in the operating range; (f) `edge_weight` stays unchanged (not in `theta_targets`) and no other coordinate moves; (g) native units map to pA by a stated factor. [to verify; not run]
  - `jomission/dynamics/hdp.py` (`h_state_dim: 2`, channels `edge_weight`, `intrinsic_drive`) is the repo's homeostatic configuration, not a limit of HDP. The minimal HDP barely moves H (1.0 to 1.0024 over 1 s, `manifests/current_state.json`): a strength setting, not a capability gap.
  - Required checks: single-neuron trace against an AdEx reference for the same `a, b, τw`; HDP-off ablation; bounded H.
  - JaxFNE already ships E/PV/SST/VIP Izhikevich presets and a spectrolaminar suite (`docs/archive/suites/07_jaxfne_suite_no_2_spectrolaminar_motif.md`, `suite2_net1_config`, `jtfne.vis.spectrolaminar_suite`): reuse before writing new code.
- Column [O, D]: types placed by `layer_fractions` in 21 full-width bins; explicit connectivity only (`connectivity_mode="explicit"`); `RuntimeConfig(recurrent_backend='edge_list', enable_hdp, hdp_params)` passed to jtfne.simulation (`jomission/minimal/builder.py`).
- Wiring to declare per type pair, with T25/T26 asymmetry as the prior: E→{E, PV, SST, VIP}; PV→{E, PV}; SST→{E, PV, VIP?}; VIP→SST (disinhibition); PV→SST and SST→SST at or near zero. Layer dependence of each edge is a free design variable: declare, seal, then run.

## 5. Readout and analysis chain

LFP proxy from synaptic currents or line-source (`line_source_lfp` in jaxfne core): D found synaptic input + bipolar best (r 0.17), none beat the Gaussian proxy (0.35); edge bins remain wrong (0.48 vs 0.86). Then relative power per bin, band reductions (delta 1-3, theta 4-7, alpha 8-12, beta 13-30, low gamma 31-90, high gamma 91-150 Hz), PCA, PLSR LOAO, Procrustes. A NumPy port exists in D and matched every manuscript number on observed data [O, D]; re-run it before trusting.

## 6. Rules that bind this work (AGENTS.md)

- Seal targets, brackets, criteria and stop states before running. Do not retune after results; a changed criterion is a new lineage.
- Omission firewall: no omission outcome informs the column; the column is a building block built blind to omission.
- Visualization contract: V0 schematic, V1 initial 1 s raster, V2 final raster, V3 atlas, before and after interpretation.
- A scalar gate can miss structure the rasters show; look at the rasters.
- Provenance: JaxFNE version recorded (0.5.8 at 2026-11-05); evidence class and status are separate fields.
- Cross-source conflict (for example T25 vs T26 parameters, D vs M numbers): stop and show both.
