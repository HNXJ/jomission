# Allen V1 column replica in CAMINOS/JaxFNE form: spec (draft for owner approval, 2026-11-09)

Owner order (Hamm, 2026-11-09): replicate the model of Herrera et al. 2026 (bioRxiv 10.64898/2026.11.04.763717)
1:1 in structure, with our cells, our synapses and the CAMINOS hypothesis as mechanism; readout is our LFP proxy
with the published spectrolaminar analysis. Unsealed first pass (owner relaxation 2026-11-09: simulations first).
Status: **approved as recommended (Hamm 2026-11-09: "Approve as recommended"; D1–D5 take the recommended column).**
Amendment after approval (realization only): silencing sets VT_mV +500 and Vpeak_mV +1000 instead of EL −90. Reason
(D): under the CAMINOS drive (560·4 Hz·2 ms·2.5 nS = 11.2 nS mean AMPA) a cell with EL −90 settles near
(10·(−90) + 11.2·0)/21.2 ≈ −42 mV, above VT −50, and still fires; with VT +500 the exponential term vanishes and V
stays below E_AMPA 0 mV < Vpeak.

## Sources

| Tag | Source |
|---|---|
| P | Herrera et al. 2026, pp. 1–56 (methods, Figs 1–5, Suppl Figs 1–9) |
| A | `AllenInstitute/oscillations-biorealistic-mouse-v1-model` @ `bd25da66bfa92d097692d1c1c5d3621bcd62bcfe` (BSD-3-Clause), `biorealistic-mouse-v1-model/base_props_default/` and `edge_funcs.py`, `build_network.py`; downloaded 2026-11-09 with owner approval |
| J | JaxFNE 0.5.9 side install (`ADEX_CELL_TYPE_DEFAULTS`, `construct_adex_model`) |
| C | `jomission/laminar/caminos_e0.py` (CAMINOS constants) |
| D | derived here (script and seed named in the row) |

Hashes (sha256): `pmax_matrix_v1dd.csv` 1446631d…887d, `sigma.csv` 1249c732…38e6, `V1model_seed_file.xlsx`
e33da358…31c4, `lgn_params.csv` a23706d6…7be, `b_ratio.csv` 65fd07d5…ff.

## 1. Structure (A: `V1model_seed_file.xlsx`, column `pop_core_count`, core radius 400 µm)

| Layer | Depth µm (A) | E | PV | SST | VIP | SST/PV (D) |
|---|---|---|---|---|---|---|
| L1 | 50–169 | – | – | – | 603 (i1Htr3a) | – |
| L2/3 | 169–357 | 16784 | 552 | 400 | 913 | 0.725 |
| L4 | 357–505 | 17403 (Nr5a1 2088, Rorb 4525, Scnn1a 5221, other 5569) | 1188 | 635 | 328 | 0.535 |
| L5 | 505–665 | 9365 (IT 6555, ET 2435, NP 375) | 1067 | 941 | 188 | 0.882 |
| L6 | 665–829 | 15053 | 519 | 519 | 94 | 1.000 |

Total 66552 (A: sheet total). The 19 connectivity populations of A (L5 E kept as ET, IT, NP; L4 E merged, as in
`pmax_matrix_v1dd.csv`) are the replica's populations. JaxFNE rows: `layer` = A population key (e.g. `L5_ET`),
`cell_type` ∈ {E, PV, SOM} (J has no VIP label; see 3).

## 2. Connectivity

A rule (`edge_funcs.connect_cells`): p(pair) = pmax(pre, post) · N₂(Δxz; μ, Σ)/N₂(μ; μ, Σ) · orientation factor ·
target-size factor; lateral distance only. σ (µm, A `sigma.csv`): e→e 127.0, e→i 99.84, i→e 96.6, i→i 126.77.
Rossi correction: E source Σ = R(θ)·diag((σ/1.5)², (σ·1.5)²)·Rᵀ, μ = −50 µm along the target's tuning θ; I source
isotropic, μ = +50 µm. Orientation factor (b_ratio e→e 0.5, others 0) and target-size factor both average 1.

Replica (point neurons, no space): Bernoulli with p_eff = pmax · g(pre_ei, post_ei), g = E[N₂/N₂(μ)] over pairs
uniform in the 400 µm disk, θ uniform (D: Monte Carlo, 4·10⁵ pairs, seed 0):
g(e→e) 0.1438, g(e→i) 0.0954, g(i→e) 0.0925, g(i→i) 0.1490. All 19×19 pmax values (A) and p_eff, K = p_eff·N_pre
go to `docs/laminar_motif/data/allen_v1_blocks.json` with the derivation script. Result: 24.0·10⁶ edges.

In-degree by source class (D):

| post | K_E | K_PV | K_SST | K_VIP+L1 |
|---|---|---|---|---|
| L2/3 E / PV / SST | 172 / 599 / 554 | 99 / 158 / 79 | 100 / 146 / 82 | 18 / 49 / 48 |
| L4 E / PV / SST | 106 / 924 / 1311 | 86 / 197 / 134 | 58 / 143 / 88 | 5 / 31 / 74 |
| L5 IT / PV / SST | 132 / 642 / 793 | 61 / 125 / 69 | 36 / 92 / 61 | 12 / 33 / 60 |
| L6 E / PV / SST | 134 / 438 / 381 | 36 / 64 / 27 | 21 / 49 / 22 | 6 / 20 / 16 |

Dropped (stated, not silent): like-to-like weight and probability rules, target-size heterogeneity, Rossi asymmetry
beyond its effect on g, the 700 µm periphery (the core's cells in A also receive from it), lognormal PSP weights,
BPTT-trained weights.

## 3. Cells (J, Tahvili Table 1; AdEx, euler, dt 0.1 ms)

E, PV, SST → J `E`, `PV`, `SOM` defaults. **VIP and L1 Inh: UNSET** (J has none). Option V1 (recommended,
assumed): label `SOM` (adapting, a 4 nS, b 80 pA, EL −55 mV), i.e. VIP takes the SST parameter set and is
undriven like SST. Option V2: label `PV` (non-adapting, driven).

## 4. Synapses (C, CAMINOS)

Conductance, AMPA τ 2 ms / E_rev 0 mV, GABA τ 8 ms / E_rev −80 mV, delay 2 ms (C). A's double-alpha kinetics,
latencies and PSP amplitudes (A `tau_syn_fast/slow.csv`, `amp_slow.csv`, `syn_types_latency.csv`) are not used:
the hypothesis assigns the rhythm to counts and wiring, not kinetics.

Weights, **choice W (owner)**:
- W1 (recommended): per post population, w_E = Q_E·560/K_E and w_I = Q_INH·105/K_I (K_E, K_I: that population's
  replica in-degree from all excitatory, all inhibitory sources, after the H or A0 wiring), so each cell's mean
  recurrent conductance per receptor equals a CAMINOS cell's (560 E, 105 inhibitory inputs; C). Keeps the operating
  point E0 measured; A sets only how inputs divide among source populations. Per-synapse weights rise up to ~5×
  where K is small (L4 E: K_E 106 → w_EE 10.6 nS).
- W2: raw CAMINOS Q (2.0, 3.0, 5.0 nS) on A's in-degrees; total conductance then differs from CAMINOS by 0.2–2.3×.

## 5. Hypothesis wiring (CAMINOS) and its control

- H (mechanism, owner hypothesis): within each layer, PV→SST = 0 and SST→SST = 0, so SST→PV is one-way; VIP keeps A's
  outputs (VIP→SST, VIP→PV) and inputs. Between-layer blocks as A. Rhythm per layer set by its SST/PV count ratio.
- A0 (control): all 361 A blocks, including PV→SST and SST→SST.

## 6. Drive (C and the J capability)

CAMINOS external: 560 private Poisson sources per E and PV cell, 4 Hz, Q_EXT 2.5 nS, AMPA; SST (and VIP under V1)
undriven. This stands in for A's BKG (100 units) plus spontaneous LGN (gray screen, P). **Layer-specific thalamic
drive is not realizable in J 0.5.9's AdEx route**: `poisson_drive` takes one mapping targeted by cell-type label only
(J `inputs.py`). A's LGN targeting (A `lgn_params.csv`, `synapse_ratio_against_e4`: L2/3 E 0.63, L4 E 1.0, L5 ET 0.5,
IT 0.35, NP 0.053, L6 E 0.11, L4 PV 2.76, …; e4 LGN fraction 0.2) is recorded for a later grating condition, not
used in the spontaneous baseline. Gratings: deferred (needs per-layer Poisson in J, or a DC stand-in, owner choice).

## 7. Readout (v2, parent ruling R5 2026-11-10; owner: LFP, not rates)

Primary proxy: RWS as `scripts/e2_motif_figure.py` (Mazzoni et al. 2015, the E2 proxy, so the two are comparable):
|I_AMPA(t − 6 ms)| − 1.65 |I_GABA(t)| per layer, I = g (V − E_rev), V fixed −65 mV, summed onto that layer's E cells;
recurrent input only; L5 E = ET+IT+NP. Secondary row, same analysis: the sum form |g_AMPA·(V−0)| + |g_GABA·(V+80)|
as `scripts/run_e2_stack.py`. The v1 text (sum form as primary) is superseded; it stays in git (3f3ee7f).
Sources at layer mid-depths (A): L2/3 263, L4 431, L5 585, L6 747 µm (L1 has no E). Virtual probe: 21 contacts evenly
over 50–829 µm (38.95 µm pitch); Gaussian volume conduction σ = 0.15 × 779 µm = 117 µm (the E2 value 0.15 in
cortical-thickness units; assumed). Analysis as `scripts/e2_motif_figure.py`: discard 2 s, 1 kHz, non-overlapping 1 s
windows, DPSS multitaper half-bandwidth 2 Hz, RP(c, f) = P/max_c P, bands 10–30 and 50–150 Hz, panels d–i layout.
Depth axis rel. mid-L4 in cortical-thickness units (779 µm). r_proto against the observed LANSA map (report only)
uses a probe at the LANSA bin centres, because that map is bin-by-bin (§14).

## 8. Experiments (P, in our form)

| Id | Condition | Realization |
|---|---|---|
| X0 | baseline, spontaneous | H and A0 |
| X1 | silence PV / SST / VIP (all layers) | `param_overrides` VT → +500, Vpeak → +1000 mV on the triple (stands in for P's hyperpolarizing current; see amendment); check: 0 spikes in the silenced population |
| X2 | silence one layer (each of 5) | same override on all its populations |
| X3 | isolated layer (L2/3, L4, L5, L6) | all other layers silenced |
| X4 | L4↔L6 swap | L4 takes L6's counts and within-layer p_eff (and vice versa), run isolated as P Suppl Fig 9 |

Run plan: 20 s, discard 2 s, seed 1 (E2 convention); X0 first, inspect, then X1–X4.

## 9. Predictions and falsifiers (D from the hypothesis; inferred)

1. Isolated-layer peak frequency decreases with SST/PV: L4 (0.535) > L2/3 (0.725) > L5 (0.882) > L6 (1.000).
   Falsified if the order of the four isolated-layer LFP peak frequencies is not this one.
2. SST silencing removes the slow (10–30 Hz) component in every layer and leaves PV-driven gamma; P reports SST
   silencing *raises* 10–20 Hz power. Disagreement here discriminates the CAMINOS mechanism from A's.
3. PV silencing abolishes 50–150 Hz power (agrees with P).
4. X4: in H, the swap carries the rhythm with the counts (L4 with L6 counts loses gamma). P attributes L4 gamma to
   PV→E kinetics, which the replica holds uniform; a swap that moves the rhythm here, with kinetics fixed, supports
   the count mechanism.
5. A0 vs H: if A0 (with PV→SST, SST→SST) gives the same laminar motif, one-way SST→PV is not necessary.

## 10. Owner decisions

| # | Choice | Recommended |
|---|---|---|
| D1 | size: full core 66552 cells, 24.0·10⁶ edges, vs a scaled column with in-degree kept | full core |
| D2 | VIP and L1 Inh parameters | V1 (SST set, undriven) |
| D3 | weights | W1 (CAMINOS total conductance per class) |
| D4 | wiring | H primary, A0 control |
| D5 | gratings / layer-specific LGN drive | defer; spontaneous first |

## 11. Build and smoke (2026-11-09T23:44Z, parent 3f3ee7f)

Code: `scripts/allen_v1_import.py` → `data/allen_v1_blocks.json` (N 66552, g as in 2, MC seed 0);
`jomission/laminar/allen_v1_replica.py`; `scripts/run_allen_v1_replica.py`. Smoke at scale 0.1 (6656 cells,
full-size in-degree kept, no block capped), wiring H, seed 1, 0.5 s, JaxFNE 0.5.9, CPU:

| Run | Edge check | Silencing | Rates (Hz) | Wall |
|---|---|---|---|---|
| baseline | 353 blocks examined (361 − 8 H zeros), 0 with abs z > 5; 2 386 493 edges vs 2 386 008 expected | – | E 1.6–4.7 (L5 NP 12.8); PV 20.0–22.8; SST 5.4–12.7; VIP 5.5–12.7; L1 10.4 | construct 54 s, simulate 21 s |
| PV silenced | same | PV 0 spikes | E 73–173, SST 74–172 (runaway without PV) | construct 406 s (cold), simulate 24 s |

Full-size cost (inferred, linear in edges and in cells × steps): construct ~10× the warm 54 s; simulate 20 s at
66552 cells ~ 21 s × 10 × 40 ≈ 2.3 h per condition on this CPU.

Chunked runs (2026-11-10). `model.simulate` on AdEx records V every step: 20 s at 66552 cells is 53 GB of V plus
13 GB of spikes, over this 64 GB machine. `allen_v1_replica.simulate_chunks` runs the resumable kernel
`simulate_adex_hdp_registered` (identity rule `synthetic_presyn_gain`, no plastic edge; W1 1.3–28.6 nS inside its
w_bounds 1e-3–50 nS) in 1 s chunks, carrying the full state; the runner reduces each chunk to counts, LFP inputs
and raster samples. Known positive: 0.1 scale, 0.2 s, chunks of 700 steps (3, one uneven) gave spikes identical to
`model.simulate` (7272 spikes); the 0.5 s baseline smoke through the chunked runner reproduced all 43 output
arrays of the first smoke exactly.

X0 full size (2026-11-10, `x0_H_s1_T20`, parent 478c34e): 66552 cells, wiring H, seed 1, 20 s. Edge check 353
blocks, 0 failed; 23 854 994 edges vs 23 847 017 expected. Rates (Hz): E 0.67–2.25 (L5 NP 10.8); PV 13.7–19.9;
SST 0.4–1.4 in L2/3–L5, 7.7 in L6; VIP 0.95–1.4 in L2/3–L5, 5.7 in L6; L1 5.4. Construct 72 s, simulate 3.40 h.
Spectra: `scripts/allen_v1_figure.py x0_H_s1_T20` (2026-11-10). Units: 4 E layers at mid-depths 263/431/585/747 µm;
RWS proxy and analysis from `e2_motif_figure.py` (imported); 2 s discard, 17 one-second windows; 21 contacts 50–829 µm,
σ 0.15 (117 µm). Bands as the E2 image set (alpha-beta 10–30, gamma 50–150 Hz).

| Row | M (per thickness unit) | jackknife z | shuffle p (24 exhaustive) | crossing |
|---|---|---|---|---|
| per E cell | +0.103 | +2.3 | 0.29 (7/24) | no |
| sum over E cells | +0.174 | +4.6 | 0.375 (9/24) | no |

Peak frequency is 34 Hz at every contact and every layer (outside both bands). Power is largest at the deepest contacts
(normalized R = 1.0 at ~0.6 depth); the top contacts sit near 0. The null-tested motif is not established at X0.

Review (workflow wf_a72de94b-2ac, 2026-11-10). Findings and status:
- Provenance: X0 started about 19:20 on 2026-11-09, before commit a168181 (19:24) changed `caminos_stack.lfp_weights`
  to return a tuple. The run used the array-returning version. The `git_sha` in `x0_H_s1_T20.json` (89a1272) is HEAD
  at the end of the run, not the code at its start; the start code is the tree before a168181. Fixed for later runs:
  `run_allen_v1_replica.py` records the SHA at start, and `allen_v1_replica.lfp_weights` accepts both return forms.
  Regression: the 0.5 s, 0.1-scale smoke on the current code reproduces `smoke_s0p1_H.npz` (43 arrays identical).
- Open (owner): §7 gives the sum-form proxy |g_AMPA·V| + |g_GABA·(V+80)| as run_e2_stack.py; the analysis line and
  `e2_motif_figure.py` use RWS. The figure uses RWS. The run's stored `lfp_pA` is the sum form and is not plotted.
- Open (check): the per-cell null mean (−0.10296) is almost exactly −M (+0.10291). Not yet explained.

## 12. Parent rulings (2026-11-10; the owner may override any)

Route: owner order 2026-11-09 sends questions to the parent, not to Hamm. Recorded here, dated.
- R1 JaxFNE. The 0.5.9 runs are off-authority and exploratory, as the E2 runs are; no sealed verdict comes from 0.5.9.
  Acceptance runs use the execution authority 0.5.8 with no JAXFNE_SITE. If 0.5.8 cannot construct the model: STOP and
  report. The authority is not edited.
- R1a Acceptance adds r_proto of the X0 map against the observed LANSA map (`docs/laminar_motif/data/lansa_lfp_Y.json`).
  Report only; not a gate.
- R2 Authorization confirmed: SCI-LAMINAR-MOTIF-1 names the Allen replica beside E2. `next_authorized_task` stays
  SCI-AGSDR-CALIBRATION.
- R3 Seal the revised spec (criteria, brackets, stop states, readout) by commit and push before X1 to X4. X0 stays
  unsealed and exploratory.
- R4 Provenance: `x0_H_s1_T20` is annotated additively (§13 run status); its JSON is not rewritten. The first sealed
  run is an X0 rerun at the sealed SHA.
- R5 Readout: RWS primary, sum form secondary (§7 v2).
- Acceptance: the reviewer ACCEPT covers this spec as revised by R3 and R5 (v2), not the v1 §7.

## 13. Seal v2 (criteria, brackets, stops)

Criteria: the replica build passes; these are not science verdicts.
- C1 Edge check on jaxfne 0.5.8: `edge_check.failed` empty (|z| ≤ 5 per block).
- C2 Silencing, non-vacuous: `scripts/run_allen_v1_replica.py 1 0.5 check_H_silPV H 0.1 PV` on 0.5.8 silences the four
  PV populations (L2/3, L4, L5, L6) and must give 0 spikes in each. The unsilenced control `check_H` fires in those
  populations (L2/3 PV 21.8 Hz, L5 PV 20.0 Hz), so the check can fail. Used for X1 to X3 as well.
- C3 Smoke equality: `scripts/run_allen_v1_replica.py 1 0.5 check_H H 0.1` on 0.5.8 reproduces the 43 arrays of
  `results/allen_v1/smoke_s0p1_H.npz` (0.5.9 run) exactly.
- C4 Independent reviewer ACCEPT of the sealed commit (this spec, `allen_v1_replica.py`, `run_allen_v1_replica.py`,
  `allen_v1_figure.py`, `allen_v1_similarity.py`), and of every earlier commit in `origin/dev..dev` (parent, 2026-11-10).
- C5 r_proto reported for X0 (R1a). No threshold.

Science predictions (§9, falsifiers as written) are sealed with this version. A result against one is a finding,
not a retune.

Brackets (fixed for X0 to X4): wiring H (A0 is the control in §8), seed 1, 20 s, discard 2 s, dt 0.1 ms, 66552 cells,
E layers at 263/431/585/747 µm, σ 117 µm, probe 50–829 µm (21), bands 10–30 and 50–150 Hz, exhaustive shuffle (24),
silencing VT +500 mV and Vpeak +1000 mV (§ amendment).

Stops:
- S1 A result on jaxfne other than 0.5.8 is never a verdict on the authority (R1, ruling b, 2026-11-10). The runner
  records `authority: off-authority (0.5.9 side install), exploratory` for such runs. The authority verdict waits for
  OPEN-37 (owner). When it lands, X0 to X4 are rerun at the authority under this same seal; the spec does not change.
- S2 Any of C1 to C4 fails. Stop at the first failure.
- S3 Construct or simulate fails on 0.5.8. Status 2026-11-10: the 0.5.8 kernel has no resumable carry (no `state` or
  `return_state` in `simulate_adex_hdp_registered`), and `Model.simulate` sets `record_v=True` for AdEx. Full-size
  runs on 0.5.8 are blocked. Ruling (b), 2026-11-10: X0 rerun and X1 to X4 run on the 0.5.9 chunked route, labelled
  off-authority; the authority verdict waits for OPEN-37 (S1).
- S4 A whole-run call that would record more than 2 GB of V is refused.
- S5 Any parameter, criterion or bracket change is a new lineage, not an edit here.

Run status (additive; the records are not rewritten):

| Run | jaxfne | Status |
|---|---|---|
| `smoke_s0p1_H`, `smoke_s0p1_H_silPV` | 0.5.9 | exploratory, off-authority; reference for C3 |
| `x0_H_s1_T20` | 0.5.9 | exploratory, off-authority; start tree predates a168181 (19:24); `git_sha` 89a1272 is end-of-run HEAD (R4) |
| `check_H` | 0.5.8 | acceptance smoke (C1, C3 pass on 0.5.8, 2026-11-10); unsilenced control for C2 |
| `check_H_silPV` | 0.5.8 | acceptance, PV silenced (C1, C2 pass; 0 spikes in the four PV populations) |
| X0 rerun (`x0rerun_H_s1_T20`), X0 A0, X1 to X4 | 0.5.9 chunked | authority: off-authority (0.5.9 side install), exploratory; queued after the seal's reviewer ACCEPT (ruling b) |

## 14. Results since v1 (2026-11-10)

- 0.5.8 acceptance, `check_H` (0.1 scale, 0.5 s, seed 1, H): `checks_pass` true in `check_H.json`; edge check 353
  blocks examined, 0 failed, 2 386 493 edges; 43 arrays identical to `smoke_s0p1_H` (15 452 spikes). Path: whole-run
  `model.simulate`, because 0.5.8 has no resumable kernel (S3).
- 0.5.8 silencing, `check_H_silPV` (PV silenced): `checks_pass` true; silenced list the four PV populations; 0 spikes in
  each (C2). The control `check_H` fires there. Replaces the vacuous C2 check on `check_H`.
- Reviewer (read-only, 2026-11-10): b98d8fd ACCEPT with caveats; 3f3ee7f, f8cc9cf, 478c34e, b1d8b36, 672a963 each ACCEPT.
  Caveats closed here: C2 (above); C4 names the runner; S1 has a runner label, not a code gate; §12 wording (the
  owner-order route and the R2 authorization are recorded in the todo text for SCI-LAMINAR-MOTIF-1); "24.0·10⁶" is the
  expected full-size count, realized 23 854 994 in X0.
- 0.5.8 full size: blocked (S3). `simulate_adex_hdp_registered` on 0.5.8 has no `state`, and `Model.simulate` records V
  for AdEx, 53 GB at X0 size. The 0.5.8 execution authority cannot run X0 to X4 as specified.
- Null convention (recomputed; the code is `e2_motif_figure.score`): shuffle statistics are stored raw, and p is the
  share of the 24 relabelings ≥ M. Recomputed p equals the code's p on X0 and on the checks below. Known positive
  (planted alpha-beta rising with depth, gamma flat): M +0.411, raw null mean −0.157, p 1/24. Known negative (identical
  flat spectra): M 0, p 1. The stored null is not centred on M and is not stored as differences, so the difference
  hypothesis is refuted. The X0 mean −0.10296 ≈ −M is not structural: the positive case gives −0.157 against +0.411.
  Its cause is unexplained; p is unaffected. With 4 units the smallest attainable p is 1/24.
- r_proto, X0 against the observed map (report only; `scripts/allen_v1_similarity.py`, LANSA-bin probe):
  per E cell −0.009, sum over E cells +0.152, null95 ≈ 0.25 (E2 method). Sum row: no similarity beyond chance (0.152
  below null95). The spec-probe values (0.335, 0.465) are not bin-aligned to the observed map and are not similarity
  results.
- Cross-version identity, 0.5.8 against 0.5.9, on the full 19-population config (0.1 scale, 0.5 s, seed 1, H): 43 arrays
  identical (`check_H` against `smoke_s0p1_H`, 15 452 spikes). Full-size pair (0.5 s, 66 552 cells) queued, not run.
- Scorer units (x0_H_s1_T20, 0.5.9 side install, 20 s, discard 2 s; off-authority, exploratory): pass false. The
  shared scorer (`scripts/e2_scorer.py`, unchanged) runs on four units, the E layers L2/3, L4, L5, L6 at depth
  relative to mid-L4 {−0.216, 0, 0.198, 0.406} (thickness units; span 0.622). Interneurons are not units; they act
  through the conductances onto E. Verdict: crossing false; r_ab 0.777; r_gm −0.163; unit-order rank 10 of 24;
  r_proto 0.152 against smooth p95 0.675 and identical-unit control 0.135; halves agree (se1 0.075, se2 0.013).
  The p95 is not comparable across unit sets: the 3-unit stack's p95 is 0.380, and this one is high because four
  units span only 0.622 depth units. Adapter: `scripts/allen_v1_units.py` (commit pending review).

## 15. Screening amendment: 5 s runs (2026-11-10; parent ruling; owner Hamm)

- Purpose and label: 5 s runs are a screen. Label every record and figure "screening". A condition that appears to
  move the motif gets a 20 s confirmation run before any claim.
- Bracket (additive to §8 and §13; those stay the authority for any motif claim): T 5 s, seed 1, discard 2 s (not
  shortened), dt 0.1 ms, 1 s windows, n_windows about 3 (5 s − 2 s = 3 s of data). Full size, 66 552 cells, wiring H
  for the comparator; conditions as §8.
- Comparator: the H run at 5 s (`x0rerun_H_s1_T5`) runs first and is the only comparator for the 5 s conditions.
  Never compare a 5 s result with a 20 s result.
- Reporting: every number with its jackknife SE (motif `jackknife_se`; scorer halves `se1`, `se2`, `se_diff`).
- Units and pass rule: the four E-layer units above, the shared scorer unchanged, unit-order rank over 4! = 24. The
  mapping is fixed: no interneuron units.
- Authority: 0.5.9 runs are off-authority and exploratory (ruling R1); the authority verdict waits for OPEN-37.
- Order: the 20 s X0 run first (running), then the full-size 0.5 s identity pair (0.5.8 and 0.5.9), then the 5 s queue.
- Record, x0_H_s1_T20 (20 s, 0.5.9 side install, off-authority, exploratory): partial. The sum-row contrast is positive,
  M 0.174 ± 0.038 (jackknife SE), z 4.56, but crossing is false. Gamma depth profile wrong: r_gm −0.163 (leg 2
  fails), r_ab 0.777. Not a units or readout limit: a free diagonal four-unit fit reaches r_gm 0.835 against the observed
  map (`tests/test_allen_v1_units.py`), and the parent's 3-unit sigma-0.15 ceilings are r_gm 0.853 (coherent) and 0.657
  (incoherent). Read as the simulated spectra, not the readout.
  Gamma depth profile (50–150 Hz mean relative power, scorer contact grid, `scratchpad/_temp_gamma_peak.py`): peak at
  0.000 (mid-L4), 0.99; falls to 0.53 at +0.25 and rises again to 0.59 at +0.35 (deep shoulder); at −0.20 it is 0.51.
  Per-run lines from now on carry the gamma peak depth and r_gm (parent request, 2026-11-10).

## 16. Condition X1b (proposed by the jomission session, 2026-11-10; exploratory screen, no owner ruling)

- X1b: PV silenced in L5 and L6 only (`SILENCE=PV@L5,L6`), 5 s, screening label, seed 1, wiring H, scale 1, discard 2 s.
  Silenced populations: L5_PV, L6_PV; all other populations, all edges and the drive spec unchanged. Silencing uses the
  §8 thresholds (VT +500 mV, Vpeak +1000 mV).
- Comparator: the H run at 5 s (`x0rerun_H_s1_T5`), and X1 PV at 5 s (all-layer PV) beside it. Never compare a 5 s
  result with a 20 s one. Every number carries its jackknife SE.
- Order: appended to the end of the queue, after X3 L6 at 5 s. The queue order is not changed.
- Test: `tests/test_allen_v1_silence.py` (silence selection; edges, drive and non-silenced parameters identical to the
  unsilenced build; silenced populations spike 0; other populations still spike).
- Run from the worktree's code (`_temp_allen_v1_screen`), not the shared dev checkout, so the option is present.

## 17. Run records (2026-11-10)

Each line: tag; wall time; sum-row M ± jackknife SE; shuffle p; r_proto (sum, LANSA-bin probe); gamma peak depth; r_gm;
authority label. Records are the runner's `<TAG>.json` and `<TAG>_motif.json`, `<TAG>_similarity.json`.
r_proto here is the `allen_v1_similarity.py` LANSA-bin probe, sum row (`<TAG>_similarity.json`). Checked: its value
equals the e2_scorer sum-row r_proto exactly (0.15179058880856583 in both, x0rerun). Both use the same CHANNELS grid and
unit spectra, so they are one number, not two; the scorer's pass rule is not applied to the replica here.

r_gm is given with its band. Image band 50–150 Hz (the verdict band, `e2_motif_figure.FIG_BANDS`): from the scorer verdict
`results/allen_v1/scorer/<TAG>_verdict.json`. Manuscript band 31–150 Hz: from `<TAG>_similarity.json`
(`lansa.sum_over_cells.r_gm`, which equals the scorer's `r_gm_manuscript`).

- x0_H_s1_T20 (first X0, sha 89a1272 end-of-run, 20 s): see §14; M 0.174 ± 0.038, p 0.375, r_proto (sum, LANSA-bin probe)
  0.152; gamma peak 0.000; r_gm image −0.163 (`x0_H_s1_T20_verdict.json`), manuscript −0.074; off-authority, exploratory.
- x0rerun_H_s1_T20 (X0 rerun, sha 734fede, 20 s): wall 7866.5 s; M 0.1738 ± 0.0381; p 0.375; crossing false; r_proto
  (sum, LANSA-bin probe) 0.152; gamma peak 0.000; r_gm image −0.163 (`x0rerun_H_s1_T20_verdict.json`, scorer run on the
  rerun), manuscript −0.074; off-authority (0.5.9 side install), exploratory. Total spikes 3 485 548, equal to the first
  X0 run.
- Identity pair (full size, 0.5 s, 66 552 cells, seed 1, H; sha 519f1ed in the dev checkout): id58_full_H_s1_T0p5
  (jaxfne 0.5.8, authority) wall 211.8 s; id59_full_H_s1_T0p5 (0.5.9, off-authority, exploratory) wall 229.7 s. All 43
  npz arrays identical; 151 017 spikes in each. No motif, shuffle, r_proto, gamma or r_gm: the figure step needs a
  2 s discard, and these runs are 0.5 s (figure log: reshape error on size 0). Scope: 0.5 s; equality beyond 0.5 s
  not tested (the 20 s equality is INFERRED).
- x0rerun_H_s1_T5 (screening, 5 s; H, seed 1; sha 519f1ed; off-authority (0.5.9 side install), exploratory): wall 1835.5 s
  (construct 43.6, simulate 1786.3). Sum-row M image 0.092 ± 0.126 (jackknife SE), shuffle p 0.333; manuscript band
  M 0.108 ± 0.152, p 0.375. crossing_2se false; n_windows 2. Why 2, not 3: the 3 s after discard (30 000 samples at
  0.1 ms) lose the RWS delay d = 6 ms at the end (`e2_motif_figure.rws` returns T − d samples), leaving 2 994 ms, so
  only two whole 1 s windows. The 20 s run has the same effect: 17 windows for 17.994 s, not 18. r_proto 0.079 (LANSA-bin
  probe = scorer sum-row). Gamma peak depth 0.000 (contact 12). r_gm image −0.256 (`scorer/x0rerun_H_s1_T5_verdict.json`);
  manuscript −0.150 (same file, `r_gm_manuscript`). Comparator x0rerun_H_s1_T20 (20 s): M 0.174 ± 0.038, p 0.375,
  r_proto 0.152, r_gm −0.163. Not comparable across T: 5 s is a screen.
- Screen resolution, 5 s screens (decision (a), spec's call; set before any X-vs-baseline read; no analysis change):
  n_windows = 2 gives one jackknife df. Per-run 2·SE: image M 0.25 (0.126), manuscript M 0.30 (0.152). A 5 s screen
  flags |ΔM| above about 0.25 for a single run against zero, and does not resolve an X-vs-H difference without its own
  SE: the two runs share seed 1 and drive, so their SEs cannot be added as independent. r_gm has no SE in this record:
  the scorer's halves have 1 window each, so se1/se2/se_diff are NaN and `agree` is false by rule. The 5 s scorer pass
  is therefore not attainable as written; 5 s rows report r_gm and crossing as descriptive only. Any X condition that
  moves M by less than about 0.25 (image) is reported as unresolved, not as no effect.
- x0rerun_A0_s1_T5 (screening, 5 s; A0 control wiring, seed 1; sha 0bf4b62; off-authority (0.5.9 side install), exploratory):
  wall 1819.6 s (construct 44.6, simulate 1769.4). Sum-row M image 0.159 ± 0.024 (jackknife SE), shuffle p 0.083
  (exhaustive, 24 permutations); manuscript band M 0.184 ± 0.012, p 0.042. crossing_2se true; n_windows 2. r_proto −0.138
  (LANSA-bin probe = scorer sum-row). Gamma peak depth +0.400 (contact 20, nearest unit L6). r_gm image −0.568
  (`scorer/x0rerun_A0_s1_T5_verdict.json`), manuscript −0.456. Sha check: H (519f1ed) and A0 (0bf4b62) differ only in
  `docs/laminar_motif/FINDINGS_2026-11-10.md`, `scripts/e2_t1_atlas.py` and `tests/test_e2_t1_atlas.py`; no replica code
  changed, so H and A0 are comparable as runs. The 5 s resolution in the paragraph above applies to every value here.
  Notes (parent review, ACCEPT): manuscript p 0.042 = 1/24 is the floor of the exhaustive shuffle, the smallest p this test
  can give, not a strong result. The SEs 0.012 and 0.024 have 1 df at n_windows 2, so they are no more trustworthy than
  H's 0.126. sha 0bf4b62 is a local dev commit of the parent, pushed after its own review. Reading, descriptive only at
  5 s: A0 puts gamma deepest (+0.40) and r_gm −0.57, worse than H (r_gm −0.26); consistent with H as the better wiring.
- x1_PV_H_s1_T5 (X1, screening, 5 s; H wiring, PV silenced at all layers, seed 1; sha 0bf4b62; off-authority (0.5.9 side
  install), exploratory): wall 1820.6 s (construct 42.5, simulate 1772.6). Sum-row M image 0.048 ± 0.077 (jackknife SE),
  shuffle p 0.500; manuscript band M 0.060 ± 0.064, p 0.375. crossing_2se false; n_windows 2. r_proto 0.388 (LANSA-bin
  probe = scorer sum-row). Gamma peak depth −0.150 (contact 9). r_gm image +0.172 (`scorer/x1_PV_H_s1_T5_verdict.json`),
  manuscript +0.284. Against the H baseline (x0rerun_H_s1_T5: r_gm image −0.256, gamma peak 0.000): r_gm moves +0.43 and
  the gamma peak moves to the upper L4/L2-3 side (−0.15). r_gm has no SE at 5 s; the shift is unresolved. Descriptive only.
- Gamma power check, X1 PV vs H vs A0 (5 s; `scratchpad/_temp_gamma_power.py`; the window-mean auto-spectra of the
  four E-layer units; imaginary part of the spectra at most 1e-17 of the real part, so the real part is used). The r_gm readout
  normalises each frequency across contacts (`e2_scorer.rp_map`), so r_gm does not depend on absolute power. Absolute
  50–150 Hz power does not fall under PV silencing; it rises: L2/3 2.9e15 (H) → 3.4e17 (X1 PV), L4 8.8e15 → 1.1e17,
  L5 1.5e15 → 5.6e16, L6 1.1e16 → 1.0e17. The 1–10 Hz share of 1–150 Hz power goes from 4–7% (H) to 67–93% (X1 PV);
  the 50–150 Hz share from 25–51% to 1–7%. So the shift is a large slow component, not vanishing gamma. Whether the slow
  component is a population state (e.g. co-modulated bursting) is not yet checked; firing rates are the next check.
- Firing rates and E synchrony, 5 s runs, 3 s analysed (`scratchpad/_temp_rates_sync.py`; rate = spikes / (size × 3 s);
  Fano = var/mean of the pooled E count in 1 ms bins). H: E pooled 1.88 Hz, Fano 50.2; PV 13–20 Hz in every layer;
  SST and VIP 0.0–7.8 Hz. A0: E 1.76 Hz, Fano 14.8. X1 PV: E pooled 82.2 Hz (×44 H), Fano 4263; E cells 46–101 Hz in
  every layer; L1_Inh, SST and VIP 46–98 Hz; the silenced PV cells at 0 Hz. This is a runaway, disinhibited state, not a
  gamma state. The X1 PV profile (r_proto 0.39, gamma peak −0.15, r_gm +0.17) is therefore not evidence that the observed
  superficial gamma needs no PV: the model is not in a physiological state under PV silence. Read X1 PV as a stability
  failure first. The H value of Fano 50 at 1 ms is also above a Poisson value of 1, so the pooled E count is already
  co-modulated under H (shared rate fluctuations across cells); a Fano above 1 does not by itself show spike synchrony.
  Descriptive only; no pairwise correlation is computed yet.
- Identity check (default path after `caminos_stack.py` changed, 734fede → HEAD): `git archive 734fede` extracted to
  scratchpad `code_734fede`, run at 0.5 s (`id734_full_H_s1_T0p5`) and at HEAD (`idHEAD_full_H_s1_T0p5`), npz compared
  bitwise. Runs in sequence after X1b, before the 20 s run; one simulation at a time.
- Queue (2026-11-10, parent request, spec §16 rule): a 20 s confirmation of x1_PV_H_s1_T5 (`x1_PV_H_s1_T20`, H, PV all
  layers, seed 1) is queued after X1b, the last item, behind the identity check. Launched by a waiter (`_temp_driver9.sh`)
  that starts when the queue driver exits. Its comparator is x0rerun_H_s1_T20 (sha 734fede). Given the runaway state
  above, this run may not be informative beyond stability; the parent decides whether to keep it.
- Queue records, 5 s screening (all off-authority (0.5.9 side install), exploratory; gamma peak on the scorer contact grid, 0 = mid-L4):
  - x3_iso_L4_H_s1_T5 (isolate L4): wall 1827.5 s; image M 0.020 ± 0.007 (p 0.417), manuscript M −0.010 ± 0.002 (p 0.500);
    r_proto 0.432; gamma peak −0.05; r_gm image 0.334, manuscript 0.407. E pooled 0.59 Hz.
  - x2_L4_H_s1_T5 (layer L4 silenced): wall 1811.2 s; image M −0.035 ± 0.042 (p 0.500), manuscript M −0.040 ± 0.050 (p 0.458);
    r_proto −0.353; gamma peak +0.40; r_gm image −0.740, manuscript −0.690. E pooled 1.35 Hz.
  - x4_swap_iso_L4_H_s1_T5 (isolate L4, L4 and L6 swapped): wall 1734.1 s; image M 0.053 ± 0.038 (p 0.417), manuscript M 0.015 ± 0.040
    (p 0.458); r_proto 0.428; gamma peak −0.05; r_gm image 0.352, manuscript 0.421. E pooled 0.25 Hz.
  - x1_SST_H_s1_T5 (SST silenced): wall 1825.2 s; image M 0.169 ± 0.089 (p 0.042), manuscript M 0.210 ± 0.079 (p 0.000);
    r_proto −0.071; gamma peak +0.40; r_gm image −0.500, manuscript −0.362. E pooled 1.77 Hz.
  - Shuffle p convention (correction to the notes above): p = mean(null ≥ observed) over the exhaustive 24 permutations, with no +1
    (`scripts/e2_motif_figure.py:142`). p values are k/24, and 0.000 is attained (x1_SST manuscript). So the floor is 0, not 1/24;
    the earlier "1/24 floor" note is withdrawn.
  - Isolated-L4 states (E 0.25–0.59 Hz) are very quiet; their r_gm/r_proto are descriptive only.
- Correction to the p-convention note above (supersedes it). The 24 permutations include the identity
  (`itertools.permutations`, `e2_motif_figure.py:121–142`). Under "≥", the identity counts, so the intended floor is 1/24.
  A p of 0.000 (x1_SST manuscript, figure path) comes from float rounding: the identity's null value (slope of a−g over the
  depth grid) and the observed M (slope of the band difference) differ by about 1e-17, so the identity can fall just below M
  and drop out. The fix is a tolerance in the count (≥ M − 1e-12·|M|); not applied here (shared scorer code, needs review).
  `e2_scorer.permutation_rank` uses ">" against the identity's own value, so the identity is never excluded there.
- Open inconsistency, not resolved: the sum-row M of the same run differs between the figure path
  (`results/allen_v1/<TAG>_motif.json`: probe grid `probe_rel_mid_L4`, `rws_layers`) and the scorer-input path
  (`scorer/<TAG>`: `CHANNELS` grid, `mf.rws` per unit). x1_SST: image M 0.169 (figure) vs 0.010 (scorer path); manuscript
  0.210 vs 0.020. The queue lines above use the figure path for M and the scorer path for r_gm and r_proto. Until one path is
  chosen, the M values are not comparable with the r_gm values.
- E→E wiring (replica H, full size, from `av.plan`; no runs): L2/3→L2/3 p 0.00646, K 108.5, w 6.49 nS; L4→L4 p 0.00439,
  K 76.4, w 10.53 nS; L2/3→L4 p 0.00103, K 17.2, w 10.53 nS; L4→L2/3 p 0.00243, K 42.3, w 6.49 nS. Weights are set by the
  post-synaptic K (`weights_w1`), so L4 receives larger per-synapse weights.
- Cause of the M difference (x1_SST, read-only; `scratchpad/_temp_m_cause.py`): the RWS signals of the two paths are identical
  (max |Δ| = 0.0). The whole difference is the contact grid. Scorer grid (`CHANNELS`, −0.6 to 0.4 step 0.03): M 0.010
  ± 0.043. Probe grid (`probe_rel_mid_L4`, −0.49 to 0.51 step 0.039): M 0.169 ± 0.089, which reproduces the recorded figure M
  exactly. The grid alone changes M about 17×. Hypothesis, not tested: CHANNELS places contacts outside the unit depths (to −0.6),
  where the profile is flat, and the slope is taken over them.
- Scorer-path sum-row numbers (the only M to quote; CHANNELS grid, per-unit RWS, 1 s windows; shuffle p over the 24 unit orders
  with the identity counted under a tolerance, `p_tol`; `scratchpad/_temp_queue_scorer.py`):

  | run | wall s | image M ± SE (p_tol) | manuscript M ± SE (p_tol) | r_proto | r_gm image / manuscript |
  |---|---|---|---|---|---|
  | x0rerun_H_s1_T20 (20 s) | 7866.5 | −0.084 ± 0.032 (0.417) | −0.070 ± 0.031 (0.417) | 0.152 | −0.163 / −0.074 |
  | x0rerun_H_s1_T5 | 1835.5 | −0.175 ± 0.080 (0.417) | −0.127 ± 0.088 (0.417) | 0.079 | −0.256 / −0.150 |
  | x0rerun_A0_s1_T5 | 1819.6 | −0.002 ± 0.038 (0.208) | −0.004 ± 0.042 (0.208) | −0.138 | −0.568 / −0.456 |
  | x1_PV_H_s1_T5 (runaway) | 1820.6 | 0.106 ± 0.069 (0.583) | 0.113 ± 0.058 (0.500) | 0.388 | 0.172 / 0.284 |
  | x3_iso_L4_H_s1_T5 | 1827.5 | 0.014 ± 0.002 (0.417) | −0.018 ± 0.007 (0.458) | 0.432 | 0.334 / 0.407 |
  | x2_L4_H_s1_T5 | 1811.2 | −0.117 ± 0.017 (0.708) | −0.133 ± 0.023 (0.708) | −0.353 | −0.740 / −0.690 |
  | x4_swap_iso_L4_H_s1_T5 | 1734.1 | 0.022 ± 0.051 (0.458) | −0.017 ± 0.051 (0.458) | 0.428 | 0.352 / 0.421 |
  | x1_SST_H_s1_T5 | 1825.2 | 0.010 ± 0.043 (0.125) | 0.020 ± 0.035 (0.125) | −0.071 | −0.500 / −0.362 |

  All runs: off-authority (0.5.9 side install), exploratory. Gamma peaks as recorded in the lines above (scorer contact grid).
- Consequence: on the scorer path the X0 20 s H contrast is negative (−0.084 ± 0.032), not the positive 0.174 recorded from the
  figure path. The positive contrast in the X0 record and the 5 s H record (+0.092, figure path) is a probe-grid artifact and is
  withdrawn. Only X1 PV (runaway) has a positive scorer-path contrast, 0.106 ± 0.069 (1.5 SE).
- Flat-tail test (`scratchpad/_temp_flat_tail.py`, scorer inputs, per-unit RWS; span = outer unit depth ± σ = [−0.366, 0.556]).
  Correction: the CHANNELS grid is 21 contacts at step 0.05 (−0.6 to 0.4), not step 0.03 as written above.
  Five CHANNELS contacts lie outside the span (−0.6 to −0.4); none lie above it.
  | run | M CHANNELS full | M CHANNELS in span (full-grid norm) | M CHANNELS in span (set norm) | M probe grid (scorer RWS) |
  |---|---|---|---|---|
  | x0rerun_H_s1_T20 | −0.084 | +0.079 | +0.079 | +0.174 |
  | x1_SST_H_s1_T5 | +0.010 | +0.282 | +0.282 | +0.169 |
  The five outer contacts have dlt = ab − gm of 0.000 to 0.023 (H) and −0.022 to 0.000 (SST), so they are near-flat, as
  hypothesised. Dropping them still flips the H sign. Mechanism (reasoned, not decomposed numerically): the least-squares slope
  depends on the centre of the grid, and the removed points pull the centre from −0.10 to +0.03. Their near-zero values add to
  the denominator and shift the centre, which can change the sign. So "dilution only" does not hold for this slope. The sign
  is not robust to the choice of contacts.
  Non-shared contacts (no contact is common to the two grids): the probe grid adds contacts at +0.46 and +0.51, where dlt
  declines (H: 0.336 and 0.246; SST: 0.103 and 0.075). CHANNELS stops at +0.40 (H: 0.341), where the profile is still
  rising. So the top of the profile is sampled differently by the two grids, as well as the bottom.
  Reading: M on either grid is a grid-sensitive statistic, and its sign is not reliable. The scorer path is still the
  authority for the seals (per parent ruling); the M values in this section should not be used to support a motif claim.
