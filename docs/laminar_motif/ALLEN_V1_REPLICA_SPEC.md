# Allen V1 column replica in CAMINOS/JaxFNE form: spec (draft for owner approval, 2026-11-09)

Owner order (Hamm, 2026-11-09): replicate the model of Herrera et al. 2026 (bioRxiv 10.64898/2026.11.04.763717)
1:1 in structure, with our cells, our synapses and the CAMINOS hypothesis as mechanism; readout is our LFP proxy
with the published spectrolaminar analysis. Unsealed first pass (owner relaxation 2026-11-09: simulations first).
Status: **values below await approval; no run beyond a ≤1 s smoke until approved.**

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

## 7. Readout (owner: LFP, not rates)

LFP proxy per layer as `scripts/run_e2_stack.py`: |g_AMPA·(V−0)| + |g_GABA·(V+80)| onto that layer's E cells,
V fixed −65 mV, recurrent input only; L5 E = ET+IT+NP. Sources at layer mid-depths (A): L2/3 263, L4 431, L5 585,
L6 747 µm (L1 has no E). Virtual probe: 21 contacts evenly over 50–829 µm (38.95 µm pitch); Gaussian volume
conduction σ = 0.15 × 779 µm = 117 µm (the E2 value 0.15 in cortical-thickness units; assumed). Analysis as
`scripts/e2_motif_figure.py`: 1 kHz, non-overlapping 1 s windows, DPSS multitaper half-bandwidth 2 Hz, RP(c, f) =
P/max_c P, bands 10–30 and 50–150 Hz, panels d–i layout. Depth axis rel. mid-L4 in cortical-thickness units.

## 8. Experiments (P, in our form)

| Id | Condition | Realization |
|---|---|---|
| X0 | baseline, spontaneous | H and A0 |
| X1 | silence PV / SST / VIP (all layers) | `param_overrides` EL → −90 mV on the triple (stands in for P's hyperpolarizing current); check: 0 spikes in the silenced population |
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
