# Plan: jomission-LANSA paper draft, v2 (2026-11-10)

v1 was reviewed by Fable 5.2 (high, independent) with verdict REVISE: 4 blockers, 7 major, minor. v2 applies
every finding; the decision on each is in §6. Owner order (chat 2026-11-10): a paper titled roughly
"Laminar-cellular architecture sets the spectrolaminar motif in a primate model of the cortical hierarchy",
with the figure order and style of Herrera et al. 2026 (bioRxiv 10.64898/2026.11.04.763717, "SLM") and the
data, conventions and statistics of the LANSA draft (Mulvey ... Bastos). Panels show LFP, rasters and CSD
from JaxFNE models. No cap on supplements. Every model gets an architecture schematic, a raster, LFP, CSD and
a details table, at least in the supplement. Tests: a ~1000-neuron motif model, E/I, the PV/SST ratio, PV/SST
stimulation and suppression, VIP, and perpendicular and crossing probes. Owner standing order (2026-11-09):
"nothing needs my ruling ... just simulate, sweep, logarithmic sweep". Questions route to the parent, not the
owner (todo.json SCI-LAMINAR-MOTIF-1, 734fede). Status: plan. Nothing below is a result unless it cites a file.

## 0. Ground truth in the repo

- **Observed data:** Y (`lansa_lfp_Y.json`, 21 x 150 x 11) and X (`lansa_density_X.json`).
- **Model:** `caminos_stack.py`, with modifiers k, kd, sst, sstd, rhod, dsz, n and dens.
- **Readout:** RWS proxy and 21-contact probe (`e2_motif_figure.py`); similarity in `e2_motif_similarity.py`.
- **Best 3-unit run:** kd1_sst15, r_proto 0.376. Jackknife SE 0.059; the seed gap on kd1 is 0.13 (Fable probe).
- **Owner start point:** kd1_sst15_sstd60 at 0.299 is unstable (halves -0.16 / +0.30, SE 0.41).
- **Parameter sources:** every CAMINOS value is from Tahvili, Vinck and di Volo (JaxFNE PORT_SPEC Tables 1-2;
  `caminos_e0.py:5`). Kopell is cited for the PING concept only. RWS is Mazzoni 2015.
- **JaxFNE:** the authority is 0.5.8. The E2 runs used a 0.5.9 side install, so they are off-authority and
  exploratory (OPEN-37, owner queue).

## 1. Figures (SLM order; LANSA conventions)

Conventions:
- Depth: position from the L4 centre, superficial (-0.6) at the top, 21 bins.
- Relative power 0.5-1, parula-like colour map.
- Colours: PV blue, CB orange, CR yellow, NG/E purple.
- Bands: alpha-beta blue, gamma red. The primary bands are 10-30 / 50-150 Hz (owner ruling,
  PLAN_LANSA_E2.md). The LANSA 8-30 / 31-150 bands are reported beside them.
- Band dashes at 4/8/12/30/90 Hz. ±2 SEM over seeds, with seeds averaged as probes are in LANSA.
- Model power is stationary; LANSA power is from 1 s peri-stimulus windows. Both are stated in every legend.

| Fig | SLM | Content |
|---|---|---|
| 1 | Fig 1 | Reference model. A: architecture with the details table. B: raster. C: per-unit, per-cell-type firing-rate spectra. D: model SL map, RWS LFP, plus a firing-rate SL map. E: observed LANSA prototype. F: band depth profiles, LFP traces, current-based CSD. |
| 2 | Fig 2 | Robustness: drive-rate sweep (S4 analogue) and seeds, as difference maps with SSIM and Pearson r. The LANSA hierarchy (T7) panel is deferred (§5). |
| 3 | Fig 3 | Interneuron manipulations: PV and SST silencing (whole population, per unit), one stimulation level each, PV/SST ratio scale, VIP (stage 2). |
| 4 | Fig 4 | Laminar source: unit silencing, isolated units, graded PV silencing inside the isolated gamma unit. |
| 5 | Fig 5 | Determinants: S↔D swaps of rho, in-degree, N and SST→E τ on isolated units (SLM-style swaps), plus E/I g in {0.5, 2}. |
| 6 | new | Probe geometry, readout only (T6), scored by contrast M and by r_proto minus the identical-unit control. |

Supplement:
- One block per model and condition: architecture, raster, LFP, current-based CSD, SL map, details table.
- Rate boxplots (S3/S5/S7 analogues).
- S1 analogue: undriven vs driven, plus the raw laminar power map.
- The finite-size ladder.
- Controls: σ, swapped units, identical units, the smooth-spectrum null.
- Seeds and wall times.

## 2. Tests: lineage, estimand and acceptance

Each test is its own lineage with a seal pushed before its first run. The seal fixes the estimand, the grid,
the stop states and the V0-V3 paths `results/viz/<lineage_id>/` (visualization contract: V0 and V1 before any
tuning). E2 stays declared exploration. No retune after results.

**Shared scorer** (W2). A result counts only if all of these hold:
1. `crossing_2se` is true.
2. r_ab ≥ 0.6 and r_gm ≥ 0.6.
3. The identity unit order ranks 1 of 6 permutations.
4. r_proto is above the p95 of the max-over-grid smooth-spectrum null and above the identical-unit control.
5. The two halves agree within 2 SE.

Selection uses seeds 1-3; confirmation runs once, on seeds 4-6 at 12 s, with the config frozen. The even/odd
area split is not a hold-out.

**L-FS: finite-size ladder** (prerequisite for T1).
- kd1_sst15 at n4000 (2 new seeds), n2000 and n1000 E per unit (3 seeds each). n1000 is 3562 cells.
- Accept the smallest n whose r_proto is within 2 seed SD of n4000 and whose peaks are within 3 Hz.
- "~1000 neurons" (owner): 1000 total cells forces every local p to 1, so the kd axis vanishes and D has 11 PV
  cells. A run at 1000 total cells therefore uses a declared rule, weights × K_ref/K_real (charge per cell
  kept), as its own lineage L-FS1k. It is reported beside the in-degree-preserving ladder. Both answer the
  owner's question; neither is retuned.

**T1: motif fine-tune** at the ladder's n.
- One axis at a time on 3 axes (kd, SST→E τ, deep SST→E τ), 7 log-spaced levels × 3 seeds at 8 s.
- The winner is confirmed on seeds 4-6 at 12 s against the shared scorer.
- 0.95 is reported against the readout ceiling (0.854 coherent, 0.693 incoherent at σ 0.15). Never claimed
  without the check.

**T2: E/I.** On the frozen T1 reference:
- inhibitory weight g ∈ {0.5, 2};
- N_I/N_E deferred.
- Estimands: r_proto, r_ab, r_gm, per-unit peak Hz, rates and absolute band power.

**T3: PV/SST ratio.** Global rho scale ∈ {0.5, 2}, with the owner ruling's ramp shape (low superficial, high
deep) kept. LANSA CB/PV is not used as SST/PV (OPEN-38: CB is not SST). Estimands as T2.

**T4: PV/SST silencing and stimulation.**
- Silencing is a hyperpolarizing current, as in SLM: per population, whole and per unit.
- Stimulation: +50 pA.
- Route: `triple_drive_pA`, the only `I_ext` path into the AdEx kernel in 0.5.8 and 0.5.9.
- Acceptance: silenced spike count 0, edge counts unchanged.
- Graded fractions of 10/25/50/75/100% run only inside the isolated gamma unit, at n ≥ 1000 E.

**T5: VIP** (stage 2, main-model design).
- AdEx accepts only E, PV and SOM labels. VIP needs its own layer label with `param_overrides`, and every
  parameter cited (OPEN-36 is UNSET).
- Estimands include absolute power and rates, since SLM assigns a gain effect to VIP.
- Not tonight.

**T6: probe geometry** (post hoc readout on saved runs; no simulation).
- Units get a declared lateral extent.
- Two crossing definitions, both run (sweep, not ask):
  - oblique through one sheet, θ ∈ {0, 30, 60, 90}°;
  - lateral offset ∈ {0, 0.25, 0.5}.
- Estimands: contrast M, and r_proto minus the identical-unit control at the same geometry.
- Contacts map to bins by their projected depth.

**CSD** (every figure).
- Built from the model's currents, not from the kernel. The saved `g_columns` (unit × receptor) become current
  sources at declared depths, each with an equal return current. Placements (ASSUMED, cited): PV
  perisomatic, SST distal apical, AMPA basal and apical.
- Forward-model φ on the probe, then the second-difference CSD against the true sources.
- Trigger: average triggered on the alpha-band trough (stationary runs have no evoked response).

## 3. Readouts each run writes

- Raster counts.
- RWS LFP per unit.
- Current sources per unit and receptor.
- Probe φ and CSD.
- Spectra, the SL map (RWS and firing rate) and motif scores (M, z, crossing).
- Similarity (r_proto, r_area, r_ab, r_gm, null95, smooth null p95, identical-unit control, permutation rank,
  halves).
- E rate per unit.
- Details table: sizes, edges per rule, every parameter with its PORT_SPEC row or "agent choice". Agent
  choices: kd, sst15, K_INTER 80, σ 0.15, V_ref -65 mV, unit depths, MODULE_BINS.
- Run identity: SHA, dirty flag, JaxFNE version, file and source SHA, seed.

## 4. Execution

The parent keeps the seals, the science calls and run launches. Workers are Haiku 5.5 (`worker`, high), one
packet each, at most 3 at once. Each packet has a red-then-green test, a writer in its own worktree, and an
independent review before the parent merges.

| # | Packet | Acceptance |
|---|---|---|
| W0 | Stable 0.5.9 install of tag v0.5.9 (e0160dfc) outside the repo and outside scratch; 0.5 s smoke on 0.5.8 and 0.5.9; timing solo and 3 in parallel | Block spike counts compared; seconds per simulated second recorded; source SHA recorded |
| W1 | `caminos_stack`: raise when a p clips at 1; run JSON gets dirty flag and JaxFNE source SHA | n270 raises, n1000 passes; the test fails on the old code |
| W2 | Scorer: smooth-spectrum max-over-grid null, unit-permutation rank, identical-unit control, seed split, halves | Synthetic positive passes; identical units fail; random rank-3 spectra pass ≤ 5% |
| W3 | Details table, V0 schematic, raster/LFP/spectra figure kit in LANSA style, firing-rate SL map; applied to kd1_sst15 | Table edges equal `edge_table`; sizes sum to N; figure measured (match-visual-style) |
| W4 | Current-based CSD with forward φ, post hoc on saved npz | CSD sums to 0 over depth; planted dipole recovered at r ≥ 0.9; rank > 3 |
| W5 | Tilted and offset probe readout, post hoc | Identical units give M = 0 and Δr = 0 at every θ; θ 0 reproduces the current map to 1e-12 |
| W6 | Drivers: ladder, T1 axis search, T4 hooks, S↔D swaps, g and rho scale; 0.5 s smoke each | Smoke JSON written; silenced count 0; edges unchanged |
| W7 | LaTeX skeleton in SLM order with stubs (flow-article) | Compiles; every panel cites a result file or "pending" |

Order:
1. W0 and W1.
2. W2, W3, W4 and W5 (no simulation; at most 3 at once).
3. W6.
4. W7 at any time.
5. Seals R0, then the runs.

Compute (Fable, from recorded wall times): 3562 cells cost about 24 min per 8 s. About 30 slot-hours a night
is about 70-90 runs. The Allen replica stream uses 1 slot, so this work gets 2 slots.

Runs in priority order:
1. Ladder: 11 runs.
2. T1: 63 runs plus 3 confirmation runs. Over budget for one night; axes run in order and stop at the first
   axis that passes.
3. Perturbations: about 20 runs, 1 seed each; whole-population PV and SST silencing get 3 seeds.

## 5. Deferred, with reason

| Item | Reason |
|---|---|
| T5 VIP | No VIP AdEx label or sourced parameters (OPEN-36); main-model design first |
| T7 hierarchy, LANSA X to area models | OPEN-38: CB is not SST, and X is relative per type (E:I cannot be read from it). If built later, it must test diagonal vs off-diagonal, since LANSA S7A already gives r ≥ 0.61 across areas. |
| 21-bin stack | About 40k cells; one reference run at most, after the event-driven JaxFNE merge |
| N_I/N_E, graded fractions outside the isolated unit, Fig 7 schizophrenia | Budget |

## 6. v1 → v2 decisions (Fable findings)

| Finding | Decision |
|---|---|
| B1 n270 clips p, so kd is a no-op | Ladder first; 1000 total cells runs as its own declared weight-scaling lineage; W1 guard |
| B2 acceptance cannot fail; null anti-conservative | Shared scorer with seed split, smooth null and halves (W2) |
| B3 T6 measures geometry | M and Δr vs the identical-unit control; lateral extent; both crossing definitions |
| B4 kernel CSD has rank 3 | Current-based CSD (W4) |
| M1 CB→SST conflicts with OPEN-38 | Ramp kept; T7 and the CB ramp deferred |
| M2 authorization and seals | Order recorded (734fede); one seal per lineage with V0-V3 paths before any run |
| M3 0.5.9 install in scratch | W0 stable install; 0.5.8 smoke tests the authority route |
| M4 VIP not buildable | Stage 2, main model |
| M5 current hook | `triple_drive_pA`; values stated |
| M6 compute | Budgeted run list above |
| M7 sources | PORT_SPEC rows; agent choices marked |
| Minor | Swaps, isolated graded silencing, SSIM, S1/S4 analogues, firing-rate SL map, bands named, colours fixed, run JSON identity |
