# Plan: LANSA-composed CAMINOS stack, first spectral run (E2), 2026-11-09

Status: PLAN ONLY. No run, seal or result. Owner brief 2026-11-09 (chat): laminar CAMINOS E/PV/SST network with LANSA cell-type composition; test whether superficial gamma / deep alpha-beta emerges without fitting target frequencies; build, simulate, inspect, then decide; no large sweep. Lineage SCI-LAMINAR-MOTIF-1, minimal model LM-0 (`MINIMAL_MODEL_SPEC.md`), experiment E2. E2 needs its own seal before any run (AGENTS.md).

## Brief steps against what exists

| Brief step | Existing work | State |
|---|---|---|
| 1. CAMINOS unit E/PV/SST, no VIP, original equations | E0 (`E0_SEAL.md`, `jomission/laminar/caminos_e0.py`): AdEx, N_E 8000, N_inh 1500, p 0.07, one-way SST to PV, JaxFNE 0.5.9 | PASS 2026-11-07, seeds 1-5, 12 s: f_pk 34.0 Hz, PV-to-SST lag 7.16 ms. Pipeline equality to the JaxFNE port, not to the paper |
| 2. Superficial/deep composition from LANSA | X imported (`data/lansa_density_X.json`: 11 areas, PV/CB/CR/NG, 21 bins -0.6..0.4 around mid-L4) | SST not measured; conflict C1 |
| 3. Frequency vs SST/PV | E1 (`E1_SEAL.md`), rho 0.25-4, seeds 1-3 | INCONCLUSIVE: see table below |
| 4. 20 s laminar run, rasters, PSD, power profile | E2 in LM-0 | not sealed, not built |
| 5. Interpret, refine smallest mechanism | E3-E6 in LM-0 are the falsifiers | not sealed |

E1 per-rho mean `f_pk` (OBSERVED, `results/laminar_e1/e1_result.json`, 3 seeds; Welch bins 1 Hz):

| rho | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| f_pk (Hz) | 46.0 | 39.7 | 34.0 | 27.0 | none (peak excess 3.04, 3.26, 4.78; seed 3 is above 4 yet `oscillates` is false; INFERRED cause: the `edge_ok` term of `jaxfne.analysis.oscillation_call`, which `E1_SEAL.md` does not state) |
| PV rate (Hz) | 8.0 | 8.9 | 11.3 | 17.8 | 30.0 |

Two consequences for E2 (INFERRED from isolated modules):
- The reachable range at E0 drive is 46 to 27 Hz: gamma to low beta. No alpha (8-12 Hz) point exists. A deep module with rho 2 predicts beta, not alpha; one with rho 4 predicts no rhythm.
- PV rate rises with rho although N_PV falls. "SST suppresses PV" does not appear as lower PV rates in E1; whether it acts on PV timing is open (E1c: symmetric wiring).

## Conflicts for the owner (STOP before seal)

| Id | Conflict | Sources | Options, recommendation first |
|---|---|---|---|
| C1 | Brief: use actual LANSA layer-resolved SST/PV. LANSA did not measure SST; owner ruling 2026-11-06: CB is not SST; macaque CB/PV is about 3.0 superficial, 0.45-0.6 deep (opposite to the hypothesis) | OPEN-38; brief 2026-11-09 | (a) standing ruling: one-parameter rho ramp, low superficial, high deep; LANSA used only for N_E/N_inh per module from NG and PV+CB+CR; (b) CB/PV as given, as a test of the proxy (predicts reversed motif); (c) wait for mouse numeric profile (Kim 2017 tables, not in hand) |
| C2 | Brief: superficial and deep (2 layers). LM-0 approved 3 modules S/M/D (minimum with a crossover) | `MINIMAL_MODEL_SPEC.md` decision 2 | (a) keep 3; (b) 2 for speed, crossover not testable |
| C3 | Brief: derive cells per unit and units per layer, no invented counts. E0/E1 fixed N_E 8000, N_inh 1500 (CAMINOS published module) | `E1_SEAL.md` | (a) one CAMINOS module per depth module, N 9500 each, only rho varies (one principal delta; E/I per depth deferred to E6); (b) scale N per module by LANSA integrated density (changes E/I and size together with rho; two deltas) |
| C4 | Deep rho: E1 has no oscillation at rho 4 | E1 result | (a) rho_D = 2 (rho_S 0.25, rho_M 1), all inside the E1 oscillating range; (b) rho_D = 4 with drive recalibrated (new principal delta) |
| C5 | Brief: replication audit reports a PV-to-SOM lag; asks to confirm one-way SST to PV | brief; E0 | not a conflict: a lag is timing, not an edge; E0 reproduces PV leading SST by 7.16 ms with one-way wiring. E1c tests the wiring directly |
| C7 | E1 verdict path: condition 1 holds (4 of 5 oscillate in each seed); condition 3 is undefined because rho 4 does not oscillate. `E1_SEAL.md` names INCONCLUSIVE only for a failed condition 1; `e1_verdict.py` (residual branch, about line 260) also returns INCONCLUSIVE for this case | `E1_SEAL.md`; `jomission/laminar/e1_verdict.py` | (a) record that the code's residual branch is the reading of the seal, additive note, verdict unchanged; (b) treat as a seal gap; E2's verdict rule must name every branch |
| C6 | Brief 20 s; E0/E1 use 12 s, 2 s discarded | seals | (a) 20 s, 2 s discarded; (b) 12 s for comparability with E1 |

## Owner rulings 2026-11-09 (chat)

- Speed over test coverage: build, simulate, inspect; tests only where a run depends on them.
- C1/C4: rho S 0.25, M 1, D 2 (ramp inside the E1 oscillating range). C2: 3 modules. C3: one CAMINOS unit per module (8000 E, 1500 inh).
- Coupling: adjacent modules (S-M, M-D), both directions, E to E, PV and SST, p 0.01, local weights (2.0 nS to E, 3.0 nS to inhibitory), 2 ms delay.
- C6: 20 s, 2 s discarded, seed 1, then inspect.
- Verdict bands: alpha-beta 10-30 Hz, gamma 50-150 Hz (published panel; manuscript 8-30/31-150 is scored as secondary only).
- Amplitude: unit power is not equalized; the depth gradient of LFP amplitude is a model result.
- Kinetics: slower SST->E next. Agent choices (owner gave no value): decay 30 ms (middle of the review's 20-40 ms), own receptor `gaba_sst`, weight 5 x 8/30 nS so the charge per spike equals CAMINOS; other synapses unchanged. Layout `e2_3unit_sst30`, 20 s, seed 1.
- Route: JaxFNE 0.5.9 AdEx through `construct_adex_model(neuron_rows=...)`; the public `column()` path is refused for AdEx. Layer selectors realize exact blocks (probe: 44 751 edges vs 44 800 expected per local E-E block).

## Steps (each closes on its check)

| # | Step | Closing check | Needs |
|---|---|---|---|
| 1 | Module profile table from X: per area and per module (bin groups S, M, D), NG, PV, CB, CR integrated density, CB/PV. Descriptive; no model | table plus script and tier-0 test; bin groups declared before reading values | owner nod (item 4 is not in the principal delta) |
| 2 | E2 seal draft: rho per module, N per module, inter-module coupling (FF/FB E to E and E to inhibitory, p and g, UNSET), drive, T, seeds, LFP proxy (summed synaptic current onto E, decided), verdict rule (gamma relative-power peak in S, alpha-beta peak in D, both against an E4 flat-profile null), V0-V3 paths | owner approves every value; committed and pushed before step 4 | C1-C6 rulings |
| 3 | Stack runner: parameterize `caminos_e0.build_spec` into 3 modules plus coupling block | gates: realized sizes and edge counts per block; zero coupling reproduces three E1 modules (byte-equal counts, 2 s probe, seed 1); V0 schematic; V1 1 s raster | step 2 |
| 4 | E2 run, sealed seeds | result JSON with `verdict`, SHA, config, JaxFNE 0.5.9; runtime INFERRED about 3 x 36 min x (T/12 s) per seed on CPU (3 modules of 9500 cells, linear scaling assumed) | step 3 |
| 5 | One figure: per-module rasters (E, PV, SST), PSD per module, relative power by depth (gamma, alpha-beta) | V2 raster; figure measured per `match-visual-style` | step 4 |
| 6 | Interpret: E2 vs E1 frequencies (coupling effect), vs in-vivo motif | in-vivo comparison needs Y (observed LFP, OPEN-38, not imported); without Y, report the simulated profile only | Y import |
| 7 | Next smallest delta, chosen after step 5: E4 (flat) and E3 (reversed) are the falsifiers, one run each | separate seal each | owner |

## First result (unsealed): e2_stack_s1_T20, seed 1, 20 s, 2 s discarded

Readout `scripts/e2_motif_figure.py` at 5c74999: RWS proxy per E cell, 21-contact probe, sigma 0.15, bands 10-30/50-150 Hz.

| Unit | Peak (Hz) | ab/gamma power | gamma power vs S | ab power vs S |
|---|---|---|---|---|
| S (rho 0.25) | 44 | 0.5 | 1x | 1x |
| M (rho 1) | 15 | 32 | 2.4x | 145x |
| D (rho 2) | 15 | 26 | 21x | 1000x |

Contrast M = +0.064 +- 0.005 (jackknife z 11.9; 3-unit null SD of z is 1.6 on synthetic data), depth-shuffle p 1/3 (6 orders), no crossing: both bands peak at D. The frequency ordering follows rho; the amplitude ordering (D loudest in every band) blocks the relative-power motif. Status: no motif, single seed, unsealed.

## Owner review 2026-11-09 (chat) and sweeps

- D 21x louder in gamma than S means the units are too isolated; superficial and deep must interact.
- Free parameters are swept logarithmically, not asked.
- Gamma source: the L2/3 E-PV loop (PING; Kopell and others), not L4 drive alone (many areas have a weak L4);
  L4 E-PV gives L4 gamma. Against Herrera et al. 2026 (gamma from L4 PV->E; alpha-beta intrinsic to every layer).
- Deep layers: larger, sparser neurons; dense small superficial cells synchronize at gamma more easily.

Sweep (`scripts/run_e2_sweep.sh`, seed 1, 8 s, 2 s discarded): 3-unit stack, E per unit from LANSA total
density (S 5785, M 3668, D 2547; mean 4000), K_INTER-based coupling with end-unit compensation.
k (inter in-degree x 80) in {0.1, 0.3, 1, 3, 10} at dsz 1; deep E-cell size dsz (C, gL, a, b x dsz; synapses
and drive unchanged) in {0.5, 2, 4} at k 1.

Coupling result (per E cell, bands 10-30/50-150; peaks S/M/D in Hz; D/S = deep over superficial unit power):

| k | peaks | gamma D/S | ab D/S | M | z | crossing | rates E S/M/D (Hz) |
|---|---|---|---|---|---|---|---|
| 0.1 | 52/15/22 | 19 | 1650 | +0.051 | +3.7 | no | 0.83/1.11/1.63 |
| 0.3 | 46/22/13 | 22 | 806 | +0.053 | +5.2 | no | - |
| 1 | 37/17/16 | 17 | 236 | +0.096 | +7.4 | no | 0.76/1.02/1.57 |
| 3 | 8/7/18 | 2.1 | 2 | -0.055 | -0.8 | no | 1.41/1.85/2.64 |
| 10 | 20/21/20 | 1.2 | 1 | -0.037 | -3.7 | no | 196 everywhere (saturated, discard) |

Coupling up to k 1 leaves D dominant; k 3 equalizes amplitude but collapses every unit to slow rhythms; k 10
saturates. No coupling value gives the motif. Diagnosis (20 s run): S is coherent with M (E counts 0.43 at
30-50 Hz) but weakly synchronized (E var/mean 3.9 vs D 32; SST volleys drive D, var/mean 169). Next: local
in-degree scaled by density (kd 0.5, 1) and deep cell size (dsz), queued.

Deep E-cell size result (k 1; dsz scales C, gL, a, b of D's E cells; synapses and drive unchanged):

| dsz | peaks | gamma D/S cell | M cell (z) | M sum (z) | crossing cell/sum | rates E S/M/D (Hz) |
|---|---|---|---|---|---|---|
| 0.5 | 46/34/34 | 1.6 | +0.456 (+5.5) | +0.485 (+4.9) | no/yes | 5.4/5.6/18.0 |
| 2 | 36/18/15 | 0.4 | +0.254 (+11.6) | +0.694 (+25.7) | no/no | 0.76/1.06/0.06 |
| 4 | 39/19/34 | 0.3 | +0.259 (+5.9) | +0.679 (+13.2) | no/no | 0.76/1.08/0.00 |

Larger deep cells at unchanged synapses and drive silence D's E cells (0.06 Hz at dsz 2, 0 at 4); D's power
falls below S's because D stops firing, not because S gains gamma. dsz 0.5 raises every rate (D_E 18 Hz) and
gives the only crossing (sum row), with no deep alpha-beta peak. None is the motif mechanism; a size model
that also scales synapse count and drive with membrane area is the faithful version (inferred).

Density-scaled local in-degree result (k 1; in-degree K_E x (density/mean)^kd, every local rule):

| kd | peaks | gamma D/S cell / sum | ab D/S cell / sum | M cell (z) | M sum (z) | rates E S/M/D (Hz) | coh S-M 30-50 |
|---|---|---|---|---|---|---|---|
| 0 | 37/17/16 | 16.9 / 3.3 | 236 / 46 | +0.096 (+7.4) | +0.262 (+19.6) | 0.76/1.02/1.57 | - |
| 0.5 | 39/24/16 | 6.9 / 1.3 | 52 / 10 | +0.155 (+11.3) | +0.460 (+8.5) | 0.59/1.09/2.00 | 0.58 |
| 1 | 45/17/16 | 3.5 / 0.7 | 22 / 4 | +0.231 (+11.3) | +0.682 (+11.8) | 0.48/1.12/2.53 | 0.57 |

kd 1, sum over cells: gamma relative power peaks superficial (0.84 at -0.4), alpha-beta peaks mid-deep (0.93
at 0.05); the profiles cross at -0.19 and alpha-beta exceeds gamma by about 0.37 (SE 0.03) at mid depths.
By the predeclared rule the crossing is False: at the bottom contact (0.4) the two are equal (0.49 vs 0.50)
because D also carries gamma. First motif-shaped profile; one seed, 8 s, unsealed. Queued: kd 2 (seed 1),
kd 1 (seed 2).

Slow SST->E result (e2_sst30_s1_T20: 3 x 8000 E, p_inter 0.01, SST->E 30 ms at equal charge, 20 s): every
unit's spikes lock to 7-8 Hz (E var/mean S 7.5, M 31, D 55; SST 20/118/138); LFP peaks S 42, M 7, D 7 Hz;
gamma D/S 3.2 (was 21), alpha-beta D/S 9 (was 1037); M +0.040, z +0.5, no crossing. Slow SST->E reaches
alpha but spreads it to the superficial unit too. Queued: kd 1 with SST->E 15 and 30 ms.

## Similarity to the observed LANSA map (owner 2026-11-09: fine-tune toward 95%+)

Metric: r_proto = mean over 1..150 Hz of Pearson r over the 21 bins against the 11-area mean observed map
(`scripts/e2_motif_similarity.py`, data `lansa_lfp_Y.json`). Target is well defined: even- vs odd-area means
agree at r 0.978. Ceiling at the default readout (3 units at -0.4/0/0.3, sigma 0.15) when each unit's
cross-spectrum is free per frequency: 0.854 (sigma 0.25: 0.994; 5 units at sigma 0.15: 0.997); with
incoherent units 0.693 (sigma 0.25: 0.925). So 0.95 needs inter-unit coherence or a wider readout, not more
units. The observed map asks, per unit relative to S: alpha-beta D > M > S, gamma M about 0.7-0.8 of S and D
about 0.

| run (k 1, sum over cells) | r_proto | null95 | r_ab | r_gm | ab M/S, D/S | gamma M/S, D/S |
|---|---|---|---|---|---|---|
| kd 1, seed 1 | -0.054 | 0.21 | 0.90 | -0.35 | 4.4, 4.2 | 0.51, 0.67 |
| kd 1, seed 2 | -0.183 | 0.21 | 0.89 | -0.52 | - | - |
| kd 2 | 0.163 | 0.24 | 0.17 | 0.28 | 1.2, 0.27 | 0.39, 0.24 |
| kd 1, SST->E 15 ms | **0.376** | 0.26 | 0.83 | 0.38 | 6.1, 4.2 | 0.26, 0.15 |
| kd 1, SST->E 30 ms | 0.273 | 0.26 | -0.35 | 0.51 | 0.64, 0.15 | 0.28, 0.087 |

kd 1 with SST->E 15 ms is the first run above null95. Remaining gaps: deep gamma not yet 0, deep alpha-beta
below middle. Next: deep-only SST->E decay (`sstdNN`) 30/60/120 ms on kd 1 + SST->E 15 ms.

## Out of scope

VIP, H-state and HDP, Izhikevich-TFNE, parameter sweeps, omission (firewall), any project-wide JaxFNE move (OPEN-37).
