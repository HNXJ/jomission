# Spec (DRAFT, for review): OPEN-38 spectrum sensitivity to observed SST/PVALB composition (lineage `o38s`)

Status: **draft; independent review ACCEPT (round 2, e089ec87; minors applied); no run authorized.** Owner decision 2026-11-16 (Option 3, staged): study approved in principle; no
smoke test until this spec passes review; no full runs until the smoke test is read. Not a seal until the owner
approves its values. Composition source: `data/open38/COMPOSITION_F001_2026-11-16.md` (monkey #1 only).

## 1. Two hypotheses, tested apart

| | Claim | Data that answer it | This spec |
|---|---|---|---|
| H-anat | The prescribed ramp (S 0.25, M 1, D 2) describes macaque laminar SST/PVALB | composition data (OPEN-38 prereg; animal-level test deferred) | no; see §8 for whether the comparison is valid at all |
| H-mech | In the LANSA model, the laminar spectral motif depends on the SST/PV composition profile | simulation | yes |

A failure of H-anat does not refute the spectral mechanism, and a pass of H-mech says nothing about cortex. The
only joint statement allowed: whether the model's motif survives when its composition is set to the profile
observed in monkey #1 (provisional input, one animal).

## 2. Frozen model (everything except composition)

Reference = the T8 reference column `e2_3unit_n4000_dens_k1_kd1_sst15` (`SEAL_T8.md`): 3 units S/M/D at depth
-0.4/0/0.3, mean 4000 E per unit, LANSA density sizes (dens), inter-unit k 1, local in-degree kd 1, SST->E decay
15 ms; E/I 8000:1500 per unit; PV+SOM per unit fixed. Connectivity, weights, kinetics, drive, dt 0.1 ms, event
delivery: unchanged. Each arm differs from ref in the per-unit N_SST/N_PV only (`rhouUX`, or T8's `rhof`/`rhor`).

Consequence stated, not hidden: changing N_SST/N_PV changes each cell's SST and PV in-degree (fixed p per rule);
the arms test the construction rule, as in T8.

## 3. Arms (layouts frozen; tag `o38s_{arm}_n4000_s{seed}_T8`)

Ratios from `composition_f001.csv` (sha256 33545e31…) by `_temp_arm_ratios.py` (scratch; to be committed as
`scripts/open38_arm_ratios.py` at seal), 3 decimals; geometric means (gm) from unrounded ratios. chc and shape are
computed from the CSV rows (SST, PVALB, PV_CHC columns), not from the rounded values in the composition record.

| Arm | Profile S / M / D | gm | Layout suffix after `e2_3unit_n4000_dens_k1_kd1_sst15` |
|---|---|---|---|
| ref | 0.25 / 1 / 2 (prescribed) | 0.794 | (none) |
| v1 | 1.688 / 0.558 / 1.584 | 1.143 | `_rhouS1p688_rhouM0p558_rhouD1p584` |
| mt | 1.280 / 0.599 / 1.014 | 0.919 | `_rhouS1p280_rhouM0p599_rhouD1p014` |
| mst | 1.220 / 0.615 / 1.103 | 0.939 | `_rhouS1p220_rhouM0p615_rhouD1p103` |
| teo | 1.409 / 0.695 / 1.153 | 1.042 | `_rhouS1p409_rhouM0p695_rhouD1p153` |
| chc | 1.345 / 0.595 / 1.178 (4-area gm, PV_CHC in PV pool) | 0.981 | `_rhouS1p345_rhouM0p595_rhouD1p178` |
| flat | 0.794 every unit (T8 rflat) | 0.794 | `_rhof0p794` |
| rev | 2 / 1 / 0.25 (T8 rrev) | 0.794 | `_rhor` |
| shape | 1.095 / 0.485 / 0.942 (4-area gm PVALB profile, S 1.388 / M 0.615 / D 1.195, rescaled to gm 0.794) | 0.794 | `_rhouS1p095_rhouM0p485_rhouD0p942` |
| nc | 0.25 / 1 / 2 through `rhou` (parser regression check, §5) | 0.794 | `_rhouS0p25_rhouM1_rhouD2` |

shape is mandatory: the area arms differ from ref in both profile shape (U, M lowest) and level (gm 0.92-1.14 vs
0.794). ref − shape is the shape effect at fixed level; shape − (4-area arms) is the level effect, descriptively.

## 4. Estimands, readout and normalization (existing chain, unchanged)

Chain per run (`run_e2_scored.sh` order): `run_e2_stack.py` -> `e2_motif_figure.py TAG 2` -> `e2_motif_similarity.py`
-> `e2_scorer.py`. Map: relative power RP(c, f) = P(c, f) / max_c P(c, f), 21-contact virtual probe, RWS proxy
summed over E cells, readout sigma 0.15, 1-150 Hz on a 1 Hz grid, 2 s discarded of 8 s.

- **Primary:** r_ab = Pearson r over the 21 contacts of the band-mean RP depth profile, 10-30 Hz (verdict band),
  against the 11-area-mean observed LANSA map. It measures deep alpha-beta placement.
- **Secondary:** r_gm (50-150 Hz, superficial gamma placement); mid-depth relative gamma (contacts -0.2 to 0,
  `e2_mid_gamma.py`); r_proto; crossing_2se; scorer pass; unit E rates S/M/D.
- **Per-arm estimand:** d(arm) = mean over seeds of X(ref, s) - X(arm, s), paired by seed.
- r_ab of the 4 area arms beside each arm's Δ = log2(R_D/R_S): illustrative only (n = 4, Δ spans -0.34 to -0.09),
  no estimand, no test.

## 5. Controls

- **Positive (prescribed-ratio recovery):**
  (a) Realized N_SOM/N_PV per unit equals the arm's rho within the rounding of `unit_sizes` (one cell), every arm.
  (b) ref seeds 1-3 reproduce the T8 ref scores (`t8_summary.json`: r_ab 0.839, 0.857, 0.827): array equality with
  the T8 runs if those arrays are obtained; otherwise the ref seed-1-3 mean r_ab lies within 3 × the SD across the
  three T8 ref seeds (seed SD, not SD of the mean; computed from `t8_summary.json` and written here at seal) of the
  T8 seed mean. n = 3 makes this a coarse check. Spiking dynamics diverge across hosts, so per-seed
  equality is not expected without the arrays.
  (c) flat and rev reproduce T8's "ref higher" on r_ab (T8 mean d 1.56, 1.58): the known composition effect.
  Failing (a), (b) or (c) stops the study before any data arm is read.
- **Negative controls.** No dynamic readout of a recurrent E-PV-SOM network is composition-invariant by
  construction (every inhibitory rate feeds back on E), so the invariance tested is the construction's:
  (a) Construction invariance, every arm vs ref, same seed: E->E edges (local and inter-unit) identical edge lists
  (inferred to share one RNG key per rule shape; verified here, and if not identical, pairing is reported as absent
  and §6 uses unpaired intervals); E cell count and parameters, per-cell drive of every population, readout geometry
  identical. For rules into or out of PV/SOM, realized edge count within the binomial 99.9% interval of
  p · N_pre · N_post for that arm.
  (b) Parser regression: nc must equal ref array by array, seed 1. It tests the `rhou` path only, not composition.
  (c) Noise floor: SD of r_ab over ref seeds 1-5; equivalence margin for "no composition effect" = ±0.1 r_ab
  **[owner]** (the 95% interval of d inside the margin). Scale: about 6 % of the T8 ref-to-flat effect (1.56) and
  about 6 × the T8 seed SD; 8 equivalence claims use unadjusted intervals (stated as such); Pearson r saturates
  near 1, so the margin is not uniform across the range of r_ab.
  **[owner]** optional dynamic control: a total-size arm keeping each unit's ratio at ref but changing PV+SOM
  (needs a new modifier; not built).

## 6. Seeds, replication, uncertainty, decision rule

- Seeds 1-5 per arm (T8 used 1-3; 5 lets the rule below have a nominal level). Same seed across arms; pairing
  holds only if §5a finds identical E->E graphs, else unpaired Welch intervals.
- Per arm and outcome: every d_s, mean, SD (ddof 1), 95% t-interval (t_{0.975,4} = 2.776).
- Minimum detectable d (80% power, n = 5) at the strictest Holm level, two-sided 0.05/8, from the SD of r_ab over
  ref seeds 1-5 (after the smoke and before any arm is read); the T8 rflat/rrev paired SD (n = 3) is quoted
  beside it as a crude prior estimate with its interval. A "no consistent difference" smaller than the MDD is
  reported as underpowered, not as no effect.
- Pairing benefit: whether the Poisson drive key is arm-independent is checked with §5a; the paired SD of d is
  reported beside the unpaired SD so the gain from pairing is measured, not assumed.
- Primary family: 8 comparisons (v1, mt, mst, teo, chc, shape, flat, rev vs ref) on r_ab; paired t-test, Holm at 0.05.
  "ref higher" / "arm higher" only if Holm-significant and every d_s has that sign; otherwise "no consistent
  difference", reported with its interval. Secondary outcomes: intervals only, no tests.
- Effect read against the noise floor (§5c) and against T8's flat/rev effect (scale of a known composition effect).
- Seed count is fixed here; no seeds added after results.

## 7. Stop states, smoke test, budget

- Stop states as SEAL_T1/T8: any unit E rate > 50 Hz or deep-unit E rate < 0.1 Hz: reported, not counted; an arm
  with a missing seed is "not evaluable" (no replacement).
- Study stops (nothing read) on: engine identity not verified (below); §5 positive (a)-(c) failing; construction
  invariance failing on E cell count, E parameters, per-cell drive or the PV/SOM binomial check (E->E edge-list
  non-identity does not stop; it switches §6 to unpaired Welch intervals); any unit with N_PV or N_SOM = 0 or a connection probability > 1 (`_prob` refusal).
- **Smoke test (after review only):** ref seed 1 and nc seed 1, 8 s. Pass: §5 positive (a), construction
  invariance of nc, nc array-equal to ref; ref seed-1 r_ab reported beside T8's 0.839 (no pass threshold on one
  seed). Measures wall time per run; the campaign (9 arms × 5 seeds = 45 runs + nc) waits for the owner after the
  smoke result.
- **Engine prerequisite:** T8 engine is JaxFNE 0.5.9 @ 351f6048, run on another host (`zsh`,
  `/usr/local/bin/python3`). On this host a 0.5.9 tree exists at `D:/tmp/jaxfne_v059` with no recorded commit
  (UNKNOWN identity); `D:/venvs/jomission_pin058` holds 0.5.8. Needed before the smoke: a verified 0.5.9 @ 351f6048
  install (owner of the JaxFNE checkout: main-vwin), and a Windows invocation of the same four scripts (no change
  to them). Until then the smoke cannot run here.
- Not retuned: arms, layouts, seeds, estimands, rule fixed at seal. No arm chosen after results.

## 8. Is "prescribed vs observed" one estimand? (owner priority 4)

| Axis | Model rho | Observed R (Chen 2023, monkey #1) | Same? |
|---|---|---|---|
| What it is | count ratio of two AdEx populations in a functional unit, chosen inside the E1 oscillating range by ruling (C1/C4, 2026-11-09); never estimated (`LANSA_SSTPV_PROVENANCE_AUDIT_2026-11-16.md`) | count ratio of transcriptomic subclasses, Spatial-ID label transfer | no: rho is a parameter, R a measurement |
| Cell classes | "PV" fast-spiking, "SOM" slow, SOM->PV one-way (CAMINOS) | PVALB subclass (32 clusters, 4 absent in space), SST subclass (32, 2 absent); PV_CHC separate | partial; physiology of transcriptomic types not established here |
| Species | CAMINOS parameters (source species not re-read for this spec: UNKNOWN); observed spectral target from LANSA monkeys | *M. fascicularis*, one male | UNKNOWN |
| Laminar modules | S/M/D = LANSA depth bins about mid-L4 (z < -0.175, |z| ≤ 0.175, z > 0.175) | S = l2+l3, M = l4, D = l5+l6, hand-drawn per section | not the same boundaries; V1 L4 convention (Brodmann vs Hassler, 4B placement) in Chen: UNKNOWN |
| Spatial sampling | one column per area-mean map | sum over all 119 sections of each area (all eccentricities) | different; LANSA sites unknown for 6 areas |
| Normalization | count ratio | count ratio within the same compartment, so density normalization cancels | same form |
| Extraction validity | — | portal area-level SST/PVALB equals Chen's density table within 0.2% in V1, MT, MST, TEO, V2, V4, 46d (`_temp_density_crosscheck.py`) | extraction verified |

Reading: the discrepancy (U-shaped observed vs monotone prescribed) is a valid statement about the parameter's
justification ("no anatomical support in monkey #1 for the imposed ramp"), not a measured biological contradiction
of the model, because rho never claimed to estimate R. Before a biological reading: layer-convention and module
boundary check (V1 L4), a second animal, and the class mapping.
