# Spec (DRAFT, for review): OPEN-38 spectrum sensitivity to observed SST/PVALB composition (lineage `o38s`)

Status: **draft; no run authorized.** Owner decision 2026-11-16 (Option 3, staged): study approved in principle; no
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

Ratios from `composition_f001.csv` (sha256 33545e31…) by `_temp_arm_ratios.py`, 3 decimals; geometric means (gm).

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
| nc | 0.25 / 1 / 2 through `rhou` (negative control, §5) | 0.794 | `_rhouS0p25_rhouM1_rhouD2` |

Not separable with these arms: data arms differ from ref in both shape (U, M lowest) and overall level (gm 0.92-1.14
vs 0.794). Proposed optional arm for review, `shape`: the 4-area gm PVALB profile (S 1.388, M 0.615, D 1.195,
gm 1.007) rescaled to gm 0.794, values computed by script at seal time; it isolates shape from level. Owner to
accept or drop.

## 4. Estimands, readout and normalization (existing chain, unchanged)

Chain per run (`run_e2_scored.sh` order): `run_e2_stack.py` -> `e2_motif_figure.py TAG 2` -> `e2_motif_similarity.py`
-> `e2_scorer.py`. Map: relative power RP(c, f) = P(c, f) / max_c P(c, f), 21-contact virtual probe, RWS proxy
summed over E cells, readout sigma 0.15, 1-150 Hz on a 1 Hz grid, 2 s discarded of 8 s.

- **Primary:** r_ab = Pearson r over the 21 contacts of the band-mean RP depth profile, 10-30 Hz (verdict band),
  against the 11-area-mean observed LANSA map. It measures deep alpha-beta placement.
- **Secondary:** r_gm (50-150 Hz, superficial gamma placement); mid-depth relative gamma (contacts -0.2 to 0,
  `e2_mid_gamma.py`); r_proto; crossing_2se; scorer pass; unit E rates S/M/D.
- **Per-arm estimand:** d(arm) = mean over seeds of X(ref, s) - X(arm, s), paired by seed.
- **Gradient estimand (descriptive):** r_ab across the 4 area arms against each arm's Δ = log2(R_D/R_S); n = 4,
  no test.

## 5. Controls

- **Positive (prescribed-ratio recovery):** ref seeds 1-3 must reproduce the T8 ref scores in
  `paper/jomission_lansa/data/t8_summary.json` (r_ab 0.839, 0.857, 0.827): |Δr_ab| ≤ 0.01 per seed, or array
  equality with the T8 runs if those arrays are obtained. flat and rev must reproduce T8's "ref higher" on r_ab
  (T8 mean d 1.56 and 1.58). Failing either means the engine or chain differs from T8: stop, no arm is read.
- **Negative (composition-invariant):** (a) `nc` encodes ref's composition through the new code path; its run must
  equal ref array by array (seed 1; any differing array stops the study). (b) Across all arms, realized E->E,
  E->PV and E->SOM edge counts per unit and the drive table must equal ref's (`edge_table`); only PV/SOM counts and
  edges leaving PV/SOM may differ. (c) Seed-only variability: ref seeds 1-5 SD of r_ab is the noise floor each d
  is read against.

## 6. Seeds, replication, uncertainty, decision rule

- Seeds 1-5 per arm (T8 used 1-3; 5 lets the rule below have a nominal level). Same seed across arms; whether a
  seed yields the same E->E graph across arms is checked in §5b and reported (pairing is assumed otherwise).
- Per arm and outcome: every d_s, mean, SD (ddof 1), 95% t-interval (t_{0.975,4} = 2.776).
- Primary family: 7 comparisons (v1, mt, mst, teo, chc, flat, rev vs ref) on r_ab; paired t-test, Holm at 0.05.
  "ref higher" / "arm higher" only if Holm-significant and every d_s has that sign; otherwise "no consistent
  difference", reported with its interval. Secondary outcomes: intervals only, no tests.
- Effect read against the noise floor (§5c) and against T8's flat/rev effect (scale of a known composition effect).
- Seed count is fixed here; no seeds added after results.

## 7. Stop states, smoke test, budget

- Stop states as SEAL_T1/T8: any unit E rate > 50 Hz or deep-unit E rate < 0.1 Hz: reported, not counted; an arm
  with a missing seed is "not evaluable" (no replacement).
- **Smoke test (after review only):** ref seed 1 and nc seed 1, 8 s. Pass: positive-control check on ref seed 1
  and nc array-equal to ref. Measures wall time per run; the campaign (8 arms × 5 seeds = 40 runs + nc) waits for
  the owner after the smoke result.
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
