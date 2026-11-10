# Seal: T8 density and SOM/PV-ratio controls (lineage `t8`), 2026-11-10

Under PLAN_PAPER_JOMISSION_LANSA.md (owner order in chat 2026-11-10: close the open items of the jomission-LANSA
manuscript; todo SCI-LAMINAR-MOTIF-1). Answers referee review 3, M4: the density mechanism and the SOM/PV gradient
were never tested as causes. Committed and pushed before the first T8 run.

- **Question:** at fixed total cell count and connectivity rule, does the depth profile of cell density, or the
  depth ramp of the SOM/PV ratio, change the reference column's laminar map? Estimand: per outcome, the paired
  (same seed) difference reference minus control.
- **Engine:** JaxFNE 0.5.9 at commit 351f6048 (the T1 axis-3 and confirmation engine), event delivery
  (`JAXFNE_EDGE_DELIVERY=event`), dt 0.1 ms. Runner `scripts/run_e2_scored.sh` (run, motif figure, similarity,
  scorer), unchanged. Seeds 1-3, 8 s, 2 s discard, N 4000 E cells per unit on average.
- **Arms (layout; tag `t8_{arm}_n4000_s{seed}_T8`):**

  | Arm | Layout | What changes against ref |
  |---|---|---|
  | ref | `e2_3unit_n4000_dens_k1_kd1_sst15` | none (the paper's reference column, rerun on the T8 engine) |
  | dflat | `e2_3unit_n4000_k1_sst15` | unit sizes equal (4000 E each) and one in-degree rule (no kd): density flattened, total E kept up to rounding |
  | drev | `e2_3unit_n4000_densr_k1_kd1_sst15` | S and D densities exchanged, for size and in-degree: density permuted |
  | rflat | `e2_3unit_n4000_dens_k1_kd1_sst15_rhof0p794` | SOM/PV 0.794 in every unit (geometric mean of 0.25, 1, 2); PV+SOM per unit kept |
  | rrev | `e2_3unit_n4000_dens_k1_kd1_sst15_rhor` | SOM/PV ramp reversed: S 2, M 1, D 0.25; PV+SOM per unit kept |

  Modifiers `densr`, `rhofX`, `rhor`: `jomission/laminar/caminos_stack.py`, tests
  `tests/test_caminos_stack_controls.py` (red on the builder before them).
- **Outcomes per run (scorer verdict file):** r_ab, r_gm, r_proto, crossing_2se, verdict; mid-depth relative gamma
  of the map (50-150 Hz, contacts -0.2 to 0, `scripts/e2_mid_gamma.py` convention); unit E rates.
- **Decision rule, per control arm c and outcome X:** d_s = X(ref, s) - X(c, s) for seeds s = 1, 2, 3. Report
  every d_s, mean and SD. "ref higher" if all three d_s > 0 and mean d > 2 SD / sqrt(3); "control higher" if
  all three d_s < 0 and mean d < -2 SD / sqrt(3); otherwise "no consistent difference". With n = 3 this is a
  descriptive screen with no nominal error rate; it is reported as such.
- **Primary outcome:** r_ab (deep alpha-beta placement), for both the density arms (dflat, drev) and the ratio
  arms (rflat, rrev). Secondary: r_gm, mid-depth gamma, r_proto, crossing and verdict.
- **Engine check:** ref seed s is compared array by array with the saved dense reference runs
  (`sw_n4000_dens_k1_kd1_sst15_s1_T8`, `lfs_n4000_s2_T8`, `lfs_n4000_s3_T8`); equality or the number of
  differing arrays is reported. The decision rule uses the T8 ref runs in every case, so all arms share one engine.
- **Stop states:** as SEAL_T1: any unit E rate above 50 Hz or deep-unit E rate below 0.1 Hz; the run is reported
  and does not count. A W1 refusal drops the run and is reported.
- **Not retuned:** arms, layouts, seeds, outcomes and decision rule are fixed here. No arm is chosen after results.
- **Visualization (contract), `results/viz/t8/`:** V0 architecture and details per arm (seed 1,
  `scripts/e2_paper_block.py`); V1 block with 1 s raster of each arm, seed 1, before any T8 score is read; V2
  block of each arm, seed 3; V3 atlas of the seed-mean maps of the five arms beside the observed map.
