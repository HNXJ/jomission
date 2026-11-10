# Seal: T1 motif fine-tune (lineage `t1`), 2026-11-10

Under PLAN_PAPER_JOMISSION_LANSA.md v2 §2 (owner order 2026-11-10, todo SCI-LAMINAR-MOTIF-1). Committed and pushed
before the first T1 run.

- **Question:** which setting of three local axes brings the 3-unit stack closest to the observed LANSA map, and
  does it pass the shared scorer (`scripts/e2_scorer.py`, pass rule fixed at 88d282d).
- **Size N:** the T1 size from SEAL_LFS.md's rule, applied mechanically to the ladder. Not chosen by hand.
- **Reference:** `e2_3unit_n{N}_dens_k1_kd1_sst15`. Everything not on the active axis stays at the reference
  or at the carried winner.
- **Axes, in order; 7 log-spaced levels each (2^(k/2) steps); seeds 1-3; 8 s; 2 s discard:**

  | Axis | Modifier | Levels |
  |---|---|---|
  | 1 local in-degree density exponent B | `kdB` | 0.25, 0.35, 0.5, 0.71, 1, 1.41, 2 |
  | 2 SST->E decay, all units (ms) | `sstNN` | 5.3, 7.5, 10.6, 15, 21, 30, 42 |
  | 3 deep-unit SST->E decay (ms) | `sstdNN` | 7.5, 10.6, 15, 21, 30, 42, 60 |

  Tags: `t1_{axis}{level}_n{N}_s{seed}_T8`, decimal point written `p` (e.g. `kd0p71`).
- **Selection (seeds 1-3 only):** per level, the seed-mean r_proto (sum row). A level counts only if all 3
  seeds hit no stop state. The axis winner is the highest seed-mean; ties go to the reference level.
  - If the winner passes the scorer on at least 2 of 3 seeds, T1 stops and goes to confirmation.
  - Otherwise the winner carries to the next axis.
  - After axis 3, the carried setting goes to confirmation whatever its verdict.
- **Confirmation:** seeds 4-6, 12 s, config frozen; the scorer verdict per seed is the T1 result. Pass = all 3
  seeds pass. A fail is reported as a fail.
- **Readout ceilings (`scripts/e2_readout_ceiling.py`, σ 0.15, unit depths S -0.4, M 0, D 0.3; computed
  before any T1 run):**

  | Quantity | Coherent | Incoherent |
  |---|---|---|
  | r_proto | 0.854 | 0.693 |
  | r_ab, image 10-30 Hz | 0.949 | 0.926 |
  | r_gm, image 50-150 Hz | 0.853 | 0.657 |
  | r_ab, manuscript 8-30 Hz | 0.948 | 0.923 |
  | r_gm, manuscript 31-150 Hz | 0.849 | 0.640 |

  - The scorer's r_ab ≥ 0.6 and r_gm ≥ 0.6 are within both ceilings, so pass is reachable in principle.
  - The owner's 0.95 target is above every r_proto ceiling. It is reported only as r / ceiling, never as r.
- **Stop states:** a W1 refusal (that level is dropped and reported); any unit E rate above 50 Hz or deep
  unit E rate below 0.1 Hz (the level does not count).
- **Not retuned:** levels, axis order, pass rule and selection rule are fixed here.
- **Visualization (contract):** `results/viz/t1/`. V0: W3 architecture and details table at size N. V1: 1 s
  raster of the reference, seed 1, before any T1 reading. V2: final rasters of the winner (seeds 4-6). V3: SL
  map atlas per axis and level.
