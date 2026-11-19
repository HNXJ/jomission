# E1b seal (DRAFT 2026-11-19): peak frequency versus SST/PV ratio, band floor and PV-count control

Status: DRAFT. Becomes SEALED on the owner's approval of every value in the decisions table, committed and pushed
before any E1b run. A changed criterion after a run is a new lineage.

Lineage: SCI-LAMINAR-MOTIF-1, LM-0, E1b, a new lineage after E1 (`E1_SEAL.md`, verdict INCONCLUSIVE, unchanged).
Why: E1 found no oscillation at rho = 4 in any seed. The re-read of the saved counts
(`E1_BAND_FLOOR_DIAG_2026-11-19.txt`, `scripts/e1_band_floor_diag.py`) shows a 21-22 Hz peak at rho = 4 in all three
seeds, with peak excess 3.04, 3.26 and 4.78. The sealed 20-80 Hz search band flags all three at its 20 Hz edge, and
two fall below the excess threshold of 4. E1b was designed after seeing that re-read; its runs use fresh seeds.

## Question and estimand

Same as E1: in one isolated CAMINOS module, does the E-population peak frequency `f_pk` fall as
`rho = N_SST / N_PV` rises? Estimand: the slope of `f_pk` on `log2(rho)` per seed, per arm.

## Arms

| Arm | Sizes | Holds fixed | Changes with rho |
|---|---|---|---|
| A | E1 sizes: `N_inh = 1500`, `N_SST = round(1500 rho / (1 + rho))` | E/I cell ratio | PV count, PV drive, PV loop |
| B | `N_PV = 750`, `N_SST = round(750 rho)` | PV count, PV drive, PV loop | SST count, total inhibitory cells, E/I cell ratio |

| rho | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| A: N_SST / N_PV | 300 / 1200 | 500 / 1000 | 750 / 750 | 1000 / 500 | 1200 / 300 |
| B: N_SST / N_PV | 188 / 750 | 375 / 750 | 750 / 750 | 1500 / 750 | 3000 / 750 |

Arm B moves the E1 confound (PV count) into the fixed set and lets the E/I cell ratio vary instead; neither arm
alone separates SST count from inhibitory total. At rho = 1 both arms build the E0 module; one set of rho = 1 runs
serves both if the counts arrays are byte-identical between the two builders.

## Design

Everything else as E1: `N_E = 8000`, p = 0.07, E0 conductances, delays, time constants and drive (SST undriven),
dt 0.1 ms, T 12 s, first 2 s discarded, float32, JaxFNE 0.5.9, CPU. Seeds 4, 5, 6 (fresh). Runs: 5 rho x 3 seeds x
2 arms minus 3 shared rho = 1 runs = 27, about 25 min each.

## Readout

E1's `analyze_run` with one change: peak search band 10-80 Hz (E1: 20-80). Edge rule 2 Hz, so a peak counts from
12 Hz. Aperiodic fit 10-150 Hz, exclusion +-8 Hz and excess threshold 4 unchanged. Band classes as E1.

## Gates before scoring (a failure stops E1b)

1. Realized sizes equal the table in each arm; zero PV-to-SST and SST-to-SST edges; block edge counts within 4 SD.
2. E0 and E1 tests pass unchanged; the arm-A builder at default arguments equals the E1 builder.
3. Positive control: the sealed 20-80 Hz readout on the saved E1 counts reproduces every E1 call (done in the
   diagnostic; rerun as a test).
4. Null: the permutation statistic on a synthetic case with `f_pk` independent of rho must not pass.

## Verdict rule, per arm

E1's rule unchanged: PASS needs (1) at least 4 of 5 rho oscillating in each seed, (2) a negative slope in every
seed over the oscillating points, (3) mean `f_pk(0.25) - f_pk(4)` at least 3 Hz, (4) one-sided within-seed
permutation p below 0.05 (10 000 draws). FAIL and INCONCLUSIVE as E1. Arm A PASS supports the E1 ingredient under
the wider band; arm B PASS shows the frequency drop with PV count fixed. Neither result changes E1's verdict.

## Owner decisions (pending)

| Id | Value | Decision |
|---|---|---|
| E1b-a | arms | A and B as tabled |
| E1b-b | rho grid | 0.25, 0.5, 1, 2, 4 |
| E1b-c | seeds | 4, 5, 6 |
| E1b-d | peak search band | 10-80 Hz, edge 2 Hz |
| E1b-e | excess threshold, drop, alpha | 4, 3 Hz, 0.05 one-sided (as E1) |
| E1b-f | shared rho = 1 runs | yes if byte-identical |
