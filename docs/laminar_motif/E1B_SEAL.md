# E1b seal (DRAFT r2 2026-11-19): peak frequency versus SST/PV ratio, band-floor sensitivity and fixed-PV arm

Status: DRAFT r2, after the owner's review of r1 (2026-11-19: revise before sealing). Becomes SEALED on the owner's
approval of every value in the decisions table, committed and pushed before any scored E1b run. A changed criterion
after a run is a new lineage. E1 (`E1_SEAL.md`) stays INCONCLUSIVE with its frozen criteria, whatever E1b finds.

Lineage: SCI-LAMINAR-MOTIF-1, LM-0, E1b. Why: E1 found no oscillation at rho = 4 in any seed. The re-read of the
saved E1 counts (`E1_BAND_FLOOR_DIAG_2026-11-19.txt`, `scripts/e1_band_floor_diag.py`; E1 seeds 1-3 only) shows a
21-22 Hz peak at rho = 4 in all three seeds, peak excess 3.04, 3.26 and 4.78: the sealed 20-80 Hz band flags all
three at its 20 Hz edge, and two are also below the excess threshold of 4. E1b was designed after that re-read.

## Question and estimands

Arm A, sensitivity analysis of the E1 design to the band floor: with E1's sizes and rules and a 10-80 Hz peak band,
does `f_pk` fall as `rho = N_SST / N_PV` rises? This is a sensitivity analysis of E1's readout on fresh seeds, not a
re-scoring of E1.

Arm B, fixed-PV SST variation: with `N_PV = 750` fixed and `N_SST = round(750 rho)`, does `f_pk` fall as `N_SST`
rises? The inference is restricted to SST count varied at fixed PV count. Total inhibitory cells (938 to 3750) and
the E/I cell ratio vary with it, so arm B does not separate SST count from total inhibition. Arm A holds total
inhibitory cells fixed and varies PV count; the two arms bracket the E1 confound and neither resolves it alone. A
matched-total-inhibition arm with PV fixed is not possible in cell counts (fixing PV and the total fixes SST); a
conductance-matched arm is outside this budget and is not run.

Estimand, per arm and seed: the slope of `f_pk` on `log2(rho)`.

## Run matrix (27 runs)

`N_E = 8000` in every run. Seeds S = (E1b-c).

| Run ids | Arm | rho | N_SST | N_PV | N_inh | Seeds |
|---|---|---|---|---|---|---|
| A1-A3 | A | 0.25 | 300 | 1200 | 1500 | S |
| A4-A6 | A | 0.5 | 500 | 1000 | 1500 | S |
| AB7-AB9 | A and B | 1 | 750 | 750 | 1500 | S |
| A10-A12 | A | 2 | 1000 | 500 | 1500 | S |
| A13-A15 | A | 4 | 1200 | 300 | 1500 | S |
| B16-B18 | B | 0.25 | 188 | 750 | 938 | S |
| B19-B21 | B | 0.5 | 375 | 750 | 1125 | S |
| B22-B24 | B | 2 | 1500 | 750 | 2250 | S |
| B25-B27 | B | 4 | 3000 | 750 | 3750 | S |

The rho = 1 runs AB7-AB9 build the E0 module in both arms (the builders are tested equal edge by edge,
`tests/test_e1b_builder.py`) and serve both arms. 3 x 9 conditions = 27 runs.

## Design

As E1: p = 0.07, E0 conductances, delays, time constants and drive (560 private 4 Hz Poisson sources per E and PV
cell, SST undriven), dt 0.1 ms, T 12 s, first 2 s discarded, float32, JaxFNE 0.5.9, CPU. Runner
`scripts/e1b_run.py` (one run per call, counts and receipt saved); builder `caminos_e0.build_spec(n_sst, n_pv)`.

## Readout

E1's `analyze_run` with one change: peak search band 10-80 Hz (E1: 20-80). Aperiodic fit 10-150 Hz, exclusion
+-8 Hz, edge 2 Hz and excess threshold 4 unchanged. The receipt also records the sealed 20-80 Hz call for
reference; it does not enter the verdict.

Peak handling, fixed before any scored run:
1. Absent: peak excess below 4 means not oscillating; `f_pk` undefined; reported, never dropped or re-read.
2. Boundary: `f_pk` within 2 Hz of 10 or 80 Hz means not oscillating (edge rule); reported with its value.
3. Multimodal: `f_pk` is the frequency of the largest residual over the aperiodic fit in the band (JaxFNE
   `aperiodic_spectral_peak`). A second local maximum at least 8 Hz from `f_pk` with excess at least 4 sets a
   `multimodal` flag and is reported with its frequency; it never changes the call or `f_pk`.
4. A run that does not complete is INCOMPLETE and is rerun once with the same seed; a second failure is reported.

## Gates before scoring (a failure stops E1b)

1. Realized sizes equal the matrix; zero PV-to-SST and SST-to-SST edges; block edge counts within 4 SD
   (`tests/test_e1b_builder.py` at rho 0.25 and 4 for arm B, `tests/test_caminos_e0.py` for the default build).
2. E0 and E1 tests pass unchanged; the default build equals the E1 build.
3. Positive control: the sealed 20-80 Hz readout on the saved E1 counts reproduces every E1 call (the diagnostic
   did; it becomes a test before scoring).
4. Null: the permutation statistic on a synthetic case with `f_pk` independent of rho must not pass.

## Verdict rule, per arm

E1's rule unchanged: PASS needs (1) at least 4 of 5 rho oscillating in each seed, (2) a negative slope in every
seed over the oscillating points, (3) mean `f_pk(0.25) - f_pk(4)` at least 3 Hz, (4) one-sided within-seed
permutation p below 0.05 (10 000 draws). FAIL and INCONCLUSIVE as E1. Arm A PASS: the frequency drop holds under
the wider band on fresh seeds. Arm B PASS: the drop holds with PV count fixed while SST count and total inhibition
rise together. Neither result changes E1's verdict.

## Seeds

The isolated module has run seeds 0 (E0 pilot), 1-5 (E0, rho = 1) and 1-3 (E1). Seeds 4 and 5 at rho = 1 are
E0 runs with known outcomes (E0 mean f_pk 34.0 Hz over seeds 1-5); they did not inform the E1b design, band or thresholds (the
diagnostic read E1 seeds 1-3 only). Seeds 6, 7, 8 have no run in this module. The cost smoke runs use seed 0.

## Cost

Smoke run, arm B rho = 4 (the largest build, 11 750 cells), seed 0, T 3 s, `scripts/e1b_run.py` at the r2 draft
(receipt `D:/tmp/_temp_e1b_smoke/e1bB_rho4_seed0_smoke.receipt.json`): realized sizes 8000/750/3000 as declared,
wall 498 s including construction and compilation, peak RSS 2.82 GB. E1's 12 s runs took 1275-1285 s at 9500
cells (A1 receipts). Estimate: arm A runs about 21 min, arm B up to about 33 min at rho = 4 (4 x 498 s, an upper
bound); 27 runs about 11-13 CPU-hours serial. Three in parallel need about 8.5 GB RAM; their speed-up is unmeasured.

Disclosure: the r2 runner computed the spectral readout on the smoke run as well. On seed 0 (not a matrix seed),
1 s kept, arm B rho = 4 gave a 26 Hz peak with excess 22 under the 10-80 Hz band. It was seen before sealing and
changed no value in this draft. The runner now skips the readout for smoke runs.

## Owner decisions (pending)

| Id | Value | Decision |
|---|---|---|
| E1b-a | arms and estimands | A (sensitivity, N_inh 1500) and B (fixed PV 750, inference restricted to SST at fixed PV) |
| E1b-b | rho grid and run matrix | 0.25, 0.5, 1, 2, 4; 27 runs as tabled |
| E1b-c | seeds | 6, 7, 8 (r1 proposed 4, 5, 6; 4 and 5 have E0 runs at rho = 1) |
| E1b-d | peak search band and peak handling | 10-80 Hz, edge 2 Hz; rules 1-4 above |
| E1b-e | excess threshold, drop, alpha | 4, 3 Hz, 0.05 one-sided (as E1) |
| E1b-f | shared rho = 1 runs | AB7-AB9 serve both arms |
