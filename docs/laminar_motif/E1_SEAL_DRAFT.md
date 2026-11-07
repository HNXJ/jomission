# E1 seal (DRAFT, 2026-11-07): peak frequency versus SST/PV ratio in one isolated module

Status: DRAFT with proposed values. Not sealed, no run authorized. Every value marked (P) is a proposal for the owner to approve, change or reject; approval turns this file into the seal and it is pushed before any run (AGENTS.md: seal before running). A changed criterion after a run is a new lineage.

Lineage: SCI-LAMINAR-MOTIF-1, minimal model LM-0, experiment E1 (`MINIMAL_MODEL_SPEC.md`). Builds on E0 (PASS, `results/laminar_e0/e0_result.json`): the same network code, JaxFNE 0.5.9.

## Question and estimand

In one isolated CAMINOS module with the one-direction wiring (SST to PV strong, no PV to SST, no SST to SST), does the E-population peak frequency `f_pk` fall as the SST/PV cell-count ratio `rho = N_SST / N_PV` rises, at fixed E/I ratio and fixed drive? Estimand: the slope of `f_pk` on `log2(rho)` per seed. This tests the mechanism ingredient only (Tahvili, Vinck and di Volo). It says nothing about depth profiles or the motif (E2 to E4).

## Design

- Fixed: `N_E = 8000`, `N_inh = N_PV + N_SST = 1500` (so `eps = N_E / N_inh` fixed), p = 0.07, all conductances, delays, time constants and the drive of E0 (560 private 4 Hz Poisson sources per E and PV cell, SST undriven), dt 0.1 ms, T 12 s, first 2 s discarded, float32, JaxFNE 0.5.9, CPU.
- Swept: `rho` on a log grid, `N_SST = round(1500 * rho / (1 + rho))`, `N_PV = 1500 - N_SST` (P):

| rho | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| N_SST | 300 | 500 | 750 | 1000 | 1200 |
| N_PV | 1200 | 1000 | 750 | 500 | 300 |

- Seeds 1, 2, 3 at every `rho` (P); 15 runs. The `rho = 1` runs are the E0 seeds 1 to 3 (same code, same configuration, same seeds); they are reused only if the counts arrays are byte-identical to a fresh `rho = 1` construct check, otherwise rerun. New runs: 12, about 36 min each on CPU, about 7.2 h.
- Why log2 spacing: the AGSDR log-exponential sweep is the owner's standing method for finding values and boundaries; this grid spans a factor 16 in `rho` symmetric about the E0 point.

## Readout

Per run: `analyze_run` of E0 unchanged (`f_pk`, peak excess, oscillates flag, rates). `f_pk` is quantized to 1 Hz (1 s Welch windows); this limits the smallest detectable difference. Band class of `f_pk` (P): gamma `f_pk >= 30 Hz`, beta `13 <= f_pk < 30 Hz`, below that alpha or lower. A run that does not oscillate (peak excess below 4) has `f_pk` undefined and is reported, never dropped.

## Gates before scoring (a failure stops E1)

1. Realized sizes: each construct has exactly the declared `N_SST`, `N_PV`, `N_E`; zero PV-to-SST and SST-to-SST edges; each block's edge count within 4 SD of `p * n_pre * n_post`.
2. E0 tests pass unchanged on the parameterized build (default arguments equal the E0 constants).
3. Positive control: the `rho = 1` build equals the E0 build (same counts for seed 1 on a 2 s probe, byte-equal).
4. Null: the permutation statistic below is computed on a synthetic case where `f_pk` is independent of `rho`; it must not pass.

## Verdict rule (P)

PASS requires all of:
1. at least 4 of the 5 `rho` values oscillate in each seed;
2. the slope of `f_pk` on `log2(rho)`, fitted over the oscillating points, is negative in every seed;
3. mean over seeds of `f_pk(rho = 0.25) - f_pk(rho = 4)` is at least 3 Hz (about three times the seed SD of 0.45 Hz from P5, and three Welch bins);
4. one-sided permutation p below 0.05 for the mean slope, permuting `rho` labels within each seed (10 000 draws).

FAIL: all five `rho` oscillate in every seed and any of 2 to 4 is not met (the falsifier: frequency does not fall as `rho` rises). INCONCLUSIVE: condition 1 fails; reported as such and not counted as a pass or a fail. A run that does not complete is INCOMPLETE. The result file carries a `verdict` field; the status comes from it.

## What a result does and does not mean

PASS: in this module, a higher SST/PV count ratio lowers the peak frequency at fixed E/I and drive, which supports the mechanism ingredient. It does not show a depth profile, a spectrolaminar motif or any biological claim. FAIL is a negative: no retuning, the difference is diagnosed in a new lineage. Alternative explanation not excluded by E1: `N_PV` changes the total PV drive and the recurrent PV loop together with `rho`; E1c and the E/I sweep in E6 address wiring and ratio specificity.

## Owner decisions needed

| Id | Value | Proposed |
|---|---|---|
| E1-a | `rho` grid | 0.25, 0.5, 1, 2, 4 |
| E1-b | seeds per `rho` | 3 (5 would cost 4.8 h more) |
| E1-c | minimum frequency drop | 3 Hz |
| E1-d | permutation alpha | 0.05, one-sided |
| E1-e | band edges for the class label | 13 and 30 Hz |
| E1-f | reuse E0 seeds 1 to 3 for `rho = 1` | yes if byte-identical |
