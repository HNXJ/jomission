# E0 seal: CAMINOS reference module on JaxFNE 0.5.8 (pre-execution, 2026-11-06)

Lineage: SCI-LAMINAR-MOTIF-1, minimal model LM-0 (`MINIMAL_MODEL_SPEC.md`, structure approved 2026-11-06). This file is committed and pushed before any E0 run. After a run, a changed criterion, margin or parameter is a new lineage, not an edit here.

## Question and estimand

Does the CAMINOS E-PV-SST reference network, built through the JaxFNE 0.5.8 public API in the jomission environment, reproduce the JaxFNE P5 reference estimands within the margins JaxFNE itself declared? Estimands (p = 0.07, seeds 1 to 5, means over seeds): peak frequency `f_pk`, PV-to-SST lag, mean rates of E, PV, SST. This is a pipeline-equality check. It does not test the paper, and it tests nothing about the laminar hypothesis.

## Reference (read from the JaxFNE P6 preregistration and result file, not from memory)

| Estimand | P5 reference | Margin | Source |
|---|---|---|---|
| `f_pk` | 33.8 Hz | 1 Hz | P6 prereg Q1; result Q1 |
| PV-to-SST lag | 7.04 ms | 0.34 ms | Q2 |
| E rate | 1.106 Hz | 6.1 % | Q3 |
| PV rate | 11.37 Hz | 1.2 % | Q3 |
| SST rate | 7.05 Hz | 8.2 % | Q3 |

Owner decision 2026-11-06: E0 is checked against the JaxFNE P6 references and margins, as above. Correction to my option text: the margins are relative to the P5 port values in this table, not to P6's own numbers (P6 measured 33.80 Hz, 7.22 ms, 1.079, 11.25 and 6.78 Hz, all inside these margins).

## Procedure (JaxFNE P6 recipe, scope reduced to Q1 to Q3)

- Model: parameters exactly as the JaxFNE port specification (`artifacts/etudes/papers/caminos/PORT_SPEC.md`): AdEx E, PV, SST; 8000, 750, 750 cells; Bernoulli p = 0.07; seven blocks, no PV-to-SST and no SST-to-SST edges; conductances, time constants and 2 ms delay as specified; 560 private 4 Hz Poisson sources per E and PV cell at p = 0.07 (`round(p * 8000)`), SST undriven.
- Run: dt 0.1 ms, T 12 s, first 2 s discarded, float32, seeds 1 to 5 through the public API (`connectivity(edge_seed=s)` and the drive seed), the ADVANCED kernel `simulate_adex(record_v=False)` as in P6.
- Analysis: JaxFNE's `caminos_analysis.py`, unchanged, scoring converted per-step population counts. Backend, device, dtype, code SHA and the JaxFNE version are recorded in each run's manifest; the version must equal the project's execution authority (0.5.8).
- Not in scope: Q4 (R1 to R4), Q5 (onset sweep), C1. They check the paper's other panels and are not needed for E1 to E6.

## Gates before scoring (a failure stops E0)

1. Parameter identity: the realized model equals every value in the port specification, exactly.
2. Realized connectivity, seed 1: each block's edge count within 4 SD of `p * n_pre * n_post`; zero autapses; zero PV-to-SST and SST-to-SST edges.
3. Kernel identity: the existing single-cell and recurrent-step tests pass.
4. Converter and null: a known-answer test on a synthetic raster, and a rate-matched Poisson raster scored by the same analysis must not oscillate. A null that oscillates invalidates the analysis.
5. Pilot: one unscored 12 s run, seed 0, for memory and time.

## Verdict rule

PASS requires all five estimands inside their margins. Any estimand outside its margin is FAIL for that estimand and for E0. A run that does not complete at 12 s scores incomplete, which is a fail. The result file carries a `verdict` field; the status comes from it, not from a test run.

| Outcome | Consequence |
|---|---|
| PASS | E1 to E6 become runnable once their own seals are committed |
| FAIL | E0 is reported as a negative; no retuning; E1 to E6 stay blocked; the difference to the JaxFNE P6 setup is diagnosed in a new lineage |
| gate fail or incomplete | STOP and report |

## Uncertainty

Seed SDs from P5 at p = 0.07: E 3.0 %, PV 0.6 %, SST 4.1 % of the mean, lag 0.17 ms, `f_pk` 0.45 Hz. The PV margin (1.2 %) is about twice its seed SD, so a false fail by seed noise alone is possible (P6 itself came within 0.1 percentage point of it, 1.1 %). A PV-only miss within 2 SD is reported as such, never rescued.

## Declared deviations from the P6 setup

JaxFNE 0.5.8 in the jomission environment instead of the P6 code identity; Windows host and its backend instead of WSL2 CUDA if no GPU is used (recorded; spike trains differ at equal seed by construction, so equivalence is on the estimands). Nothing else is changed.

## Evidence class

DERIVED from simulation under the declared model; the reference values are OBSERVED in JaxFNE's own receipts.
