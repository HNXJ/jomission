# Jomission handoff

Routing document. Values in the generated block come from the manifests and a
tier-0 test fails if they drift. When a manifest and a sealed result disagree,
STOP and surface both.

Startup (AGENTS.md): run `python scripts/project_check.py`, read
`manifests/current_state.json` and the next item in `manifests/todo.json`, open
only the artifacts it names, follow `jomission-gate-runner`. This page is for
routing and contradictions.

## Verify first

```bash
python scripts/project_check.py
```

It must end `PROJECT_CHECK_PASS`: workspace, tree, branch ancestry, remote over
SSH, JaxFNE identity, editable install, all manifests, tier-0 tests. Flags:
`--allow-dirty`, `--offline`, `--no-tests`.

## Objective

Generic cortical operation → functional recurrence → zero-tonic active basin →
propagation → T_R → functional hierarchy → Gen-2 freeze → blind omission.
Retired axes and interpretation rules: `docs/project-sources/SUBSTRATE_PROGRAM.md`.

## State

<!-- generated:state:begin (python -m jomission.harness.handoff) -->

### Authority

- scientific head: main `be0db96`
- execution: jaxfne 0.4.24 (`results/jaxfne_0424_migration.json`)
- program: Closeout program (owner 2026-09-19): empty the todo and problem stacks — harness closeout, context/pages overhaul, code overhaul, and the authorized SCI-EI-REGIME-1 diagnosis; the V2 chain and the coupled-background program are retired with recorded reasons

### Current state

- gate `V2_LOCAL_OPERATION` = **FAIL** (definition `manifests/gates/v2_local_operation.json`)
- stopped because: SCI-WS-OSC-2 sealed INTRINSIC_GENERATOR_CONFIRMED and its stop_condition is reached: two arms, then STOP. The owner set the closeout program 2026-09-19: empty the todo and problem stacks (harness closeout, context/pages overhaul, code overhaul); scientific execution is limited to the authorized SCI-EI-REGIME-1 diagnosis and the one propagation observable the owner picks from the recorded proposals. No other spec is written

- V2.1: **FAIL** — `results/v21_lineage.json`
- V2.1b: **FAIL** — `results/v21b_lineage.json`
- E_TEMPORAL_IRREGULARITY: **PARTIAL** (component of V2.1b) — `results/v21b_isi_classes.json`
- E_RATE_HETEROGENEITY: **FAIL** (component of V2.1b) — `results/v21b_lineage.json`
- H5_PRIVATE_STOCHASTIC_DRIVE: **INSUFFICIENT_ALONE** (component of V2.1b) — `results/v21b_lineage.json`
- V2.1c_INTRINSIC_DESIGN: **INFEASIBLE** — `results/v21c_boundary.json`
- V2.1_COUPLED_BACKGROUND: **UNRESOLVED** — `results/v21_coupled_background.json`
- V2.1_PV_STATIONARITY: **FAIL** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_stationarity.json`
- V2.1_BACKGROUND: **FAIL** (evidence reads BKG_DESIGN_UNRESOLVED; held until a lineage that changes or explains the mechanism producing PV state dependence) — `results/v21_background_design_r2.json`
- V2.1_PV_PREDECESSOR: **UNRESOLVED** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_predecessor.json`
- V2.1_PV_MICROSTATE: **UNRESOLVED** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_microstate.json`
- V2.1_PV_M0_DECOMPOSITION: **PASS** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_m0_r2.json`
- V2.1_PV_M1_TRANSPLANT: **FAIL** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_m1.json`
- V2.1_PV_M2_PAIR_TRANSPLANT: **PASS** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_m2.json`
- PV_INTRINSIC_VU_PAIR_SUFFICIENT: **PASS** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_m2.json`
- WHOLE_SYSTEM_PROPAGATION: **UNRESOLVED** — `results/whole_system_causal_propagation.json`
- WHOLE_SYSTEM_PROPAGATION_HISTORICAL: **QUARANTINED** (component of WHOLE_SYSTEM_PROPAGATION) — `results/whole_system_causal_propagation.json`
- WHOLE_SYSTEM_PERIODIC_MODE: **PASS** — `results/whole_system_oscillator_comparison.json`
- WHOLE_SYSTEM_BURST_STRUCTURE: **UNRESOLVED** (component of WHOLE_SYSTEM_PERIODIC_MODE) — `results/whole_system_oscillator_comparison.json`
- WHOLE_SYSTEM_PERIOD_MEASURED: **PASS** (component of WHOLE_SYSTEM_PERIODIC_MODE) — `results/whole_system_oscillator_comparison.json`
- WHOLE_SYSTEM_PERIOD_HISTORICAL: **QUARANTINED** (component of WHOLE_SYSTEM_PERIOD_MEASURED) — `results/whole_system_oscillator_comparison.json`
- WHOLE_SYSTEM_INTRINSIC_GENERATOR: **PASS** — `results/whole_system_isolated_comparison.json`
- E_RATE_AND_PERIOD_ARE_ONE_DOF: **PASS** (component of WHOLE_SYSTEM_INTRINSIC_GENERATOR) — `results/whole_system_isolated_pop.json`
- INTERNEURON_RATE_REGIME: **FAIL** (component of WHOLE_SYSTEM_INTRINSIC_GENERATOR) — `results/whole_system_oscillator_single_area.json`

### Latest evidence (observed, with receipts)

- WS-PROP-1: all six cells ran one process each, 152.9 to 153.2 s, and each wrote a complete result with verdict WHOLE_SYSTEM_SYNCHRONY_FAIL as predeclared — `9e2372b:results/whole_system_causal_propagation.json`
- WS-PROP-1: R_A(g = 0.5) is 0.000000 in all six areas, and the ON minus OFF paired difference is exactly 0.0 in every area of all three pairs: the g = 0.5 test pair, the g = 0 control pair and the g = 0 replicate at seed 1 — `9e2372b:results/whole_system_causal_propagation.json`
- WS-PROP-1: the paired difference is exactly 0.0 at V1_1 as well, the area the retina drives directly, so the observable has no sensitivity at the point of injection — `9e2372b:results/whole_system_causal_propagation.json`
- WS-PROP-1: the held-out null floor over V4_1, V4_2, FEF and PFC is max_abs 0.0 and rms 0.0 with is_degenerate true, and evaluate refused with ValueError rather than substituting a default tolerance — `9e2372b:results/whole_system_causal_propagation.json`
- WS-PROP-1: across all 36 area-cells the baseline plus stimulus window total is exactly 21.000000 spikes per E cell: six areas by six cells, both couplings, both seeds, ON and OFF, with no exception — `9e2372b:results/whole_system_causal_propagation.json`
- WS-PROP-1: the stimulus does act. At g = 0 the retinal receipt is 64 lit units, population 0.6875 Hz and an implied 11.0 Hz per lit unit against 0, 0.0 and undefined for OFF, and V1_1's synchrony CV moves 2.896 to 2.9326 and its drift slope -1.8038 to -1.7754 while its window counts do not move at all — `9e2372b:results/whole_system_causal_propagation.json`
- WS-PROP-1: zero-threshold presence diagnostic: at g = 0 the ON and OFF runs differ in 12 V1_1 groups and in no group of any other area, all 133 groups compared. At g = 0.5 they differ in every area — `9e2372b:results/whole_system_causal_propagation.json`
- SCI-WS-OSC-1: the SINGLE_AREA arm ran one process, 156 s, and wrote a complete result with the whole-system gate evaluated in the sealed order, verdict WHOLE_SYSTEM_SYNCHRONY_FAIL from V4_2 at CV 3.0129 against a 3.0 threshold with the other five areas at 2.83 to 2.95 — `3b239c9:results/whole_system_oscillator_single_area.json`
- SCI-WS-OSC-1: the EE_CUT arm ran one process, 157 s, and wrote a complete result with the same gate in the same order. Synchrony passes and the first load-bearing failure is propagation, the field WS-PROP-1 qualified NON_IDENTIFYING — `3b239c9:results/whole_system_oscillator_ee_cut.json`
- SCI-WS-OSC-1: the EE_CUT arm realized the cut as specified: 113436 edges targeted, weight sum 526.727051 before and 0.0 after, max_abs_after 0.0, and 133408 edges untouched with weight sum 34.78334 — `3b239c9:results/whole_system_oscillator_ee_cut.json`
- SCI-WS-OSC-1: a periodic mode is present in all six areas of both arms, above each area's own Poisson null of about 0.104 to 0.107 — `3b239c9:results/whole_system_oscillator_comparison.json`
- SCI-WS-OSC-1: the period is 94 to 95 ms in the intact arm and 94 to 95 ms in the cut arm. The intact pooled measurement is 94.0 ms, 10.6383 Hz, peak autocorrelation 0.885737 — `3b239c9:results/whole_system_oscillator_comparison.json`
- SCI-WS-OSC-1: peak autocorrelation falls in every area when E->E is cut, to between 0.64 and 0.74 of its intact value: V1_1 0.832 to 0.612, V1_2 0.848 to 0.593, V4_1 0.849 to 0.584, V4_2 0.850 to 0.574, FEF 0.855 to 0.544, PFC 0.834 to 0.550 — `3b239c9:results/whole_system_oscillator_comparison.json`
- SCI-WS-OSC-1: the intact arm's final 1000 ms is discrete synchronous bursts reaching about 100 Hz with near silence between them — `3b239c9:results/viz/SCI-WS-OSC-1/raster_final.png`
- SCI-WS-OSC-1: the cut arm's final 1000 ms over the same window, ordering and conventions shows no discrete bursts: the population rate is a smooth 3 to 30 Hz wave at the same period — `3b239c9:results/viz/SCI-WS-OSC-1/raster_final_ee_cut.png`
- SCI-WS-OSC-1: results/whole_system_authority_g000.json contains no period, frequency or autocorrelation field anywhere in its structure, and before this lineage no module in the repository computed one. The 190 ms figure quoted in prose has no receipt — `3b239c9:results/whole_system_authority_g000.json`
- SCI-WS-OSC-2: the ISOLATED_POP arm ran one process, 162 s, with all 246844 edge weights zeroed, weight sum 561.510437 to 0.0, max_abs_after 0.0 and n_nonzero_after 0 — `8f3ab6d:results/whole_system_isolated_pop.json`
- SCI-WS-OSC-2: all 828 cortical E cells fire with every synapse zeroed. None is silent, and the pooled interspike interval over the final 1000 ms is 94.1505 ms with standard deviation 1.3292 ms, CV 0.014118, implying 10.6213 Hz — `8f3ab6d:results/whole_system_isolated_pop.json`
- SCI-WS-OSC-2: the per-cell mean interval spans 93.0 to 95.3 ms over the final window and 93.8043 to 94.4453 ms over the whole run, across cells whose parameters are identical, so that residual spread is v0 and the noise draw alone — `8f3ab6d:results/whole_system_isolated_pop.json`
- SCI-WS-OSC-2: the ISOLATED_1 arm compiled one neuron and zero edges at V1_1.L5.E with a 0.02, b 0.2, c -65, d 8, u0 -13, v0 -6.738214 and drive 5.0, each sliced from the intact construction, and its pooled interval over the final 1000 ms is 94.5 ms with standard deviation 1.2042 ms, CV 0.012742, implying 10.582 Hz — `8f3ab6d:results/whole_system_isolated_1.json`
- SCI-WS-OSC-2: the population-rate estimator applied to the single cell reported a period of 378.0 ms, exactly four times its 94.5 ms interval, with peak autocorrelation 0.268523 against a null threshold of 0.220528 — `8f3ab6d:results/whole_system_isolated_1.json`
- SCI-WS-OSC-2: the population rate of ISOLATED_POP did not flatten. The estimator still found a mode in all six areas at 93 to 95 ms with peak autocorrelation 0.580 to 0.663 against nulls near 0.102 — `8f3ab6d:results/whole_system_isolated_pop.json`
- SCI-WS-OSC-2: the isolated final-window raster shows a smooth 0 to 33 Hz rate wave at about 95 ms with no discrete bursts, the same waveform SCI-WS-OSC-1's EE_CUT arm produced — `8f3ab6d:results/viz/SCI-WS-OSC-2/raster_final.png`

### Next authorized task

- `HD-STALE-EXPECTATIONS` (OPEN): Owner repair of stale bounds / quarantine cover as classified
- stop: no silent bound loosening

### Locked gates

- 

### Critical invariants

- `ONE_DELTA`, `PRE_EXECUTION_SEAL`, `NO_RETUNE_AFTER_RESULTS`, `DRIVE_ADDITIVITY`, `ESTIMATOR_METADATA`, `CONFIGURED_REALIZED_EXECUTED_EFFECTIVE`, `OMISSION_FIREWALL`, `SEALED_ARTIFACT_IMMUTABILITY`, `WORKSPACE_PROVENANCE`, `NATIVE_NOISE_DEFAULT`, `VISUALIZATION_CONTRACT` — definitions in `docs/project-sources/SUBSTRATE_PROGRAM.md`

### Open uncertainties

- E b_crit at I=0 lies below the linear rest-state instability (bistability); boundary shifts ~1e-5 with initial state, duration, dt → record only
- only single-parameter lognormal E dispersions were tested; joint or other-distribution intrinsic dispersion is untested and not authorized → record only
- low_sm2p0 VIP ~130 Hz exceeds the prospective VIP [1,80] band; the shot background is not class-generic → SCI-V21-BACKGROUND
- silent E cells escape the ISI gate but raise E rate-CV → SCI-V21-GATE-REPAIR
- VIP->E pathway (8400 edges) omitted from the V2.1 battery's recorded currents → record only
- PV single-cell threshold lies between F-table grid points at I=2.8; tonic derivation uses exact assays → record only
- HDP re-entry point unspecified beyond 'after the basin gate'; V2-ACTIVE-BASIN retired 2026-09-19, so the route is closed → record only
- shot parameters outside the sealed 9-cell family untested (no new brackets authorized) → SHOT_NOISE_BRACKETS retired
- whether SST-c/VIP-c single-cell bursting entrains E irregularity (secondary hypothesis) → record only
- cross-process float32 floor ~1e-4 on active-motif w/I means (rates exact) → record only
- Pages has no V2.1b/V2.1c entry → PUB-V21B-PAGES

<!-- generated:state:end -->

## Repository map

| need | owner |
|---|---|
| state, verdicts, locked gates, uncertainties | `manifests/current_state.json` |
| work items (one next task) | `manifests/todo.json` |
| objective, retired axes, invariants, interpretation rules | `docs/project-sources/SUBSTRATE_PROGRAM.md` |
| gate thresholds and estimators | `manifests/gates/v2_local_operation.json` via `jomission.harness.gates` |
| lineage edges with receipts | `manifests/lineage_registry.json` |
| immutable files | `manifests/sealed_artifacts.json` |
| status labels vs evidence classes | `manifests/vocabulary.json` |
| which tests for which change | `manifests/test_tiers.json` |
| known environment failures | `manifests/harness/environment_failures.json` |
| procedures | `.claude/skills/jomission-gate-runner`, `.claude/skills/jomission-lineage-review` |
| publication | `site-src/manifest.json`, `scripts/build_pages.py` |

Historical only: `docs/PROJECT_HANDOFF.md`, `docs/sessions/`,
`manifests/project_truth_ledger.json`, `manifests/environment.json`. The
narrative handoff this page replaced is `a488348:docs/HANDOFF_CURRENT.md`;
superseded V2.1 wording is listed in `results/v21_provenance_correction.json`.

## Fresh-Actor instruction

> Reconstruct the state from repository evidence before acting. Treat this handoff as a routing map, not authority over executed artifacts. Independently challenge the current conclusion before implementing the next step. If repository evidence contradicts this handoff, STOP and surface the conflict. Make the smallest authorized scientific delta, verify it, seal it, and stop at the first failed gate.

Work only in `E:\repos\jomission`. Push with
`git push git@github.com:HNXJ/jomission.git <branch>:refs/heads/<branch>`.
