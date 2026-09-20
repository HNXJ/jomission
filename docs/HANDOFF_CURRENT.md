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
- stopped because: owner assessment 2026-09-19 accepted the terminal state at 100/100 and ordered the scientific program: SCI-EI-INTERACTION-1 (authorized 2x2 factorial, next), then SCI-AGSDR-CALIBRATION (blocked on a viable-regime verdict plus design authorization), then SCI-PROP-QUALIFIED-1 (locked until the plant is defined and the design is authorized). The propagation lineage stands closed as PROPAGATION_DETECTED_ORDERED with no stronger interpretation. No spec beyond the interaction lineage is written

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

- SCI-WS-OSC-2: the ISOLATED_POP arm ran one process, 162 s, with all 246844 edge weights zeroed, weight sum 561.510437 to 0.0, max_abs_after 0.0 and n_nonzero_after 0 — `8f3ab6d:results/whole_system_isolated_pop.json`
- SCI-WS-OSC-2: all 828 cortical E cells fire with every synapse zeroed. None is silent, and the pooled interspike interval over the final 1000 ms is 94.1505 ms with standard deviation 1.3292 ms, CV 0.014118, implying 10.6213 Hz — `8f3ab6d:results/whole_system_isolated_pop.json`
- SCI-WS-OSC-2: the per-cell mean interval spans 93.0 to 95.3 ms over the final window and 93.8043 to 94.4453 ms over the whole run, across cells whose parameters are identical, so that residual spread is v0 and the noise draw alone — `8f3ab6d:results/whole_system_isolated_pop.json`
- SCI-WS-OSC-2: the ISOLATED_1 arm compiled one neuron and zero edges at V1_1.L5.E with a 0.02, b 0.2, c -65, d 8, u0 -13, v0 -6.738214 and drive 5.0, each sliced from the intact construction, and its pooled interval over the final 1000 ms is 94.5 ms with standard deviation 1.2042 ms, CV 0.012742, implying 10.582 Hz — `8f3ab6d:results/whole_system_isolated_1.json`
- SCI-WS-OSC-2: the population-rate estimator applied to the single cell reported a period of 378.0 ms, exactly four times its 94.5 ms interval, with peak autocorrelation 0.268523 against a null threshold of 0.220528 — `8f3ab6d:results/whole_system_isolated_1.json`
- SCI-WS-OSC-2: the population rate of ISOLATED_POP did not flatten. The estimator still found a mode in all six areas at 93 to 95 ms with peak autocorrelation 0.580 to 0.663 against nulls near 0.102 — `8f3ab6d:results/whole_system_isolated_pop.json`
- SCI-WS-OSC-2: the isolated final-window raster shows a smooth 0 to 33 Hz rate wave at about 95 ms with no discrete bursts, the same waveform SCI-WS-OSC-1's EE_CUT arm produced — `8f3ab6d:results/viz/SCI-WS-OSC-2/raster_final.png`
- SCI-EI-REGIME-1: one intact arm completed 13000 ms at g = 0 in a single process; run-health gates execution, collapse and runaway pass; verdict EI_REGIME_UNRESOLVED_MULTIPLE in 155 s of execution — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: E full-run rate 10.6154 Hz at exact 1 ms bins against B1 band 5 to 8 — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: PV full-run rate 0.0621 Hz at exact 1 ms bins against B1 band 12 to 25 — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: SST full-run rate 50.1991 Hz at exact 1 ms bins against B1 band 6 to 12 — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: VIP full-run rate 0.0466 Hz at exact 1 ms bins against B1 band 8 to 15 — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: exact spike totals over the run: E 114264, PV 126, SST 66564, VIP 69 — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: B1 comparison: all four classes out of band and the required E < PV ordering inverted — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: C2 readings: mean realized E->E 0.005488 exceeds E->PV 0.004644 and E->VIP 0.004638 — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: C3 readings: PV (a,b) exactly canonical fast-spiking, SST (a,b) 0.05/0.25 against canonical low-threshold 0.02/0.25, VIP reported with no canonical type — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: C4 readings: SST 50.2 Hz above the B1 top with early dominance, but the SST to PV/VIP weight sums do not both clear zero and PV/VIP net currents read +2.99 Hz-equivalent with positive tonic — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-EI-REGIME-1: V0 schematic and V1 initial raster rendered from this lineage's construction and viewed before the run; V2 final raster reuses V1's ordering and window; V3 atlas built with 5 of 6 panels (plasticity correctly UNAVAILABLE with no HDP run) — `28af326:results/whole_system_ei_regime_diagnosis.json`
- SCI-PROP-TRAJECTORY: V1_1 detects in the test pair with peak per-bin divergence 72.32 against the null-derived threshold 1e-12 — `be18f0d:results/propagation_trajectory.json`
- SCI-PROP-TRAJECTORY: every downstream g=0 null reads at most the threshold in both the test-null and the held-out replicate pair — `be18f0d:results/propagation_trajectory.json`
- SCI-PROP-TRAJECTORY: four downstream areas detect at g=0.5 with sustained onsets: V1_2 11100, V4_1 11360, V4_2 12020, PFC 12700 ms; FEF detects by peak without a sustained onset and is unordered — `be18f0d:results/propagation_trajectory.json`
- SCI-PROP-TRAJECTORY: the 1 s window-count contrast that read exactly 0.000000 everywhere, including the directly driven V1_1, is superseded for propagation use by the 10 ms trajectory divergence; the count finding stands as sealed and is not retracted — `be18f0d:results/propagation_trajectory.json`

### Next authorized task

- `SCI-EI-INTERACTION-1` (OPEN): determine whether repaired E->I weights, canonical SST adaptation, or only their joint application produce a viable cortical E-I regime, via the smallest causal factorial
- stop: four cells, then STOP. No tuning after seeing the four cells; viability is classification against ratified bands, never a tuning target

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
- Pages had no V2.1b/V2.1c entry; resolved 2026-09-19 by PUB-V21B-PAGES (V2.1b, ISI and V2.1c panels live on the activity page) → record only

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
