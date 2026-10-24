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
- execution: jaxfne 0.4.25 (`results/jaxfne_0425_migration.json`)
- program: Closeout program (owner 2026-09-19): empty the todo and problem stacks — harness closeout, context/pages overhaul, code overhaul, and the authorized SCI-EI-REGIME-1 diagnosis; the V2 chain and the coupled-background program are retired with recorded reasons

### Current state

- gate `V2_LOCAL_OPERATION` = **FAIL** (definition `manifests/gates/v2_local_operation.json`)
- stopped because: owner assessment 2026-09-19 accepted the terminal state at 100/100 and ordered the scientific program. SCI-EI-NATIVE-STATE-1 completed and sealed with verdict MIXED_STATE_MECHANISM (PV silence = DYNAMIC_RECRUITMENT_DEFICIT, VIP silence = INTRINSIC_STATE_VETO, SST hyperactivity = SST_STATE_ADVANTAGE; qualification status FAIL). Lineage classification accepted, final state reconciled; STOP for review; no task open (next_authorized_task: null); awaiting reviewer direction before opening downstream work

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
- EI_NATIVE_STATE: **FAIL** (component of INTERNEURON_RATE_REGIME) — `results/whole_system_ei_native_state.json`

### Latest evidence (observed, with receipts)

- SCI-EI-RECRUITMENT-1: PV rate 0.0596 Hz with total drive 2.9939 against fresh rheobase 4.0; reconstructed split withheld by its residual gate — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-RECRUITMENT-1: VIP rate 0.0488 Hz with total drive 2.9933 against fresh rheobase 24.0; reconstructed split withheld by its residual gate — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-RECRUITMENT-1: the mean-field split disagrees with retained-minus-tonic in sign and 20x magnitude, so DEFICIT and VETO readings are withheld as unreliable per the predeclared residual rule — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-RECRUITMENT-1: neither class crosses its rheobase transiently (PV max 3.1869, VIP max 3.1905), so no STATE pattern; both cause UNRESOLVED — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-RECRUITMENT-1: the current-budget atlas aligns per-class rate, total current with tonic and rheobase references, reconstructed splits, and v/u state over the full run — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-NATIVE-STATE-1: PV representative cell receives sub-rheobase net current during SST firing (mean 2.95 < 4.0 rheobase) — `cc45eb0:results/whole_system_ei_native_state.json`
- SCI-EI-NATIVE-STATE-1: VIP representative cell is locked in deep hyperpolarization with positive adaptation state u = +8.58 — `cc45eb0:results/whole_system_ei_native_state.json`
- SCI-EI-NATIVE-STATE-1: SST representative cell autonomous pacemaker firing occurs across all current bins with 653 spikes and zero distance to threshold at crossing — `cc45eb0:results/whole_system_ei_native_state.json`
- JAXFNE-0-5-7-MIGRATION: the compatibility deltas leave the Gen2 realized graph bit-identical across jaxfne 0.4.25 and 0.5.7: 10666 edges, weight sha256 038b5bf1b577f25b343244e36cfdf036e00e96c5f50b0abfb14ad1ba7be4afde, delays {20: 10076, 80: 287, 120: 303}, tau {2.0, 5.0}, 400 neurons — `e9b2367:results/jaxfne_0_5_7_migration.json`
- JAXFNE-0-5-7-MIGRATION: the TFNE construction is identical across jaxfne 0.4.25 and 0.5.7: 2224 neurons, 1294396 edges before enforce and 246844 after (the sealed results/whole_system_isolated_pop.json count), identical weight bytes sha256 8fccec054edeee9dce09166c622c77862e457acdf7fb2df393060fbf3d0df813 and an identical projection identity set — `e9b2367:results/jaxfne_0_5_7_migration.json`
- JAXFNE-0-5-7-MIGRATION: the probe arrays of the Gen2 model are identical under jaxfne 0.4.25 before and after the compatibility deltas, and identical under jaxfne 0.5.0; they differ under 0.5.5 (335 against 337 spikes over 20 ms) and under 0.5.7 (331 against 337) — `e9b2367:results/jaxfne_0_5_7_migration.json`
- JAXFNE-0-5-7-MIGRATION: jaxfne >= 0.5.6 computes the default native-noise coefficient as 0.5*sqrt(DT_REF_MS/dt_ms) with DT_REF_MS 0.5, so at the canonical dt 0.1 ms the per-step coefficient is 1.1180 where jaxfne <= 0.5.5 use 0.5 — `e9b2367:results/jaxfne_0_5_7_migration.json`

### Next authorized task

- `NONE` (no task currently open; awaiting reviewer direction)

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

Work only in `E:\jomission`. Push with
`git push git@github.com:HNXJ/jomission.git <branch>:refs/heads/<branch>`.
