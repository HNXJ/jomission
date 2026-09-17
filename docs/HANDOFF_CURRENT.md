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
- program: Generic Substrate V2 (results/generic_substrate_v2_spec.json rev2 + amendment 1; amendment 2 in results/generic_substrate_v2_amendment_2.json)

### Current state

- gate `V2_LOCAL_OPERATION` = **FAIL** (definition `manifests/gates/v2_local_operation.json`)
- stopped because: SCI-V21-PV-STATE-PREDECESSOR: MISSING_OBSERVABLE (results/v21_pv_state_predecessor.json, branch v21-pv-state-predecessor); background family STOPPED; no instrumented rerun authorized

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

### Latest evidence (observed, with receipts)

- V2.1-coupled-background: endpoint runs reproduce the sealed V2.1 rates to 3.6e-15 Hz (lambda 0 and 1) — `54ff926:results/v21_coupled_background.json`
- V2.1-coupled-background: class_rate_bands fail at lambda 0 (PV 0.0 Hz) and 0.25 (PV window rates 1.40, 1.22, 0.89 Hz) and pass at 0.3125, 0.375, 0.5, 0.65625 and 1; passing run [0.3125, 1], boundary in (0.25, 0.3125] — `54ff926:results/v21_coupled_background.json`
- V2.1-coupled-background: mean PV rate rises with lambda: 0.0, 1.17, 3.73, 8.13, 14.32 Hz at 0, 0.25, 0.3125, 0.375, 0.5; E and VIP 10.67 to 15.6 Hz; SST 10.78 to 20.22 Hz — `54ff926:results/v21_coupled_background.json`
- V2.1-coupled-background: candidate lambda 0.65625 (margin 0.40625) passes the bands fresh, but its PV window rates are 16.88, 16.58, 10.2 Hz (PV drift 0.46) — `54ff926:results/v21_coupled_background.json`
- V2.1-coupled-background: at lambda 0.65625 continued from the lambda-0 end state, PV window rates are 17.05, 15.36, 2.01 Hz (mean 11.47, -3.08 Hz vs fresh; tolerance 1.455 Hz); continued from the lambda-1 end state, 14.65, 14.69, 14.66 Hz (+0.11 Hz). All three runs pass the bands; E, SST, VIP agree within tolerance — `54ff926:results/v21_coupled_background.json`
- V2.1-pv-stationarity: windows 1-3 of all three trajectories reproduce the sealed 20 s runs exactly (max difference 0.0 Hz) — `907c7f2:results/v21_pv_stationarity.json`
- V2.1-pv-stationarity: lambda-1 history: PV 14.36-14.74 Hz in windows 1-11; late drift 0.012; bands pass — `907c7f2:results/v21_pv_stationarity.json`
- V2.1-pv-stationarity: fresh: PV 16.88, 16.58, 10.2, 16.77, 13.79, 1.68, 1.22, 1.03, 0.85, 0.87, 0.4 Hz; late PV mean 0.71 Hz, late PV drift 0.672; bands fail (PV < 1 Hz) — `907c7f2:results/v21_pv_stationarity.json`
- V2.1-pv-stationarity: lambda-0 history: PV 17.05, 15.36, 2.01, then 0.07-0.24 Hz in windows 4-11; late PV mean 0.09 Hz, late PV drift 0.589; bands fail — `907c7f2:results/v21_pv_stationarity.json`
- V2.1-pv-stationarity: E (16.8-17.2 Hz), VIP (16.9-17.2 Hz) and SST (19.9-22.8 Hz) are nearly identical across the three trajectories; late pairwise differences <= 0.37 Hz, while late PV differs by 13.7-14.3 Hz — `907c7f2:results/v21_pv_stationarity.json`
- V2.1-pv-predecessor-instrumented: runs A and B reproduce the sealed 60 s window rates for every class and window with max difference 0.0 Hz; Delta_instrumentation = 0 — `b4130f3:results/v21_pv_predecessor.json`
- V2.1-pv-predecessor-instrumented: u reconstruction matches the engine carry at every chunk boundary to 8.5e-06 or better in all three runs — `b4130f3:results/v21_pv_predecessor.json`
- V2.1-pv-predecessor-instrumented: the engine source term includes the tonic: residual after removing tonic and the reconstructed class inputs has mean 0.016-0.030 against a reconstructed input scale of 0.202-0.231 — `b4130f3:results/v21_pv_predecessor.json`
- V2.1-pv-predecessor-instrumented: late PV rate: A 0.71 Hz, B 14.43 Hz, C 15.11 Hz; C and B share the same starting state hash 8704423170e6ae32 and differ only in key stream — `b4130f3:results/v21_pv_predecessor.json`
- V2.1-pv-predecessor-instrumented: no recorded channel (v_PV, u_PV, sources, the four class inputs, the two residuals) leaves its own 0-5 s baseline by 5 sigma sustained 1 s in any run; the PV population rate does so at 15 s in run A — `b4130f3:results/v21_pv_predecessor.json`
- V2.1-pv-predecessor-instrumented: the run A collapse is uniform across the PV population: early per-cell 16.40-17.20 Hz, late 0.00-2.60 Hz, one cell below 0.1 Hz — `b4130f3:results/v21_pv_predecessor.json`

### Next authorized task

- `SCI-V21-PV-MICROSTATE` (OPEN): decide whether the fate-carrying information already exists in the recorded microscopic state, before any further execution
- stop: classification recorded in results/v21_pv_microstate.json; no new execution and no intervention follows automatically; STOP for review

### Locked gates

- `V2.2_RECURRENCE`, `V2_ACTIVE_BASIN`, `V2_PROPAGATION`, `V2_RESPONSE_TIMESCALE`, `V2_HIERARCHY`, `GEN2_FREEZE`, `BLIND_OMISSION`

### Critical invariants

- `ONE_DELTA`, `PRE_EXECUTION_SEAL`, `NO_RETUNE_AFTER_RESULTS`, `DRIVE_ADDITIVITY`, `ESTIMATOR_METADATA`, `CONFIGURED_REALIZED_EXECUTED_EFFECTIVE`, `OMISSION_FIREWALL`, `SEALED_ARTIFACT_IMMUTABILITY`, `WORKSPACE_PROVENANCE`, `NATIVE_NOISE_DEFAULT` — definitions in `docs/project-sources/SUBSTRATE_PROGRAM.md`

### Open uncertainties

- E b_crit at I=0 lies below the linear rest-state instability (bistability); boundary shifts ~1e-5 with initial state, duration, dt → record only
- only single-parameter lognormal E dispersions were tested; joint or other-distribution intrinsic dispersion is untested and not authorized → record only
- low_sm2p0 VIP ~130 Hz exceeds the prospective VIP [1,80] band; the shot background is not class-generic → SCI-V21-BACKGROUND
- silent E cells escape the ISI gate but raise E rate-CV → SCI-V21-GATE-REPAIR
- VIP->E pathway (8400 edges) omitted from the V2.1 battery's recorded currents → record only
- PV single-cell threshold lies between F-table grid points at I=2.8; tonic derivation uses exact assays → record only
- HDP re-entry point unspecified beyond 'after the basin gate' → V2-ACTIVE-BASIN
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
