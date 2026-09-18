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
- V2.1_PV_MICROSTATE: **UNRESOLVED** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_microstate.json`
- V2.1_PV_M0_DECOMPOSITION: **PASS** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_m0_r2.json`

### Latest evidence (observed, with receipts)

- V2.1-pv-microstate: trace identity confirmed: the three npz files match the sha256 pinned in the spec and reproduce the sealed population-mean numbers to the printed precision — `9c3b7ad:results/v21_pv_microstate.json`
- V2.1-pv-microstate: level 1 per-cell v: d_AB 0.511 and d_AC 0.513 at epoch 0 against a same-fate scale max d_BC 0.048, qualifying from epoch 0 — `9c3b7ad:results/v21_pv_microstate.json`
- V2.1-pv-microstate: level 1 population mean of v qualifies at the same epoch 0 (scale 0.0476, threshold 0.238) — `9c3b7ad:results/v21_pv_microstate.json`
- V2.1-pv-microstate: level 2 per-cell u: d_AB 0.014-0.046 and d_AC 0.016-0.054 against d_BC 0.008-0.016, no qualifying epoch in the lead window — `9c3b7ad:results/v21_pv_microstate.json`
- V2.1-pv-microstate: level 3 joint (v, u) qualifies from epoch 0 and its population-mean test also qualifies at epoch 0 — `9c3b7ad:results/v21_pv_microstate.json`
- V2.1-pv-microstate: level 4 per-cell ISI structure: d_AB 9.3-16.8 and d_AC 5.9-18.9 against threshold 8.72, no three consecutive qualifying epochs — `9c3b7ad:results/v21_pv_microstate.json`
- V2.1-pv-microstate: level 5 per-cell input was not evaluated: the spec makes it conditional on levels 1-4 producing no qualifying separation — `9c3b7ad:results/v21_pv_microstate.json`
- V2.1-pv-m0-state-decomposition: both states regenerate with exactly the sealed sha256 of dynamic.v (A 83ce0d0c3fc44a71, B 8704423170e6ae32) — `b381896:results/v21_pv_m0.json`
- V2.1-pv-m0-state-decomposition: the trajectory match failed on PV alone by -0.695 Hz against a 0.5 Hz tolerance — `b381896:results/v21_pv_m0.json`
- V2.1-pv-m0-state-decomposition-r2: the corrected trajectory match is exact: 0.0 Hz deviation on E, PV, SST and VIP against run B window 1 — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m0-state-decomposition-r2: three typed blocks differ between h_A(0) and h_B(0): v (400, max 71.4), u (400, max 10.56) and syn_state (159600, max 0.368), each differing in every entry — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m0-state-decomposition-r2: five blocks are bitwise identical: w (159600), H (400), prev_spikes (400), b (400), and the empty theta_S and aux — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m0-state-decomposition-r2: u differs least in PV (max 0.370, mean 0.296) against E (10.56), SST (2.78) and VIP (7.40); v differs comparably in every class (mean 27.6-28.9) — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m0-state-decomposition-r2: both states are saved and pinned for a later authorized transplant: h_A0 sha256 19be5c09, h_B0 sha256 cbd15369 — `8ece148:results/v21_pv_m0_r2.json`

### Next authorized task

- `HARNESS-LINEAGE-RECEIPT` (OPEN): Check declared lineage claims against selected receipt values, not only receipt existence
- stop: no rewriting of sealed results

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
