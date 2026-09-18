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
- stopped because: WS-DIAG-2 sealed WHOLE_SYSTEM_SYNCHRONY_FAIL at the same gate as WS-DIAG-1 with a 5x longer settle. The synchrony is not a settling artifact and the retinal interface drives no attributable response. Observe, localize, STOP: no stabilization in this lineage. Awaiting reviewer direction.

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

### Latest evidence (observed, with receipts)

- WS-REALIZE-R4C: the engine's sequential allocation policy preserves sum_c N[l.c] = N[l] in all 12006 audited cases, six layers by layer totals 0 to 2000 — `0659293:results/ctx_allocation_audit.json`
- WS-REALIZE-R4C: the independent per-class formula int(round(N_l*P)) violates that invariant in 5031 of the same 12006 cases — `0659293:results/ctx_allocation_audit.json`
- WS-REALIZE-R4C: at N = 200 the layer split is L1 20, L2 30, L3 40, L4 20, L5 60, L6 30, summing to 200, and the class counts sum to 200 — `0659293:results/ctx_allocation_audit.json`
- WS-REALIZE-R4C: with R adopted, delta_objects = delta_populations = delta_projections = delta_index = 0 and every declared interface resolves — `0659293:results/whole_system_realization_r2.json`
- WS-REALIZE-R4C: the corrected run reproduces the parent's topology exactly: the same 24 identities, 7020 cross-area edges of 245820, no skip projections — `0659293:results/whole_system_realization_r2.json`
- WS-REALIZE-R4C: per-edge tau_ms is not readable from the constructed model; the configured value is 2 ms — `0659293:results/whole_system_realization_r2.json`
- WS-DIAG-1: execution was finite throughout: no non-finite v, u or synaptic current at any of the 50000 steps — `50bd3c5:results/whole_system_diagnostic.json`
- WS-DIAG-1: the input interface was enforced and verified: 1047552 intra-retinal edges removed, retinal tonic 5.0 zeroed, exactly 1024 retinal edges realized, and the 24 cortical projection identities unchanged — `50bd3c5:results/whole_system_diagnostic.json`
- WS-DIAG-1: the retina was silent at baseline and its 64 lit units fired at 11.0 Hz during stimulation — `50bd3c5:results/whole_system_diagnostic.json`
- WS-DIAG-1: PFC E population CV over the final 1000 ms is 3.1138, above the predeclared tolerance of 3.0; FEF 2.9046, V1.1 2.8988, V1.2 2.8880, V4.1 2.8807, V4.2 2.7960 — `50bd3c5:results/whole_system_diagnostic.json`
- WS-DIAG-1: no area E rate fell below 0.1 Hz and no group sustained over 200 Hz or full activation for 100 ms, so collapse and runaway passed — `50bd3c5:results/whole_system_diagnostic.json`
- WS-DIAG-1: area E rate slopes over the final 1500 ms are 1.25 to 1.90 Hz/s, above the 0.5 Hz/s tolerance, but none satisfied the predeclared sustained rule — `50bd3c5:results/whole_system_diagnostic.json`
- WS-DIAG-1: the directly targeted V1_1.L4.E fell 1.43 Hz from baseline to stimulus while untargeted V1_2.L4.E rose 3.57 Hz and V4_1.L4.E rose 2.14 Hz — `50bd3c5:results/whole_system_diagnostic.json`
- WS-DIAG-1: w and theta_S were bitwise unchanged across all ten chunk boundaries, confirming no plastic stabilization under the baseline kernel — `50bd3c5:results/whole_system_diagnostic.json`
- WS-DIAG-2: execution was finite throughout all 130000 steps, and interface enforcement reproduced exactly: 1047552 intra-retinal edges removed, retinal tonic 0.0, 1024 retinal edges, 24 cortical identities unchanged — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: synchrony still fails and four areas now exceed the 3.0 tolerance: V4.1 3.1642, V4.2 3.1636, PFC 3.0409, V1.2 3.0168, with FEF 2.9425 and V1.1 2.9782 below — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: area E rate slopes over the final 1500 ms are -1.90 to -2.03 Hz/s, and none satisfied the predeclared sustained rule — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: every area moved +0.41 to +0.64 Hz from baseline to stimulus, all within their bands; the targeted V1_1 delta of +0.478 is no larger than the untargeted V1_2 at +0.493 or FEF at +0.594 — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: the lit retinal units again fired at 11.0 Hz and no unlit unit fired — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: w and theta_S were bitwise unchanged across all 26 chunk boundaries — `0e76900:results/whole_system_diagnostic_2.json`

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
