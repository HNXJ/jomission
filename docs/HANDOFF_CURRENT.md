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
- V2.1_PV_M1_TRANSPLANT: **FAIL** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_m1.json`
- V2.1_PV_M2_PAIR_TRANSPLANT: **PASS** (component of V2.1_COUPLED_BACKGROUND) — `results/v21_pv_m2.json`

### Latest evidence (observed, with receipts)

- V2.1-pv-m0-state-decomposition-r2: the corrected trajectory match is exact: 0.0 Hz deviation on E, PV, SST and VIP against run B window 1 — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m0-state-decomposition-r2: three typed blocks differ between h_A(0) and h_B(0): v (400, max 71.4), u (400, max 10.56) and syn_state (159600, max 0.368), each differing in every entry — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m0-state-decomposition-r2: five blocks are bitwise identical: w (159600), H (400), prev_spikes (400), b (400), and the empty theta_S and aux — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m0-state-decomposition-r2: u differs least in PV (max 0.370, mean 0.296) against E (10.56), SST (2.78) and VIP (7.40); v differs comparably in every class (mean 27.6-28.9) — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m0-state-decomposition-r2: both states are saved and pinned for a later authorized transplant: h_A0 sha256 19be5c09, h_B0 sha256 cbd15369 — `8ece148:results/v21_pv_m0_r2.json`
- V2.1-pv-m1-transplant: A_CTRL reproduces sealed run A with max difference 0.0 Hz and ends low at 0.707 Hz — `ed1be04:results/v21_pv_m1.json`
- V2.1-pv-m1-transplant: B_CTRL ends at 15.110 Hz, matching its sealed reference (run C of the parent, 15.11 Hz) — `ed1be04:results/v21_pv_m1.json`
- V2.1-pv-m1-transplant: each transplant moved exactly its named block: the block took the donor value bitwise and every other DynamicState leaf equalled the base, with one differing leaf per arm — `ed1be04:results/v21_pv_m1.json`
- V2.1-pv-m1-transplant: the PV-incoming synaptic arm covered 15960 of 159600 edges (10 percent), with presynaptic classes E, PV, SST and VIP — `ed1be04:results/v21_pv_m1.json`
- V2.1-pv-m1-transplant: late PV rates: A_v 0.303 Hz, A_u 0.107 Hz, A_syn 0.085 Hz, against an active fate of 14.425 Hz and a base of 0.707 Hz — `ed1be04:results/v21_pv_m1.json`
- V2.1-pv-m1-transplant: E, SST and VIP late rates are within 0.4 Hz across all five arms (E 16.93-17.07, SST 22.36-22.74, VIP 16.93-17.07) — `ed1be04:results/v21_pv_m1.json`
- V2.1-pv-m2-pair-transplant: A_CTRL reproduces sealed run A with max difference 0.0 Hz and ends at 0.707 Hz; B_CTRL ends at 15.110 Hz — `4b74f5e:results/v21_pv_m2.json`
- V2.1-pv-m2-pair-transplant: each arm moved exactly its two named blocks: both took the donor value bitwise and every other DynamicState leaf equalled the base — `4b74f5e:results/v21_pv_m2.json`
- V2.1-pv-m2-pair-transplant: A_vu ends at 15.055 Hz, inside the active band and within 0.06 Hz of the donor control — `4b74f5e:results/v21_pv_m2.json`
- V2.1-pv-m2-pair-transplant: A_vs ends at 0.162 Hz and A_us at 0.000 Hz, both low — `4b74f5e:results/v21_pv_m2.json`
- V2.1-pv-m2-pair-transplant: in M1 the same blocks alone gave A_v 0.303 Hz and A_u 0.107 Hz, below the untouched base of 0.707 Hz — `4b74f5e:results/v21_pv_m1.json`

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
