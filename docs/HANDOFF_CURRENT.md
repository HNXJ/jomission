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

- scientific head: main `4859d3c`
- execution: jaxfne 0.4.24 (`results/jaxfne_0424_migration.json`)
- program: Generic Substrate V2 (results/generic_substrate_v2_spec.json rev2 + amendment 1; amendment 2 in results/generic_substrate_v2_amendment_2.json)

### Current state

- gate `V2_LOCAL_OPERATION` = **FAIL** (definition `manifests/gates/v2_local_operation.json`)
- stopped because: reviewer D1-D3 (2026-09-17): isolated PV/SST retuning rejected; coupled network operating point authoritative. SCI-V21-COUPLED-BACKGROUND opened with a sealed spec (results/v21_coupled_background_spec.json)

- V2.1: **FAIL** — `results/v21_lineage.json`
- V2.1b: **FAIL** — `results/v21b_lineage.json`
- E_TEMPORAL_IRREGULARITY: **PARTIAL** (component of V2.1b) — `results/v21b_isi_classes.json`
- E_RATE_HETEROGENEITY: **FAIL** (component of V2.1b) — `results/v21b_lineage.json`
- H5_PRIVATE_STOCHASTIC_DRIVE: **INSUFFICIENT_ALONE** (component of V2.1b) — `results/v21b_lineage.json`
- V2.1c_INTRINSIC_DESIGN: **INFEASIBLE** — `results/v21c_boundary.json`

### Latest evidence (observed, with receipts)

- V2.1c-design: reduced model validation passed — `07d3966:results/v21c_validation.json`
- V2.1c-design: b reaches E rate-CV 0.377 at w=0.3 with class-mean rheobase shift 0.421 — `07d3966:results/v21c_width_b.json`
- V2.1c-design: E cells with b >= 0.265 fire at I=0; 13.7% of cells at w=0.3 — `07d3966:results/v21c_design_notes.json`
- V2.1c-boundary: all 8 predeclared gates pass; ordering SPONTANEOUS_BEFORE_CV — `d3a0c3c:results/v21c_boundary.json`
- V2.1c-boundary: b_crit at I=0 in (0.2604980, 0.2604987] — `d3a0c3c:results/v21c_boundary.json`
- V2.1c-boundary: w_spontaneous in (0.091406, 0.091504]; first pacemaker b 0.26055 at 5.8 Hz — `d3a0c3c:results/v21c_boundary.json`
- V2.1c-boundary: w_CV=0.3 in (0.234375, 0.2375]: min-window E rate-CV 0.2972 / 0.3005; 31-32 of 300 E cells spontaneous — `d3a0c3c:results/v21c_boundary.json`
- V2.1c-boundary: E rate-CV at w_spontaneous 0.1197; sealed CV at w=0.20 and 0.30 reproduced exactly — `d3a0c3c:results/v21c_boundary.json`
- V2.1-bkg-pv-calibration: isolated PV class-mean rate 0.0 Hz at I = 2.8 and at I = 3.6 in every analysis window; 10 Hz not bracketed, no bisection run — `d70b575:results/v21_bkg_pv_calibration.json`
- V2.1-bkg-pv-calibration: network PV at I0 2.8 (V_MID): 0.0 Hz in every window — `d70b575:results/v21_mid.json`
- V2.1-bkg-pv-calibration: network PV at I0 3.6 (V_HIGH): 20.0-20.2 Hz — `d70b575:results/v21_high.json`
- V2.1-bkg-pv-calibration: pre-seal timing probe (I = 3.6, seed 1) gave 0.0 Hz; disclosed in the spec field pre_seal_disclosure — `d70b575:results/v21_bkg_pv_calibration_spec.json`

### Next authorized task

- `SCI-V21-COUPLED-BACKGROUND` (OPEN): find whether a finite interior coupled tonic background on the V_MID to V_HIGH path satisfies all prospective_v2 class-rate bands
- stop: verdict recorded; candidate lambda reported, not frozen; no susceptibility measurement, no shot-noise design, ISI-CV/synchrony/heterogeneity are not criteria; STOP for review

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
