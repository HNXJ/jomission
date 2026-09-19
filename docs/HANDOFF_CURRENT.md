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
- stopped because: WS-DIAG-3 sealed WHOLE_SYSTEM_SYNCHRONY_FAIL with the intended HDP mechanism ON (jomission_authority_v3), matched to the WS-DIAG-2 baseline in every other respect. Every predeclared equilibrium was met and neither H bound was touched, so the verdict is readable as evidence about the architecture. Synchrony improved but did not clear: areas over the 3.0 tolerance fell from four to one, PFC 3.0372. Efficacy redistribution acted almost only on inhibition, because E sits at H = 1 where m(H) = 1 is neutral while silent PV and VIP reach m(7.43) = 1.76. Propagation still unattributable, now with the opposite sign: every area falls about 1 Hz during the stimulus, four of them exactly 11.0 -> 10.0, the signature of a globally periodic burst regime. Observe, localize, STOP: no stabilization or retuning in this lineage. Awaiting reviewer direction.

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

- WS-DIAG-2: execution was finite throughout all 130000 steps, and interface enforcement reproduced exactly: 1047552 intra-retinal edges removed, retinal tonic 0.0, 1024 retinal edges, 24 cortical identities unchanged — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: synchrony still fails and four areas now exceed the 3.0 tolerance: V4.1 3.1642, V4.2 3.1636, PFC 3.0409, V1.2 3.0168, with FEF 2.9425 and V1.1 2.9782 below — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: area E rate slopes over the final 1500 ms are -1.90 to -2.03 Hz/s, and none satisfied the predeclared sustained rule — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: every area moved +0.41 to +0.64 Hz from baseline to stimulus, all within their bands; the targeted V1_1 delta of +0.478 is no larger than the untargeted V1_2 at +0.493 or FEF at +0.594 — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: the lit retinal units again fired at 11.0 Hz and no unlit unit fired — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-2: w and theta_S were bitwise unchanged across all 26 chunk boundaries — `0e76900:results/whole_system_diagnostic_2.json`
- WS-DIAG-3: execution was finite throughout all 130000 steps and interface enforcement reproduced exactly: 1047552 intra-retinal edges removed, 246844 edges after, retinal tonic 0.0, 1024 retinal edges, 24 cortical identities with no missing and no extra — `d4c9d17:results/whole_system_diagnostic_3.json`
- WS-DIAG-3: synchrony CV over the final 1000 ms fell in five of six areas: FEF 2.9425 -> 2.9705, PFC 3.0409 -> 3.0372, V1_1 2.9782 -> 2.8731, V1_2 3.0168 -> 2.7107, V4_1 3.1642 -> 2.7376, V4_2 3.1636 -> 2.8356. Areas over the 3.0 tolerance fell from four to one — `d4c9d17:results/whole_system_diagnostic_3.json`
- WS-DIAG-3: H reached neither bound: final min 0.3626 and max 7.4324 against clamps 0.1 and 10.0, with H_frac_above_5 0.1214 and 270 of 2224 neurons being PV or VIP — `d4c9d17:results/whole_system_diagnostic_3.json`
- WS-DIAG-3: A reached 0.0 exactly and A_frac_zero was 0.0 at every chunk boundary, so no neuron was held at a floor while some reached zero instantaneously — `d4c9d17:results/whole_system_diagnostic_3.json`
- WS-DIAG-3: weights stayed inside their construction bound: w_max 0.500805 -> 0.513842 and w_min -0.008179 -> -0.013081 across the run, with w_changed true — `d4c9d17:results/whole_system_diagnostic_3.json`
- WS-DIAG-3: every area's E rate FELL during the stimulus, by -1.0000 Hz at V1_1, V1_2, V4_1 and V4_2 (11.0 -> 10.0 exactly), -1.1159 at FEF and -1.1014 at PFC, where the HDP-off arm had risen by +0.41 to +0.64 Hz — `d4c9d17:results/whole_system_diagnostic_3.json`
- WS-DIAG-3: the lit retinal units again fired at 11.0 Hz and no unlit unit fired: population rate 0.0 -> 0.6875 Hz over the 1024-unit population — `d4c9d17:results/whole_system_diagnostic_3.json`
- WS-DIAG-3: execution, collapse, runaway and drift all passed; synchrony and propagation failed — `d4c9d17:results/whole_system_diagnostic_3.json`
- WS-REALIZE-R4-PARAM: the frozen weight 0.5 is realized on 7020 of 7020 cross-area edges and on 0 of 238800 intra-area edges. Intra-area mean |w| is 0.006321, cross-area exactly 0.500000, a disparity of 79.1x — `7996113:results/parameter_realization_audit.json`
- WS-REALIZE-R4-PARAM: all 24 cross-area identities realize p = 1.0 with uniform |w| = 0.5, receptor index 0, tau 2.0 ms and delay 0 steps. Per receiving cell: ff L2.E->L4.E K = 7.5 (15 afferents), ff L3.E->L4.E K = 10.0 (20), fb L6.E->L1.E K = 14.0 (28), lat L3.E->L3.E K = 10.0 (20) — `7996113:results/parameter_realization_audit.json`
- WS-REALIZE-R4-PARAM: the local reference K for an E cell from intra-area excitation is 0.8660, from intra-area inhibition 0.3921. No long-range projection targets PV, SST or VIP: their long_ff, long_fb and long_lat are all exactly 0.0 — `7996113:results/parameter_realization_audit.json`
- WS-REALIZE-R4-PARAM: tau and delay are not stored per edge. tau_storage is sign_from_receptor, so tau is 2.0 ms for every receptor-0 edge and 5.0 ms for every receptor-1 edge, local and long-range alike; uniform_delay_steps is 0 for the whole edge list — `7996113:results/parameter_realization_audit.json`
- WS-REALIZE-R4-PARAM: the string 'weight' does not appear in results/whole_system_realization_r2.json, so no gate in the realization chain compared a realized weight against the spec — `7996113:results/whole_system_realization_r2.json`

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
