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

- WS-REALIZE-R4-PARAM: the frozen weight 0.5 is realized on 7020 of 7020 cross-area edges and on 0 of 238800 intra-area edges. Intra-area mean |w| is 0.006321, cross-area exactly 0.500000, a disparity of 79.1x — `7996113:results/parameter_realization_audit.json`
- WS-REALIZE-R4-PARAM: all 24 cross-area identities realize p = 1.0 with uniform |w| = 0.5, receptor index 0, tau 2.0 ms and delay 0 steps. Per receiving cell: ff L2.E->L4.E K = 7.5 (15 afferents), ff L3.E->L4.E K = 10.0 (20), fb L6.E->L1.E K = 14.0 (28), lat L3.E->L3.E K = 10.0 (20) — `7996113:results/parameter_realization_audit.json`
- WS-REALIZE-R4-PARAM: the local reference K for an E cell from intra-area excitation is 0.8660, from intra-area inhibition 0.3921. No long-range projection targets PV, SST or VIP: their long_ff, long_fb and long_lat are all exactly 0.0 — `7996113:results/parameter_realization_audit.json`
- WS-REALIZE-R4-PARAM: tau and delay are not stored per edge. tau_storage is sign_from_receptor, so tau is 2.0 ms for every receptor-0 edge and 5.0 ms for every receptor-1 edge, local and long-range alike; uniform_delay_steps is 0 for the whole edge list — `7996113:results/parameter_realization_audit.json`
- WS-REALIZE-R4-PARAM: the string 'weight' does not appear in results/whole_system_realization_r2.json, so no gate in the realization chain compared a realized weight against the spec — `7996113:results/whole_system_realization_r2.json`
- WS-O-AUTHORITY-1: K_local, the summed local excitatory afferent weight per E cell, is 0.8660. The six layer-wise means span 0.8649 to 0.8683, a spread of 0.0022 — `d38e6cb:results/o_authority_parameterization.json`
- WS-O-AUTHORITY-1: long-range afferents per receiving neuron, counted over all source areas: ff onto L4.E is 35 at V4_1 and V4_2 and 70 at FEF and PFC, over 56 target cells; fb onto L1.E is 28 at V1_1 and V1_2 and 56 at V4_1 and V4_2, over 40 target cells; lat onto L3.E is 20 at all six areas, over 120 target cells — `d38e6cb:results/o_authority_parameterization.json`
- WS-O-AUTHORITY-1: at the frozen w = 0.5 the realized authorities are g_ff 20.21 and 40.41, g_fb 16.17 and 32.33, g_lat 11.55 — `d38e6cb:results/o_authority_parameterization.json`
- WS-AUTH-1: every arm passed the pre-execution realization acceptance in-run: 216 cells checked, being 56 ff, 40 fb and 120 lat targets; max relative error of K_long over K_local minus g was 5.14e-08 at g = 0.05, 0.1 and 0.2 and 5.90e-08 at g = 0.5, measured on the stored float32 weights. 7020 edges reweighted, every other weight bit-identical — `9172036:results/whole_system_authority_bracket.json`
- WS-AUTH-1: all four arms returned WHOLE_SYSTEM_SYNCHRONY_FAIL. Execution, collapse, runaway and drift passed in every arm; synchrony and propagation failed in every arm — `9172036:results/whole_system_authority_bracket.json`
- WS-AUTH-1: mean synchrony CV over the final 1000 ms was 2.9375, 2.9487, 2.9513 and 2.9784 at g = 0.05, 0.1, 0.2 and 0.5, against 3.0510 at the inherited weights in WS-DIAG-2. Areas over the 3.0 tolerance fell from four to one at the three lower g and to two at g = 0.5 — `9172036:results/whole_system_authority_bracket.json`
- WS-AUTH-1: the pooled E burst period is 190 ms at g = 0.05, at g = 0.5 and in WS-DIAG-2, measured identically on the decimated 10 ms series over the final 3 s, with autocorrelation 0.896, 0.873 and 0.880 and pooled E swinging from 0 to about 140 Hz in every case — `9172036:results/whole_system_authority_bracket.json`
- WS-AUTH-1: mean pairwise zero-lag correlation of area E rates over the final 3 s was 0.8381, 0.8147, 0.8332 and 0.8925 at g = 0.05, 0.1, 0.2 and 0.5, against 0.8498 in WS-DIAG-2 — `9172036:results/whole_system_authority_bracket.json`
- WS-AUTH-1: the targeted and untargeted hemifield responded equally: the V1_1 minus V1_2 delta was 0.0000, +0.0145, +0.0145 and 0.0000 Hz across the bracket, against -0.0145 in WS-DIAG-2, while every area including FEF and PFC rose by 0.71 to 0.90 Hz — `9172036:results/whole_system_authority_bracket.json`

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
