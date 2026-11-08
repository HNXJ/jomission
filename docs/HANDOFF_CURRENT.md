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
- execution: jaxfne 0.5.8 (`results/jaxfne_058_migration.json`)
- program: Closeout program (owner 2026-09-19): empty the todo and problem stacks — harness closeout, context/pages overhaul, code overhaul, and the authorized SCI-EI-REGIME-1 diagnosis; the V2 chain and the coupled-background program are retired with recorded reasons

### Current state

- gate `V2_LOCAL_OPERATION` = **FAIL** (definition `manifests/gates/v2_local_operation.json`)
- stopped because: owner assessment 2026-09-19 accepted the terminal state at 100/100 and ordered the scientific program. SCI-EI-NATIVE-STATE-1 completed and sealed with verdict MIXED_STATE_MECHANISM (PV silence = DYNAMIC_RECRUITMENT_DEFICIT, VIP silence = INTRINSIC_STATE_VETO, SST hyperactivity = SST_STATE_ADVANTAGE; qualification status FAIL). Lineage classification accepted, final state reconciled; STOP for review; the STOP recorded here is superseded by next_authorized_task below (SCI-AGSDR-CALIBRATION, open); awaiting reviewer direction before downstream work beyond it

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

- SCI-EI-NATIVE-STATE-1: PV representative cell receives sub-rheobase net current during SST firing (mean 2.95 < 4.0 rheobase) — `cc45eb0:results/whole_system_ei_native_state.json`
- SCI-EI-NATIVE-STATE-1: VIP representative cell is locked in deep hyperpolarization with positive adaptation state u = +8.58 — `cc45eb0:results/whole_system_ei_native_state.json`
- SCI-EI-NATIVE-STATE-1: SST representative cell autonomous pacemaker firing occurs across all current bins with 653 spikes and zero distance to threshold at crossing — `cc45eb0:results/whole_system_ei_native_state.json`
- SCI-LAMINAR-MOTIF-1: E0 CAMINOS reference replication on JaxFNE 0.5.9: all 5 estimands within P6 margins across seeds 1-5; peak frequency mean 34.0 Hz — `ea249f1:results/laminar_e0/e0_result.json`
- SCI-LAMINAR-MOTIF-1: E0 PV-to-SST lag mean 7.16 ms within declared margin 0.34 ms of P5 reference 7.04 ms — `ea249f1:results/laminar_e0/e0_result.json`
- SCI-LAMINAR-E1: E1 frequency slope is strictly negative in all 3 seeds (slopes -6.6, -5.9, -6.3; mean -6.27 Hz/octave; permutation p = 0.0000 < 0.05) — `8200fac:results/laminar_e1/e1_result.json`
- SCI-LAMINAR-E1: E1 peak frequency shifts monotonically from high gamma to beta across oscillating ratios (mean f_pk: rho=0.25 is 46.0 Hz, rho=0.5 is 39.7 Hz, rho=1.0 is 34.0 Hz, rho=2.0 is 27.0 Hz) — `8200fac:results/laminar_e1/e1_result.json`
- SCI-LAMINAR-E1: At rho = 4.0 (N_SST = 1200, N_PV = 300), the network leaves the oscillatory regime (peak excess < 4 or within 2 Hz of band edge), leaving condition 3 f_pk(0.25) - f_pk(4) undefined — `8200fac:results/laminar_e1/e1_result.json`

### Next authorized task

- `SCI-AGSDR-CALIBRATION` (OPEN): calibrate AGSDR per-class rates against the EI-defined plant
- stop: stage 0 run and correction packet done (packet f23f8f4; audit fixes a30ce48; ruling 1 in 4e72c8a). Stage 1 is not authorized until: the rulings in blocked_by are recorded (OPEN-18, OPEN-19 closed, OPEN-23, OPEN-24, OPEN-26); the viable-regime condition of the procedure is resolved or ruled not to apply (OPEN-25); and reviewer authorization is recorded at a location the TODO names

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

## Stage-1 review conditions and open stack (hand-written; home of each fact named)

- R1 to R6 (AGSDR stage 1): table in `checklist-score.md` ("Review conditions before stage 1"). Status 2026-11-05: R1, R2, R3 PARTIAL; R4 RECORDED; R5 RULED; R6 BLOCKED (viable-regime verdict ruled Q16; still needs selection rule OPEN-18/23, shortfall OPEN-33 (4 feasible vs 8 slots)).
- OPEN-25 evidence 2026-11-05: `docs/receipts/vip_drive_probe/summary.json` (VIABLE_BRANCH_ON_VIP_DRIVE_AXIS), accepted as the viable-regime verdict (Q16); amendment 4 (frozen VIP drive x2.25) in effect; stage 0 under it: `docs/receipts/agsdr_stage0_a4/summary.json` (4 of 24 feasible after H11).
- H07: ruled Q18 (term 4 >= 0.9 x canonical); H07 PASS 24 of 24; final hard-feasible set 4 of 24 under amendment 4 (`docs/receipts/agsdr_h07h11/h07_q18_summary.json`, OPEN-29).
- Problems, blockers, decision queue: `docs/OPEN.md`. Plan: `docs/PROGRESS_REVIEW_2026-11-05.md`. Score: `checklist-score.md`.

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
