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
- stopped because: owner assessment 2026-09-19 accepted the terminal state at 100/100 and ordered the scientific program: SCI-EI-INTERACTION-1 (authorized 2x2 factorial, next), then SCI-AGSDR-CALIBRATION (blocked on a viable-regime verdict plus design authorization), then SCI-PROP-QUALIFIED-1 (locked until the plant is defined and the design is authorized). The propagation lineage stands closed as PROPAGATION_DETECTED_ORDERED with no stronger interpretation. SCI-EI-INTERACTION-1 has since sealed EI_REGIME_STILL_UNRESOLVED: no cell viable across the factorial, so the silence mechanism lies outside mean weight scale and adaptation magnitude. Owner assessment 2026-09-19 orders measurement before repair: SCI-EI-RECRUITMENT-1 has since sealed EI_RECRUITMENT_UNRESOLVED (PV/VIP match no predeclared cause at 1 ms group resolution; mean-field split withheld by its residual gate); AGSDR stays BLOCKED; SCI-PROP-QUALIFIED-1 stays LOCKED. Owner order 2026-09-21 opens JAXFNE-0425-MIGRATION: availability verified the same day (0.4.25 on index, wheel fetched, read-only delta shows HDP-kernel-only changes outside sealed baseline evidence). Execution authority is 0.4.25 since the sealed migration. Owner assessment 2026-09-21 accepts the 0.4.25 migration and the UNRESOLVED recruitment, keeps AGSDR BLOCKED and PROP-QUALIFIED-1 LOCKED, and opens SCI-EI-NATIVE-STATE-1 (authorized, next): native state geometry with five predeclared verdicts and no modifications; if it cannot distinguish PV/VIP from SST, the substrate direction stops rather than manufacturing repair axes. No other spec is written

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

- SCI-PROP-TRAJECTORY: V1_1 detects in the test pair with peak per-bin divergence 72.32 against the null-derived threshold 1e-12 — `be18f0d:results/propagation_trajectory.json`
- SCI-PROP-TRAJECTORY: every downstream g=0 null reads at most the threshold in both the test-null and the held-out replicate pair — `be18f0d:results/propagation_trajectory.json`
- SCI-PROP-TRAJECTORY: four downstream areas detect at g=0.5 with sustained onsets: V1_2 11100, V4_1 11360, V4_2 12020, PFC 12700 ms; FEF detects by peak without a sustained onset and is unordered — `be18f0d:results/propagation_trajectory.json`
- SCI-PROP-TRAJECTORY: the 1 s window-count contrast that read exactly 0.000000 everywhere, including the directly driven V1_1, is superseded for propagation use by the 10 ms trajectory divergence; the count finding stands as sealed and is not retracted — `be18f0d:results/propagation_trajectory.json`
- SCI-EI-INTERACTION-1: baseline cell exactly rebuilds the EI diagnosis (E 10.6154, PV 0.0621, SST 50.1991, VIP 0.0466 Hz) — `d846eff:results/whole_system_ei_interaction.json`
- SCI-EI-INTERACTION-1: weight-only cell leaves PV 0.0621 and VIP 0.0466 Hz unchanged against 18 percent stronger E->I weights; SST 50.2112 Hz — `d846eff:results/whole_system_ei_interaction.json`
- SCI-EI-INTERACTION-1: adaptation-only cell halves SST 50.2 to 29.2496 Hz with PV 0.0621 and VIP 0.0466 Hz unchanged — `d846eff:results/whole_system_ei_interaction.json`
- SCI-EI-INTERACTION-1: both-repaired cell reads PV 0.0621, VIP 0.0466, SST 29.2504 Hz: no cell viable, verdict EI_REGIME_STILL_UNRESOLVED — `d846eff:results/whole_system_ei_interaction.json`
- SCI-EI-INTERACTION-1: departures from additivity per class are recorded in the decomposition block; synchrony and the 94-95 ms mode reported per cell, not gating — `d846eff:results/whole_system_ei_interaction.json`
- SCI-EI-INTERACTION-1: V0 schematic and V1 initial raster of the baseline construction viewed before the runs; V2 reuses V1 ordering and window; V3 atlas 5 of 6 panels — `d846eff:results/whole_system_ei_interaction.json`
- SCI-EI-RECRUITMENT-1: PV rate 0.0596 Hz with total drive 2.9939 against fresh rheobase 4.0; reconstructed split withheld by its residual gate — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-RECRUITMENT-1: VIP rate 0.0488 Hz with total drive 2.9933 against fresh rheobase 24.0; reconstructed split withheld by its residual gate — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-RECRUITMENT-1: the mean-field split disagrees with retained-minus-tonic in sign and 20x magnitude, so DEFICIT and VETO readings are withheld as unreliable per the predeclared residual rule — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-RECRUITMENT-1: neither class crosses its rheobase transiently (PV max 3.1869, VIP max 3.1905), so no STATE pattern; both cause UNRESOLVED — `dd8d6c4:results/whole_system_ei_recruitment.json`
- SCI-EI-RECRUITMENT-1: the current-budget atlas aligns per-class rate, total current with tonic and rheobase references, reconstructed splits, and v/u state over the full run — `dd8d6c4:results/whole_system_ei_recruitment.json`

### Next authorized task

- `SCI-EI-NATIVE-STATE-1` (OPEN): determine what state-space event permits SST to spike repeatedly while PV/VIP almost never cross their spike/reset boundary under the same realized circuit, by native-resolution phase-plane measurement with no parameter changes
- stop: measurement only. One evaluation (or one authorized retention run), then STOP; no parameter is changed, nothing is calibrated, and no repair axis is manufactured from the result

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
