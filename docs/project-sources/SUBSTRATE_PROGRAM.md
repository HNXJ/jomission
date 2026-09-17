# Substrate program model (durable)

Owner of: program objective, retired axes, active invariants. Mutable state
lives in `manifests/current_state.json`; executable work in `manifests/todo.json`;
evidence in `results/`. IDs in backticks are referenced by the current-state
validator.

## Objective

```text
generic cortical operation -> functional recurrence -> zero-tonic active basin
  -> propagation -> T_R -> functional hierarchy -> Gen-2 freeze -> blind omission
```

Omission stays downstream: Gen-2 may not use omission-success outcomes to build
the substrate. Physiology is asked after the substrate passes, never tuned toward
before.

## Retired axes (do not reopen without new authority)

| ID | question | verdict | evidence |
|---|---|---|---|
| `RECURRENT_FAMILIES_7` | scalar/motif/slow-I/slow-E/delay/modular/tonic-transfer recurrence | falsified: closed == cut, basin dominance | results/gen2_seal.json |
| `GLOBAL_HDP_SCALE` | does global HDP amplitude (s=2..36) fix geometry? | SCALE_INSUFFICIENT | branch hdp-efficacy-scale |
| `NATIVE_BASIN_S9` | finite active basin on the native manifold? | ACTIVE_BASIN_FAIL | branch s9-basin |
| `BASIN_ACCESS_B` | tiny basin vs off-manifold? | OFF_NATIVE_MANIFOLD | branch b-basin-access |
| `NATIVE_GEOMETRY_N1_N4` | intermediate native region? | NO_INTERMEDIATE | branch n-native-geometry |
| `MICROSTATE_N5` | causal microstate owner of fate? | NONE_IDENTIFIED | branch n5-microstate |
| `CMIN_FAMILY` | repair C-min further? | STOOD_DOWN | results/cmin_family_seal.json |
| `PROPAGATION_OLD_PLANT` | FF/FB transmission on the old plant | STRUCTURAL_ONLY, stale after plant change | results/propagation.json |
| `DURATION_SCRATCH` | sustained drive scratch runs | VOID: tonic duplicated in schedule | results/duration_postmortem.json |
| `SHOT_NOISE_BRACKETS` | further private shot-noise brackets on V2.1 | V2.1b FAIL; no new noise brackets | results/v21b_lineage.json |
| `INTRINSIC_SINGLE_PARAM_E` | one zero-mean E intrinsic dispersion (a, b, c, or d) for rate heterogeneity | INFEASIBLE: a/d need w 0.7, c bursts, b makes E pacemakers at w ~0.09 before rate-CV 0.3 at w ~0.235 | results/v21c_design.json, results/v21c_boundary.json |
| `V21_BACKGROUND_INPUT_SEARCH` | further tonic, lambda, duration, initial-state or noise-amplitude searches for a V2.1 operating background | STOPPED: identical tonic at lambda 0.65625 gives history-dependent PV (late 14.43 / 0.71 / 0.09 Hz), PV_NONSTATIONARY; reopen only with a lineage that changes or explains the PV state-dependence mechanism | results/v21_coupled_background.json, results/v21_pv_stationarity.json |

Kept as qualified reductions (not retired): selectivity inverse, selective native
HDP (S6), coarse geometry S7 and linear stability S8 as reductions only.

## Active invariants

- `ONE_DELTA`: one principal delta per lineage edge; exceptions are recorded
  (results/lineage_notes.json).
- `PRE_EXECUTION_SEAL`: brackets, rules, and criteria are committed before the run
  that uses them (manifests/lineage_registry.json, tests/test_harness_guards.py).
- `NO_RETUNE_AFTER_RESULTS`: sealed negatives stay negative; criteria and parameters
  do not move after outputs are seen.
- `DRIVE_ADDITIVITY`: executed input = emitter tonic + schedule; a schedule never
  carries the tonic (jomission/harness/drive.py).
- `ESTIMATOR_METADATA`: every transient statistic carries bin width, aggregation,
  baseline, sign (jomission/harness/validate.py validate_estimator).
- `CONFIGURED_REALIZED_EXECUTED_EFFECTIVE`: a qualified mechanism has evidence for
  each stage (lineage `qualification`).
- `OMISSION_FIREWALL`: no omission outcome informs substrate construction; Pages
  prohibited strings in site-src/manifest.json.
- `SEALED_ARTIFACT_IMMUTABILITY`: registered sealed files never change
  (manifests/sealed_artifacts.json).
- `WORKSPACE_PROVENANCE`: jomission imports from this checkout; JaxFNE equals the
  execution authority (scripts/project_check.py).
- `NATIVE_NOISE_DEFAULT`: `compile_step_fn(kernel="baseline")` without `noise_scale`
  adds 0.5*N(0,1) per neuron per step; "deterministic" requires `noise_scale=0`
  (results/v21_provenance_correction.json).

## Interpretation rules

Population irregularity does not imply per-cell ISI irregularity; a pooled ISI
fraction does not imply E irregularity; temporal irregularity does not imply rate
heterogeneity; static fixed point does not imply native basin; current magnitude
does not imply causal authority; structural connectivity does not imply functional
propagation. F-table interpolation overestimates near threshold jumps: use exact
assays for tonic derivation. Cross-process float32 floor ~1e-4 on active-motif w/I
means (rates exact).
