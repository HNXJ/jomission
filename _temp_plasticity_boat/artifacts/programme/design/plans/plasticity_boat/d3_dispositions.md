# D3 dispositions: Plasticity Boat memory-guided DMS spec v1

Recorded 2026-11-21. Status: RATIFIED by the owner 2026-11-21 (D3.1–D3.6, with the refinements below). Qualifications apply.

| Item | Disposition | Qualification | Implemented in |
|---|---|---|---|
| D3.1 t_END | Accept | Early arrival terminates the trial. Fixed-deadline metrics are secondary and defined separately. | `dms_task.py` `_check_close`, `_close` |
| D3.2 Feasibility | Accept with refinement | Analytic bound at GO onset is necessary: \|q - p\| <= g * max(\|v\|, Fmax/beta). Oracle success shows demonstrated reachability only, not sufficiency for all feasible targets. | `_feasibility`; T07 |
| D3.3 Seed 2026 | Accept | Keep sealed until the authorised lock. Pre-register separate DMS development and validation seeds (not yet named). | not implemented; no seed used by the backend |
| D3.4 Calibration | Accept with refinement | Radius and GO duration calibrated from dynamics. Delay levels chosen independently of the final test set. A no-memory control is evaluated, but its failure alone does not set the final delay. | radius, timings and max_force are provisional config values |
| D3.5 Reconstruction | Accept | Remembered cue plus proprioception permitted. Cue memory, path integration and recurrent neural memory distinguished experimentally. | `proprioception` flag; experiments pending |
| D3.6 Memory oracle | Accept | Explicit-memory register is the positive control. LIF recurrence evaluated independently. | `b2_memory_controller.py` (register controller, not learned); T11, T12 |

## Acceptance test status (`code/test_dms_task.py`)

Run from `code/`: `python3 -m pytest -q test_dms_task.py`.

| Test | Result | Note |
|---|---|---|
| T01 schedule, grid, overlap rejection | pass | |
| T02 cue visible only in CUE | pass | |
| T03 no leakage after cue | pass | Known positive: targets differ in CUE |
| T04 motor gated outside GO | pass | Momentum after GO differs legitimately; pre-GO trajectory compared |
| T05 state continuity | pass | Recurrence on v and p across all phases |
| T06 deadline inclusive and strict | pass | Oracle arrival step L; deadline L passes, L-1 fails |
| T07 feasibility bound and oracle | pass | Near target feasible and reached; far target flagged and not reached |
| T08 determinism | pass | |
| T09 plant equals legacy `sailor_step` | pass with `PYTHONPATH` to `jaxfne-crazy-sailor`; SKIPPED with the default installed jaxfne (no `SailorConfig`) | Installed jaxfne lacks the sailor API, as in the V0 pinned failure |
| T10 no leakage between trials | pass | Asserts trial 2 `min_dist` is not inherited from trial 1 (reviewer finding) |
| T11 cue dependence | pass | Intact register solves all 3 feasible targets; absent and shuffled cues fail (known positive) |
| T12 known-memory oracle across delays | pass | Register holds the target from CUE through GO for delays 0, 1, 2, 4 s |

Counts: default environment 13 passed, 1 skipped (T09); with `PYTHONPATH=/Users/hamednejat/workspace/dev/jaxfne-crazy-sailor` 14 passed.

Mutation checks (scratch copies): gate forced on fails T04; cue masking removed fails T02 and T03; `_min` reset removed fails T10 (added after review).

Tier-0 (`manifests/test_tiers.json` tier 0, 31 files) at dfbcc2d: 258 passed, 0 failed. Run in a clone under `suite_lock.py`. The global interpreter run gave 1 failure, `test_jomission_import_resolves_to_this_checkout`: the installed editable `jomission` points at `_temp_fixation_boat_skill`. That is an environment failure (ENV-STALE-EDITABLE), not a model result. The gate ran in a scratch venv whose `.pth` points at the clone.

Independent review (reviewer agent, non-writer, 2026-11-21): ACCEPT. Minor findings: T10 leak gap (fixed above); T05 known-positive is near trivial (`np.any(V[100:])`), neural-state continuity not testable until a controller is connected; feasibility bound is loose (valid as necessary condition, not tight; probe max reach ~8.9 vs bound 10 at target (8, 8)).

## Not done

- Seeds for DMS development and validation not named.
- Radius and timings not calibrated.
- T05 known-positive strengthening (minor, open).
- Feasibility bound tightness: documentation only (minor, open).
