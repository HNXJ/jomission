# D3 dispositions: Plasticity Boat memory-guided DMS spec v1

Recorded 2026-11-21. Status: RATIFIED by the owner 2026-11-21 (D3.1–D3.6, with the refinements below). Qualifications apply.

Status pointer: `memory_dms_spec_v1.md` (its status lines 3 and 48) predates the backend. The current state of the backend, tests and reviews is in this file.

| Item | Disposition | Qualification | Implemented in |
|---|---|---|---|
| D3.1 t_END | Accept | Early arrival terminates the trial. Fixed-deadline metrics are secondary and defined separately. | `dms_task.py` `_check_close`, `_close` |
| D3.2 Feasibility | Accept with refinement | Analytic bound at GO onset is necessary: \|q - p\| <= g * max(\|v\|, sqrt(2) Fmax/beta). The sqrt(2) factor follows the componentwise clip (owner ruling 2026-11-22, option a; plant unchanged). Oracle success shows demonstrated reachability only, not sufficiency for all feasible targets. | `_feasibility` (sqrt(2) bound at GO onset); `schedule()` guard `damping*dt <= 1`; T07, T07b, T07c; T01 unstable and nominal-1 damping tests |
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

Counts (2026-11-22, measured at the commit that adds `test_T01_accepts_nominal_unit_damping_step`): default environment 18 passed, 1 skipped (T09); with `PYTHONPATH=/Users/hamednejat/workspace/dev/jaxfne-crazy-sailor` 19 passed. Earlier: 15/1 and 16 at e144a7a.

Mutation checks (scratch copies, 2026-11-21): gate forced on fails T04 and T06; cue visible after CUE fails T02, T03 and T10; `_min` reset removed fails T10; the intact latch ignoring p_onset fails the new T12 `register_exact_at_moving_onset`. Euclidean-bound mutant (2026-11-22, `fm/beta` in `_feasibility`): fails T07 and T07b.

Tier-0 (`manifests/test_tiers.json` tier 0, 33 files) at e144a7a (2026-11-22, suite_lock, background): 258 passed, 0 failed, rc 0. On the integration candidate (dev 1b5f4f0 + boat + subproject docs): 261 passed, 0 failed, rc 0 (session report; not recorded here, since this file cannot cite its own commit). At dfbcc2d and f8b1654: 258 passed, 0 failed, rc 0. Run in a clone under `suite_lock.py` with a scratch venv whose jomission resolves to the clone. The gate does not execute `test_dms_task.py`; DMS evidence is the DMS test run alone. The global interpreter run gave 1 failure, `test_jomission_import_resolves_to_this_checkout`: the installed editable `jomission` points at `_temp_fixation_boat_skill`. That is an environment failure (ENV-STALE-EDITABLE), not a model result. The gate ran in a scratch venv whose `.pth` points at the clone.

Independent review of dfbcc2d (2026-11-21): ACCEPT. Minor: T05 known-positive is near trivial (`np.any(V[100:])`); neural-state continuity not testable until a controller is connected.

Independent review of f8b1654 (B2, T10; workflow `wf_9ad0947c-63d`, 2026-11-21): REVISE.
- major: intact latch p_onset term untested. FIXED: T12 `register_exact_at_moving_onset`.
- major, RESOLVED 2026-11-22 (owner ruling option a: bound uses sqrt(2) fm; T07 updated, T07b added): the feasibility bound `g*max(|v|, fm/beta)` assumed Euclidean force magnitude, but the plant clips each component to [-fm, fm]. Diagonal reach can reach sqrt(2)*fm, so the bound is not necessary under the implemented plant (scratch check: diagonal target at D=21, go 10 s, bound 20.0, oracle min_dist 0.9531 implies a GO position at |p| >= 20.047). Latent for axis-aligned T07 cases; affects calibration. Options: (a) bound uses sqrt(2)*fm, (b) clip the force by norm, F*min(1, fm/|F|), which changes the plant semantics.
- minor, FIXED: T10 comment ("~0 at arrival" is in (0, radius]); T11 comment; mutation list completed; tier-0 file count 33.
- minor, OPEN: the B2 docstring says proprioception is required, but nothing enforces it.
- minor, OPEN: the first-CUE latch rule is only partly exercised by T12; a latch-every-CUE-step mutant still survives.

Independent review of e144a7a (feasibility sqrt(2) bound, T07b; reviewer agent, 2026-11-22): ACCEPT. Bound valid for the discrete plant (invariant |v| <= M, displacement per step <= dt·M). Mutant fails T07 and T07b. Minors: T07 validity is checked only through the exact bound value (open); T07b covers only from-rest diagonal, and the |v0| > sqrt(2) fm/beta branch is RESOLVED by T07c (74e18e6); T07 comment "from rest" is FIXED (74e18e6).

Reviews after e144a7a: 74e18e6 (T07c, T07 comment): reviewer ACCEPT for the bound and tests, minors only. 0efc0ec (damping*dt guard): reviewer ACCEPT. 3b58c74 (float slack) and the nominal-1 test: not reviewed separately; covered by the round-2 review of the integrated candidate (session report).

## Not done

- Feasibility: done (sqrt(2) bound, T07b). Norm clip (option b) not adopted; a plant change needs a new ruling.
- B2 proprioception guard; latch-every-step test.
- Seeds for DMS development and validation not named.
- Radius and timings not calibrated.
- T05 known-positive strengthening (minor, open).
