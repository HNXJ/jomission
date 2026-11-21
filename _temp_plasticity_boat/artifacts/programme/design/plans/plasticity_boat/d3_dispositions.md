# D3 dispositions: Plasticity Boat memory-guided DMS spec v1

Recorded 2026-11-21. Status: proposed for adoption from the review summary; NOT yet ratified by the owner. Each item becomes a ruling only when the owner confirms it.

| Item | Disposition | Qualification | Implemented in |
|---|---|---|---|
| D3.1 t_END | Accept | Early arrival terminates the trial. Fixed-deadline metrics are secondary and defined separately. | `dms_task.py` `_check_close`, `_close` |
| D3.2 Feasibility | Accept with refinement | Analytic bound at GO onset is necessary: \|q - p\| <= g * max(\|v\|, Fmax/beta). Oracle success shows demonstrated reachability only, not sufficiency for all feasible targets. | `_feasibility`; T07 |
| D3.3 Seed 2026 | Accept | Keep sealed until the authorised lock. Pre-register separate DMS development and validation seeds (not yet named). | not implemented; no seed used by the backend |
| D3.4 Calibration | Accept with refinement | Radius and GO duration calibrated from dynamics. Delay levels chosen independently of the final test set. A no-memory control is evaluated, but its failure alone does not set the final delay. | radius, timings and max_force are provisional config values |
| D3.5 Reconstruction | Accept | Remembered cue plus proprioception permitted. Cue memory, path integration and recurrent neural memory distinguished experimentally. | `proprioception` flag; experiments pending |
| D3.6 Memory oracle | Accept | Explicit-memory register is the positive control. LIF recurrence evaluated independently. | B2, not built; T11 and T12 skipped |

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
| T10 no leakage between trials | pass | |
| T11 cue dependence | NOT RUN | Needs B2 comparator |
| T12 known-memory oracle across delays | NOT RUN | Needs B2 comparator |

Mutation checks (scratch copy): gate forced on fails T04; cue masking removed fails T02 and T03. The tests detect both breaks.

## Not done

- Tier-0 suite not run: the suite lock is denied by the sandbox. Push deferred until it runs.
- Seeds for DMS development and validation not named.
- Radius and timings not calibrated.
