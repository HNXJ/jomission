# Next goals: optimization and organization, beside the todos

Status: ADOPTED by the owner, 2026-11-02 (chose "All three lanes"). Adds no science and changes no
gate. Each lane names the files it may touch; no two lanes share one.

## Evidence behind the goals (observed)

- Tier-0 tests: 213 passed in 105.5 s (`project_check`, 2026-11-02, commit d7a2c9e).
- `results/`: 922 MB, 419 entries. 17 untracked result files belong to other agents; not touched.
- `manifests/todo.json`: 3 DONE and 8 RETIRED entries remain. Finished items are deleted under the
  stack rule; history stays in git.
- `docs/`: 85 files; three PLAN files (`PLAN_DECISION_TREE`, `PLAN_LONGTERM`, `PLAN_MINIMAL_MODELS`)
  overlap in scope and status.

## Lanes

| Lane | Goal | First step | Check | May touch |
|---|---|---|---|---|
| Optimization | Shorten tier-0 without losing a test | Measure the 15 slowest tests (`--durations`) | Pass count unchanged at 213; new time recorded | `tests/`, `manifests/test_tiers.json` |
| Optimization | Account for the results footprint | Size each `results/` subtree; list entries no manifest cites | Report only; any move waits for approval | none (read-only) |
| Organization | Clear finished todo entries | Remove 3 DONE and 8 RETIRED items | `project_check` todo passes; history in git | `manifests/todo.json` |
| Organization | One home for the three PLAN files | Map which fact each file owns | Each fact has one home; links point to it | `docs/PLAN_*.md` |
| Progress | MIN-07 realization, test first | Failing test on the unpatched build | Receipt committed after the spec; S-13 scored | `jomission/minimal/`, `docs/minimal/` |
| Progress | Recorded owner answers into the files | S-09 restatement; AGENTS.md G: path; PV→VIP note | `project_check` passes; each answer cited | `docs/minimal/connectivity.json`, `AGENTS.md`, `docs/PLAN_MINIMAL_MODELS.md` |

## Approval points (owner)

- Any deletion or move needs your word first. No hard deletes; moves go to `_temp_delete/`.
- Move results only after the footprint report.
- Test moves between tiers: only with a reason logged in `test_tiers.json`.

## Stop conditions

- A test count changes: stop and report.
- A gate fails for a new reason: stop and report.
- A lane needs a file another lane owns: stop and re-split the lanes.
