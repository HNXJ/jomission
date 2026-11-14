# OPEN-43 state reconciliation receipt

Branch `_temp_open43_state_reconcile`, from `d1b5d27` (origin/dev and origin/main tip). Touches only `manifests/current_state.json`: two fields, two lines.

## Changes

| Field | Before | After | Basis |
|---|---|---|---|
| `updated` | `2026-11-05` | `2026-11-14` | the OPEN register, the checklist interim and the handoff are dated 2026-11-11 or later |
| `next_authorized_task_note` | starts `2026-11-05 seal:` | prepends a `2026-11-14 reconciliation (OPEN-43)` sentence; the 2026-11-05 text is kept | see the four facts below |

The prepended sentence records four facts, each checked on disk:
1. The pointer `next_authorized_task` (SCI-AGSDR-CALIBRATION) is unchanged and awaits Hamm's ruling. The todo item exists and is OPEN.
2. The active preprint lineage is SCI-LAMINAR-MOTIF-1 (LANSA). `docs/laminar_motif/HANDOUT_CURSOR_LANSA_100.md` exists.
3. The frozen execution baseline is jaxfne 0.5.8 (11957afa). `results/jaxfne_058_migration.json` exists and records the promotion.
4. The laminar records `results/laminar_e0/e0_result.json` and `docs/laminar_motif/E0_PILOT_RECEIPT.json` record jaxfne 0.5.9. This is a non-authority lane whose outputs are not comparable to 0.5.8 without a migration (OPEN-37).

Not changed: `next_authorized_task`, `current_program`, `jaxfne_version`, `execution_authority`, `current_gate`, `latest_verdict`, `locked_gates`, `retired_axes`, any scientific status, any acceptance criterion, any sealed artifact, `results/`, `docs/OPEN.md`, `docs/HANDOFF_CURRENT.md`.

## Checks

| Check | Command | Result |
|---|---|---|
| JSON parses | `json.load` on the file | ok |
| Handoff not stale | `python -m jomission.harness.handoff --check` on the clone (import confirmed as the clone) | rc 0 |
| Cited files exist | file tests on the four basis paths | all four present |
| Tier 0 vs baseline | tier-0 list on this branch, compared with the same list on `6092f75` (OPEN-47 branch, same base) | 240 passed, 5 failed, 2 collection errors. Identical failing set to the baseline at `d1b5d27` (environment and path issues, see `docs/receipts/open47/VERIFICATION.md`). No new failures |

## Not verified

- `project_check.py` on this clone: the shared venv's editable install points at `D:\repos\_work\jomission`, so the check fails for the clone path. This is the same environment mismatch as the clean-clone run, not a state defect.
- The tests do not read `updated` or `next_authorized_task_note` (grep of `tests/test_harness_state.py`, `tests/test_environment_provenance.py`).
