# Gate reconciliation receipt (OPEN-47 + OPEN-43 candidate)

Branch `_temp_gate_repair`, base `d1b5d27`. Commits: caminos seconds-form fix, OPEN-47 receipts, OPEN-43 state reconciliation, OPEN-43 receipt correction (all cherry-picked from the pushed branches, so SHAs differ from the originals), and the regenerated `docs/DASHBOARD.html`. Candidate tip at the gate run: `f6d1d1f`. Not integrated into dev.

## Environment

- Isolated qualification venv `D:\venvs\jomission-qual` (Python 3.14, NOT `--system-site-packages`). Pinned: jax/jaxlib 0.10.1, numpy 2.4.6, scipy 1.18.0, pytest 9.0.2, pytest-cov 7.1.0, matplotlib 3.10.8, plotly 6.6.0, ruff 0.16.2. jaxfne 0.5.8 installed non-editable from `git archive v0.5.8` (11957afa).
- Site packages copied from the shared `C:\Python314\Lib\site-packages`, excluding: stray top-level `tests`, `docs`; jaxfne 0.4.25 and its metadata; jomission metadata; editable hooks; pip. Script: `scratchpad/copy_site_packages.py` (outside the repo).
- Checkout under test selected by a path file `Lib\site-packages\_qual_checkout.pth` (one line, the checkout). Needed for `test_environment_provenance`, which strips PYTHONPATH and expects jomission to import from a checkout on the path.
- Network is unavailable from this host (pip cannot reach PyPI), so the venv was populated from the shared Python.

## Reproduction of the collection errors

| Module | Shared venv (`D:\venvs\jomission`) at d1b5d27 | Isolated venv at d1b5d27 |
|---|---|---|
| tests/test_activity_balance_sync.py, _sync2.py, _sync_fixref.py | ImportError: `tests.test_activity_balance` (stray `C:\Python314\Lib\site-packages\tests`) | collects |
| tests/test_v21_pv_microstate.py | ImportError: `scripts` from `D:\repos\_work\jaxfne_v058\scripts` (jaxfne editable install adds its repo root) | collects |
| tests/test_whole_system_ei_native_state.py | ModuleNotFoundError: `scripts.whole_system_ei_native_state_arm` (same `scripts` shadowing) | collects (27 tests in the five modules) |

The target modules exist in the repo at d1b5d27. The errors are import-resolution failures from the shared environment.

## Gate results (isolated venv)

| Run | Tier 0 | Tier 2 |
|---|---|---|
| `d1b5d27` baseline, no path file | 5 failed, 253 passed | 10 failed, 650 passed, 3 skipped |
| `02cfc1f` OPEN-47 candidate, no path file | 5 failed, 253 passed | 7 failed, 653 passed, 3 skipped |
| `f6d1d1f` combined candidate, path file set | 2 failed, 256 passed | 4 failed, 656 passed, 3 skipped, 0 collection errors |

Logs: `/d/tmp/qual_logs/{base_d1b5d27,cand_02cfc1f,repair_f6d1d1f}_tier{0,2}.log` (outside the repo).

The baseline and OPEN-47 runs had no path file, so `test_environment_provenance` failed in them for environment reasons. With the path file, that test passes at `d1b5d27` (2 passed) and at `f6d1d1f` (2 passed).

## Classification

| Test | Class | Evidence | Status |
|---|---|---|---|
| tests/test_caminos_e0.py (3) | source defect (API mismatch) | TypeError on `dt_ms` at d1b5d27 | fixed by OPEN-47 change; passes |
| tests/test_environment_provenance.py | environment (no checkout path) | fails without path file; passes with it | repaired in venv |
| tests/test_dashboard_build.py (2) | stale generated artifact | committed page was generated from an older `OPEN.md`; see "Dashboard regeneration diff" below | repaired by regeneration in `f6d1d1f`; passes |

## Dashboard regeneration diff (`f6d1d1f`)

The regenerated `docs/DASHBOARD.html` differs from the previous committed page in more than the row count. All changes come from the current `docs/OPEN.md`:
- Heading: "all 40 rows" becomes "all 45 rows".
- OPEN-44 status: `closed` becomes `closed (2026-11-11)`.
- OPEN-45 status: `closed` becomes `open (reopened 2026-11-11)`. The first commit message did not mention this.
- Rows OPEN-46 to OPEN-50 are added.

The page is generated, not edited by hand, and the change is consistent with `docs/OPEN.md`. The earlier wording ("40 rows and OPEN-44 closed") described only part of the change.

Commits above `d1b5d27` at `f6d1d1f`: five (`ea72109`, `80660a1`, `245434e`, `52e5839`, `f6d1d1f`).
| tests/test_allen_v1_units.py (2) | environment limitation (Windows privilege) | `WinError 1314` from `os.symlink` (`scripts/allen_v1_units.py:56`) | NOT repaired: needs Windows Developer Mode or elevation (system setting), or an owner-authorized alternative |
| tests/test_project_memory.py | source content defect | `MEMORY.md` 6872 characters, budget 6000 | NOT repaired: owner decision on what to trim |
| tests/test_series_dashboard.py | source content defect | absolute Windows path in text copied from `docs/OPEN.md` (row OPEN-50) into the series dashboard | NOT repaired: owner decision on rewording or builder redaction |
| collection errors (5 tier 2, 2 tier 0) | environment (shared venv shadowing) | reproduced at d1b5d27 in the shared venv; collect in the isolated venv | repaired in venv |

## Gate status

Integration to dev is blocked. The combined candidate still fails:
- tier 0: `test_project_memory` and `test_series_dashboard` (content defects).
- tier 2: those two plus `test_allen_v1_units` (2) (environment limitation).

Owner decisions needed:
1. Windows symlink privilege on this host (enable Developer Mode, or accept an alternative verification for the two Allen tests).
2. `MEMORY.md` budget: approve a trimming, or raise the budget with the owner's ruling.
3. The OPEN-50 row wording (absolute paths), or a builder rule that redacts them.

## Not verified

- Scientific numerical results on real data (no simulation run; the caminos tests use synthetic counts).
- Independent review of this classification: a reviewer agent was started at the time of writing; its verdict is not recorded here.
