# OPEN-47 verification receipt

Branch `_temp_open47_population_rates`. Code commit `6092f75` (parent `d1b5d27`, the origin/dev and origin/main tip). Touches only `jomission/laminar/caminos_analysis.py`: one call, 3 insertions, 2 deletions.

Change: `population_rates(arr, sizes=..., pops=..., dt_ms=dt*1000, discard_ms=DISCARD_S*1000)` becomes `population_rates(arr, dt=dt, sizes=..., pops=..., discard_s=DISCARD_S)`. The seconds form exists on JaxFNE 0.5.8 (the pinned baseline, 11957afa) and on 0.5.9 (deprecated there; DeprecationWarning only).

No engine change. No sealed result, manifest or gate touched. OPEN-37 stays open.

## Checks

| # | Check | Command (interpreter `D:\venvs\jomission\Scripts\python.exe`, clone with `PYTHONPATH` set to the clone) | Result |
|---|---|---|---|
| 1 | Red before the change | `pytest tests/test_caminos_e0.py tests/test_e1_gates.py` at `d1b5d27` | 3 failed (`TypeError: population_rates() got an unexpected keyword argument 'dt_ms'`), 8 passed |
| 2 | Green after the change | `pytest tests/test_caminos_e0.py` at `6092f75` (run before the final comment-only edit; rerun below) | 7 passed |
| 3 | Final tree | same command at `6092f75` (the pushed commit) | 7 passed in 18.4 s |
| 4 | Tier 0 (`manifests/test_tiers.json`, tier-0 list) | at `6092f75` | 240 passed, 5 failed, 2 collection errors. Failures are the pre-existing set in check 6 |
| 5 | Tier 2 (tier-2 command, collection errors reported) | at `6092f75` | 626 passed, 7 failed, 3 skipped, 5 collection errors (832 s). Failing set identical to check 6 |
| 6 | Baseline at `d1b5d27`, targeted | `pytest` on `test_caminos_e0`, `test_allen_v1_units`, `test_dashboard_build`, `test_environment_provenance`, `test_project_memory`, `test_series_dashboard` | 10 failed, 20 passed: the 3 caminos TypeErrors plus the same 7 failures as check 5 |
| 7 | Numeric equivalence (synthetic Poisson counts, 3 seeds, dt = 1e-4 s, discard 2 s, 12 s runs) | `docs/receipts/open47/poprates_equiv.py` run on each engine, then `compare` | 0.5.8 seconds form = 0.5.9 seconds form = 0.5.9 ms form for all 18 values, worst relative difference 0. Both raise ValueError on a run shorter than the discard. 0.5.9 seconds form warns DeprecationWarning |
| 8 | Independent review (reviewer agent, read-only) | source read of both `population_rates`, call sites, pytest config | PASS. Units and discard identical; no warning-to-error config; only caller is this line; commit touches one file. Flagged `docs/laminar_motif/E0_SEAL.md:56` as stale (sealed text, not edited) |

## What the failures are

Check 6 shows the 7 non-caminos failures are present at `d1b5d27` without the patch. They are environment or path issues, not caused by the change:
- `test_jomission_import_resolves_to_this_checkout`: the shared venv's editable install points at `D:\repos\_work\jomission`, not at the scratch clone.
- `test_allen_v1_units` (2): `WinError 1314`, Windows privilege needed to create the link.
- `test_dashboard_build` (2), `test_series_dashboard` (1): generated output differs or contains an absolute path (`C:\`); depends on the checkout location.
- `test_project_memory` (1): `MEMORY.md` is 6872 characters against a 6000-character budget (existing content).
- Tier-2 collection errors (5): `tests.test_activity_balance*` and `scripts.*` import names collide with the `tests` and `scripts` packages in the shared editable jaxfne tree (`D:\repos\_work\jaxfne_v058`). The patch does not touch these modules, and grep shows none of them imports `caminos_analysis`.

Derived, not observed: the tier-2 total at `d1b5d27` would be 623 passed and 10 failed (626 + 3 and 7 + 3). This was not run in full; check 6 covers the affected tests.

## Gate status

- Remote checkpoint (this branch): allowed. Targeted gate (`test_caminos_e0.py`) passes at the pushed SHA; the failing rest is pre-existing and listed above.
- dev integration: NOT ready. Full tier 2 shows 7 failures and 5 collection errors, all pre-existing (not introduced by this change). Hamm's rule is that native gates pass before dev. Integration waits for those to be fixed or ruled.

## Open items (not fixed here)

- OPEN-37: 0.5.9 changes VIP/I dynamics and default-route u0; keep 0.5.8 as the baseline until Hamm rules.
- `docs/laminar_motif/E0_SEAL.md:56` describes the old `dt_ms` call. Sealed text; Hamm's amendment.
- Private-API coupling in scripts and the `drive_bias` unit gap (peer audit, read-only): not yet recorded in OPEN.
