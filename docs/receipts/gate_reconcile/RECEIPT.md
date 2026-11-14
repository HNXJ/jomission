# Gate reconciliation receipt (OPEN-47 + OPEN-43 candidate)

Branch `_temp_gate_repair`, base `d1b5d27`. Code and state tip tested: `aab5c4b`. The gate results below are for `aab5c4b`; this file was committed after it (`62389ad`). Not integrated into dev.

Commits `d1b5d27..aab5c4b` (cherry-picked from the pushed `_temp_open47_population_rates` and `_temp_open43_state_reconcile`, so SHAs differ from those originals):
`ea72109` CAMINOS seconds-form fix; `80660a1` OPEN-47 receipts; `245434e` OPEN-43 state reconciliation; `52e5839` OPEN-43 receipt correction; `f6d1d1f` DASHBOARD.html regeneration; `6cbb6c7` gate receipt; `0906e5c` receipt correction; `943ea0a` OPEN-50 placeholders, dashboard regeneration, MEMORY.md compression (reverted in `9934535`); `9934535` lessons.json restored to base, MEMORY.md re-rendered; `65b160d` receipt at 9934535; `aab5c4b` L11 restored in lessons.json, MEMORY budget 7000.

## Environment

- Windows isolated qualification venv `D:\venvs\jomission-qual` (Python 3.14, NOT `--system-site-packages`). Pinned: jax/jaxlib 0.10.1, numpy 2.4.6, scipy 1.18.0, pytest 9.0.2, pytest-cov 7.1.0, matplotlib 3.10.8, plotly 6.6.0, ruff 0.16.2. jaxfne 0.5.8 installed non-editable from `git archive v0.5.8` (11957afa).
- Site packages copied from the shared `C:\Python314\Lib\site-packages`, excluding: stray top-level `tests`, `docs`; jaxfne 0.4.25 and its metadata; jomission metadata; editable hooks; pip. Script: `scratchpad/copy_site_packages.py` (outside the repo).
- Checkout under test selected by a path file `Lib\site-packages\_qual_checkout.pth` (one line: `D:/tmp/combined`). Needed for `test_environment_provenance`, which strips PYTHONPATH and expects jomission to import from a checkout on the path.
- Network is unavailable from this host (pip cannot reach PyPI), so the venv was populated from the shared Python.
- WSL Ubuntu (kernel 6.6, Python 3.12.3), venv `~/jq-venv`: numpy 2.4.6, scipy 1.18.0, jax/jaxlib 0.10.1, pytest 9.0.2; jaxfne 0.5.8 non-editable from `/mnt/d/tmp/jaxfne_v058_src` (`jaxfne.__version__` = 0.5.8). jomission imports from the clone `~/jomission-cand`, detached: at 9934535 (`jomission.__file__` checked) and at `aab5c4b` (fetched from a bundle, because `/mnt/d/tmp/combined` has an alternate object path that WSL cannot resolve; `__file__` not rechecked).

## Reproduction of the collection errors

| Module | Shared venv (`D:\venvs\jomission`) at d1b5d27 | Isolated venv at d1b5d27 |
|---|---|---|
| tests/test_activity_balance_sync.py, _sync2.py, _sync_fixref.py | ImportError: `tests.test_activity_balance` (stray `C:\Python314\Lib\site-packages\tests`) | collects |
| tests/test_v21_pv_microstate.py | ImportError: `scripts` from `D:\repos\_work\jaxfne_v058\scripts` (jaxfne editable install adds its repo root) | collects |
| tests/test_whole_system_ei_native_state.py | ModuleNotFoundError: `scripts.whole_system_ei_native_state_arm` (same `scripts` shadowing) | collects (27 tests in the five modules) |

The target modules exist in the repo at d1b5d27. The errors are import-resolution failures from the shared environment.

## Gate results

| Run | Tier 0 | Tier 2 |
|---|---|---|
| `d1b5d27` baseline, no path file | 5 failed, 253 passed | 10 failed, 650 passed, 3 skipped |
| `02cfc1f` OPEN-47 candidate, no path file | 5 failed, 253 passed | 7 failed, 653 passed, 3 skipped |
| `f6d1d1f` combined candidate, path file set | 2 failed, 256 passed | 4 failed, 656 passed, 3 skipped |
| `943ea0a` (superseded) | rc 0: 258 passed, 5 warnings | 2 failed, 658 passed, 3 skipped |
| `9934535` (superseded) | rc 0: 258 passed, 5 warnings | 2 failed, 658 passed, 3 skipped, 0 collection errors |
| **`aab5c4b` (current)** | **rc 0: 258 passed, 5 warnings** | **2 failed, 658 passed, 3 skipped, 0 collection errors (952 s)** |

The two tier-2 failures at `aab5c4b` are both in `tests/test_allen_v1_units.py`:
- `test_end_to_end_write_inputs_then_verdict_puts_planted_gaba_tone_in_its_unit_only`: `OSError: [WinError 1314] A required privilege is not held by the client` from `os.symlink` (`scripts/allen_v1_units.py:56`), at `e2e.npz`. The test fails before its assertion (`tests/test_allen_v1_units.py:132`, `assert summary["units"] == units and (dst / "e2e.npz").is_symlink()`).
- `test_ampa_vs_gaba_tone_ratio_matches_rws_prediction_and_swapped_order_fails`: same `WinError 1314` at `ampaRun.npz`.

Alternative evidence, not a Windows PASS: WSL Ubuntu, same file, `pytest -q -rs`: 6 passed, 0 skipped at 9934535 (`allen_wsl_9934.log`) and at `aab5c4b` (`aab5c4b_allen_wsl.log`). These two tests are not executed on Windows.

Logs: scratch folder, outside the repo (`aab5c4b_tier0.log`, `aab5c4b_tier2.log`; earlier `cand9934_*`, `cand943_*`, `repair_f6d1d1f_*`, `cand_02cfc1f_*`, `base_d1b5d27_*`; `allen_wsl_9934.log`, `aab5c4b_allen_wsl.log`).

The baseline and OPEN-47 runs had no path file, so `test_environment_provenance` failed in them for environment reasons. With the path file it passes at `d1b5d27`, `f6d1d1f`, `9934535` and `aab5c4b`.

## Classification

| Test | Class | Evidence | Status |
|---|---|---|---|
| tests/test_caminos_e0.py (3) | source defect (API mismatch) | TypeError on `dt_ms` at d1b5d27 | fixed by OPEN-47 change; passes |
| tests/test_environment_provenance.py | environment (no checkout path) | fails without path file; passes with it | repaired in venv |
| tests/test_dashboard_build.py (2) | stale generated artifact | committed page generated from an older `OPEN.md` | repaired by regeneration in `f6d1d1f`; passes |
| tests/test_series_dashboard.py | source content defect | absolute Windows path in the OPEN-50 row, copied into the series dashboard | repaired by placeholder swap in `943ea0a`; passes |
| tests/test_project_memory.py | source content defect (stale file); L11 omission | base `MEMORY.md` 6872 chars, not a render of its lessons (see below) | repaired in `aab5c4b`: passes at 6872 chars, budget 7000, 11 lessons |
| tests/test_allen_v1_units.py (2) | environment limitation (Windows privilege) | `WinError 1314` from `os.symlink` (`scripts/allen_v1_units.py:56`) | NOT repaired on Windows; WSL 6/6 passes at 9934535 and `aab5c4b`; owner decision |
| collection errors (5 tier 2, 2 tier 0) | environment (shared venv shadowing) | reproduced at d1b5d27 in the shared venv; collect in the isolated venv | repaired in venv |

## MEMORY.md and lesson L11 (resolved in `aab5c4b`; plugin budget open)

- Base `MEMORY.md` (d1b5d27, 6872 chars) had an 11th line, `L11-STALE-SYSTEM-JAXFNE`, added by hand in d1b5d27 (that commit changed only MEMORY.md, +1 line). It was not in `lessons.json`.
- `943ea0a` compressed `lessons.json` text and lost unique facts and two scope strings. `9934535` restored `lessons.json` to its d1b5d27 content, which dropped L11 from MEMORY.md (regression).
- `aab5c4b` adds L11 to `lessons.json` as an 11th entry. Its five fields were parsed from the base MEMORY line, and the split was checked to rebuild that line exactly. The first ten entries are unchanged from d1b5d27.
- Owner decision (2026-11-14, option 1): budget 6000 → 7000 in `scripts/render_memory.py:21` and `tests/test_project_memory.py:15`.
- Result: `MEMORY.md` at `aab5c4b` is byte-identical to d1b5d27 (`git diff d1b5d27 -- MEMORY.md` is empty), 6872 chars, 11 lessons, not trimmed; `render_memory.py --check` exits 0.
- **Open, outside this repo:** the global autocompact-memory plugin's project read budget is still 6000 (per the renderer's former docstring). At 6872 chars the injected view would be cut unless the plugin budget is raised to 7000 too.

## Review

- First independent review (verdict REVISE): blockers were (a) L3 and L6 scope strings changed; (b) unique facts lost in the compression; (c) this receipt stale. (a) and (b) are resolved (lessons.json identical to base at 9934535). (c) is resolved by this file.
- Second review of 9934535: items 1–6 PASS (lessons identical to base; render equals `--check`; 10 ids; not trimmed; no changed path in `manifests/sealed_artifacts.json`). It found the L11 omission, now resolved in `aab5c4b`.
- `aab5c4b` was independently reviewed (read-only `reviewer` agent, 2026-11-14): PASS on 8 claims. Covered: the first ten lessons unchanged and the L11 fields rebuilding the base line; MEMORY diff empty and `--check` 0; budget 7000 with no other copy; both memory tests passing; unique ids; `manifests/sealed_artifacts.json` unchanged; the receipt matching the code. Flags: line 3 wording (corrected in the commit after `62389ad`); the reviewer did not rerun tier 0 or tier 2 (counts checked against the logs).
- Fact-diff workflows (wf_ae76678b-d3c, wf_df5d98dc-a1a) checked the compressed text and are superseded: the compression is reverted.

## Gate status

- Tier 0: green at `aab5c4b` (258 passed, rc 0).
- Tier 2: 2 failed (Allen symlink, Windows privilege), 658 passed, 3 skipped at `aab5c4b`.
- MEMORY.md: 11 lessons, 6872 chars, budget 7000 (repo). Plugin budget open.
- Dev integration: blocked until the Allen tests have a Windows PASS or the owner accepts the WSL run as their verification of record.

## Owner decisions needed

1. Windows symlink privilege: enable Developer Mode or elevation (a system setting, owner only), or accept the WSL run (`tests/test_allen_v1_units.py`, 6/6 at 9934535 and `aab5c4b`) as the verification of record for the two Allen tests.
2. Raise the global autocompact-memory plugin's project read budget from 6000 to 7000 (outside this repo).

## Not verified

- Scientific numerical results on real data (no simulation run; the caminos tests use synthetic counts).
- GitHub CI on `_temp_gate_repair` was not read.
- Tier 0 and tier 2 were not run in WSL (only the Allen file, at 9934535 and `aab5c4b`).
- The plugin's read budget and its enforcement were not checked (external).
- Tier 0 and tier 2 counts at `aab5c4b` were not rerun by the reviewer; they are read from `aab5c4b_tier0.log` (258 passed) and `aab5c4b_tier2.log` (2 failed, 658 passed, 3 skipped).
