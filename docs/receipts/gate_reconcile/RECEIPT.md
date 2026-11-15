# Gate receipt: OPEN-47 + OPEN-43 candidate (integration)

Branch `_temp_gate_repair`, base `d1b5d27` (= `origin/dev` before integration).
Candidate code tip: `fb6e288`. Gate results below are for `fb6e288` unless marked otherwise.
This file was committed after the candidate (`62389ad`, `258d118`, then the integration commit); those receipt-only commits change this file and nothing else (independent review, see below).

## Commits `d1b5d27..fb6e288`

`ea72109` CAMINOS seconds-form fix; `80660a1` OPEN-47 receipts; `245434e` OPEN-43 state reconciliation; `52e5839` OPEN-43 receipt correction; `f6d1d1f` DASHBOARD.html regeneration; `6cbb6c7` gate receipt; `0906e5c` receipt correction; `943ea0a` OPEN-50 placeholders, dashboard regeneration, MEMORY.md compression (reverted in `9934535`); `9934535` lessons.json restored to base; `65b160d` receipt; `aab5c4b` L11 restored in lessons.json; `62389ad` receipt; `258d118` receipt wording; `fb6e288` MEMORY.md compact view, renderer fails over budget instead of trimming.

Cherry-picked from the pushed `_temp_open47_population_rates` and `_temp_open43_state_reconcile`, so SHAs differ from those originals.

## Verdict (platform-qualified)

| Gate | Verdict | Evidence |
|---|---|---|
| Tier 0, full list, qualified checkout `D:\tmp\combined` at `fb6e288` | PASS | 258 passed, 5 warnings, rc 0 (`fb6e288_tier0.log`) |
| Tier 2 at `fb6e288`, Windows | 2 FAILED, accepted as platform-specific | 658 passed, 3 skipped, 2 failed (`fb6e288_tier2.log`) |
| Allen tests, WSL Ubuntu | PASS (verification of record for the symlink-capable environment) | 6/6 at `fb6e288` (`fb6e288_allen_wsl.log`); 6/6 also at `aab5c4b` and `9934535` |
| Allen tests, Windows | FAIL, unresolved environment limitation | `WinError 1314` at `os.symlink`; owner ruling: no Windows Developer Mode change |
| Import-path test (`test_environment_provenance::test_jomission_import_resolves_to_this_checkout`) | PASS in the qualified checkout | passes inside tier 0 at `fb6e288` |
| Independent review | PASS | `aab5c4b` content (8 claims) and `fb6e288` memory commit (9 claims); see Review |

Tier 2 classification: the two failures are exactly `tests/test_allen_v1_units.py::test_end_to_end_write_inputs_then_verdict_puts_planted_gaba_tone_in_its_unit_only` and `::test_ampa_vs_gaba_tone_ratio_matches_rws_prediction_and_swapped_order_fails`. Each raises `OSError: [WinError 1314]` from `os.symlink` (`scripts/allen_v1_units.py:56`). No other test fails.

Owner ruling (2026-11-14): accept the WSL run as verification of record for the symlink-capable environment. This establishes the Allen behaviour under WSL, not Windows compatibility. Do not enable Windows Developer Mode.

## Earlier runs (for the record)

| Run | Tier 0 | Tier 2 | Note |
|---|---|---|---|
| `d1b5d27` baseline, no path file | 5 failed, 253 passed | 10 failed, 650 passed, 3 skipped | environment (shared venv shadowing, no path file) |
| `02cfc1f` OPEN-47 candidate, no path file | 5 failed, 253 passed | 7 failed, 653 passed, 3 skipped | environment |
| `f6d1d1f` combined, path file set | 2 failed, 256 passed | 4 failed, 656 passed, 3 skipped | dashboard and series-dashboard content repaired later |
| `aab5c4b` | 258 passed, rc 0 | 2 failed, 658 passed, 3 skipped (952 s) | Allen only |
| `258d118` in the qualified tree | 258 passed, rc 0 | not run | receipt-only over `aab5c4b` |
| `258d118` in a `git clone --shared` | 257 passed, 1 failed | not run | the failure, `test_jomission_import_resolves_to_this_checkout`, was caused by the clone sitting outside the path the venv `.pth` points to (`D:/tmp/combined`); the same test passes in the qualified tree (2 passed) |
| `fb6e288` | 258 passed, rc 0 | 2 failed (Allen), 658 passed, 3 skipped | final candidate |

## Environment

- Windows qualification venv `D:\venvs\jomission-qual` (Python 3.14, not `--system-site-packages`). jaxfne 0.5.8 installed non-editable. Path file `Lib\site-packages\_qual_checkout.pth` points to `D:/tmp/combined`.
- WSL Ubuntu, venv `~/jq-venv` (Python 3.12.3), jaxfne 0.5.8 non-editable. Clone `~/jomission-cand` checked out detached at `fb6e288` for the Allen run. The branch reached WSL by bundle (`/mnt/d/tmp/gate_fb6e288.bundle`), because `/mnt/d/tmp/combined` has an alternate object path WSL cannot resolve.

## MEMORY and lessons

- `manifests/harness/lessons.json` holds the full store: 11 lessons, each `{trigger, cause, repair, evidence, scope}`. Unchanged since `aab5c4b`.
- `MEMORY.md` is a compact orientation view: one line per lesson, `- {id}: {trigger} -> {repair}`. It renders to 3592 characters. Budget 4000 in `scripts/render_memory.py` and `tests/test_project_memory.py`. Cause, evidence and scope are only in `lessons.json`.
- The renderer raises when over budget; it no longer trims the oldest lessons silently.
- The injection limit of the global autocompact-memory plugin is not verified. Searched 2026-11-14: installed plugins, plugin cache, and settings. No config found. The earlier figure "6000" was the repo's own old renderer constant, not a verified plugin limit. The 4000 budget is a chosen ceiling below that figure, not a measured limit.
- Historical: `docs/receipts/open47/VERIFICATION.md` (for `6092f75`) records a 6000-character budget against the then-current 6872-character file. Not changed.

## Review

- `aab5c4b` (content): reviewer PASS on 8 claims. Covered: the first ten lessons unchanged; the L11 fields rebuild the base line exactly; MEMORY diff empty; `--check` 0; budget constant; both memory tests pass; unique ids; `manifests/sealed_artifacts.json` unchanged; receipt matches code. The reviewer did not rerun tier 0 or tier 2; those counts are read from the logs above.
- `fb6e288` (memory commit): reviewer PASS on the content (lessons byte-identical; `--check` 0 and re-render clean; 11 lines matched verbatim; over-budget path exits 1 and writes nothing; budget 4000 in both files; no silent trimming path; test passes). Claim 1 was worded as "exactly 3 files" and failed as worded, because the range also contains the two receipt commits. Those receipt commits touch only this file.
- Receipt-only commits `62389ad` and `258d118`: each touches only `docs/receipts/gate_reconcile/RECEIPT.md` (`git show --stat`).

## Not verified

- Scientific results on real data: not run. The caminos tests use synthetic counts.
- Windows Allen tests: not passing; WSL is the accepted verification.
- GitHub CI on this branch: not read.
- Plugin injection limit: not verified (see above).
