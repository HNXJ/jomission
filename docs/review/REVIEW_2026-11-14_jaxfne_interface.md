# Review 2026-11-14: JaxFNE interface and VIP construction

> ID note (merge 2026-11-20): this document uses the branch numbering. In docs/OPEN.md the rows are: branch OPEN-47 (population_rates) = OPEN-61; branch OPEN-57 (origin/dev test status, duplicate of the dev row) = OPEN-47; 51 = OPEN-62; 52 = OPEN-63; 53 = OPEN-64; 55 = OPEN-65; 56 = OPEN-66; 54 = OPEN-70. OPEN-15, OPEN-37, OPEN-41 and OPEN-50 are unchanged.

Scope: independent review of the JaxFNE 0.5.8 pin against Jomission call sites, from workflow `wf_f4f71947-2eb` (4 map audits, 6 verifiers). Owner rulings of 2026-11-14 are binding here. Base: `origin/dev` d1b5d27. Status: candidates are claims until verified; the OPEN-47 classification is pending (section 6).

## 1. Result

| Item | Verdict | Status |
|---|---|---|
| `population_rates` keyword call breaks under the 0.5.8 pin | CONFIRMED (4 independent verifier passes) | Owner ruling: Option A, seconds form on 0.5.8. Corrected call on `_temp_open47_population_rates` (6092f75), not on dev |
| VIP equation differs by construction path (0.5.9) | CONFIRMED that the builder route yields class-1 VIP under 0.5.9 (verify:2) | Owner ruling: not adopted. OPEN-51 |
| "VIP equation depends on construction path" as stated | REFUTED as stated (verify:3); surviving part is the builder route | Same as above |
| Other 37 map candidates | UNVERIFIED | Preserved as inventory: `docs/review/UNVERIFIED_INVENTORY_2026-11-14.md` |

## 2. Deduplication of `population_rates` (independent evidence kept)

Four map audits reported the same defect. Each is listed, not discarded:

| Source | Map audit | Verifier pass | Verifier note kept |
|---|---|---|---|
| jaxside | map:jaxside | verify:0 | Read-only; ran one `python -c` regex check (process deviation, section 4a). Confirmed signature at `v0.5.8 metrics.py:1082`; no shim in `analysis/__init__.py:40-51` |
| drift | map:drift | verify:5 | Confirmed; installed wheel = v0.5.8 tag not re-checked; runtime TypeError not re-run |
| tests | map:tests | verify:4 | Corrections: test `:61` runs before `:62`; OPEN-41 cited at `docs/OPEN.md:43` is the OPEN-15 row (OPEN-41 is `:67`); the "unreleased" wording conflicts with OPEN-37 (`:63`). Pin conflict with E0 Amendment 1 noted |
| api | map:api | verify:1 | Confirmed; pin `pyproject.toml:18` and `manifests/current_state.json:15` conflict with Amendment 1 (`E0_SEAL.md:56`). Sent a cross-session message to main-jomission [ed0b47] (process deviation, section 4b) |

Net: one defect, four confirmations, one set of corrections (test ordering, OPEN-41 citation, release wording, pin-versus-amendment conflict). Corrections are already in `OPEN.md` (OPEN-15, OPEN-37) and `E0_PROVENANCE_ADDENDUM_1.md`.

Fix: `caminos_analysis.py:105` on branch `_temp_open47_population_rates`: `population_rates(arr, dt=dt, sizes=sizes, pops=POPS, discard_s=DISCARD_S)`. Equivalence on synthetic counts: worst relative difference 0 (18 values; `docs/receipts/open47/`). Sealed E0 results unchanged; see `docs/laminar_motif/E0_PROVENANCE_ADDENDUM_1.md`.

## 3. VIP equation (OPEN-51)

Probe: `docs/receipts/vip_equation/vip_probe.py`, outputs `vip_058.json`, `vip_059.json`; full table in `docs/review/VIP_EQUATION_2026-11-14.md`.

- 0.5.8: every path standard (5/140).
- 0.5.9: builder (64 VIP cells) and raw cmin (28) class-1 (4.1/108); repaired cmin standard (`v_lin = None`).
- Mechanism (source): the 0.5.9 VIP/I presets carry class-1 coefficients; `dataclasses.replace` keeps them; `with_emitter_parameters` calls `refresh_v_coefs`, which resets to standard.
- Verifier refutation of the "as stated" claim (verify:3): cmin VIP starts at b = -0.10, not 0.20; the d-jitter paths do not move VIP alone; the SST `a` replacement does not touch VIP.
- Provisional (owner to confirm): standard 5/140 for Jomission. Dynamic effect not tested.

## 4. Process deviations (recorded, not scientific failures)

a. verify:0 ran one `python -c` regex check. Breaks the CLAUDE.md shell rule (scripts go in files, run by path). Disclosed by the agent. Result unaffected: the claim was confirmed from source.
b. verify:1 sent a cross-session message to main-jomission [ed0b47] with its verdict. Not authorized. Read-only reviewers do not message without owner authorization. Disclosed to main-jomission by the lead.
c. map:api wrote three scratch files (`init_058.py` and two others) despite a read-only brief (inventory row 10).

## 5. Cost and formation

Workflow `wf_f4f71947-2eb`: 10 agents (4 map, 6 verify), 1,825,989 tokens, 359 tool calls, 1,061,335 ms (from the workflow result).

Yield: 6 candidates with a verdict (5 confirmed, 1 refuted as stated), covering 2 distinct defects (population_rates; VIP). 37 candidates remain unverified.

Plan, not run: repeat the same 43-candidate set with a smaller formation (for example one map audit plus one verifier per candidate) and compare unique verified findings per token. Repeat the 10-agent configuration only if the larger formation yields more unique verified findings per token. Owner decision: OPEN-52.

## 6. OPEN-47 gate classification and resolution

Classification: read-only worker on a clone of `02cfc1f` (12 items; receipts in its run). Re-checked here: the dashboard and memory items (reproduced on d1b5d27 by a clean clone), the shared venv pointer (`_editable_impl_jomission.pth`, `pyvenv.cfg`), and the clean-venv tier-0 run.

| Class | Items | Resolution |
|---|---|---|
| REPO_DEFECT | `test_rebuild_is_byte_identical`, `test_open_stack_rows_shown`, `test_no_innerhtml_and_no_absolute_paths` (drive-letter path in OPEN-50), `test_project_memory_current_and_within_budget` | Fixed on `_temp_open47_integrate_20261114` (342942c, 93a8157): OPEN-50 paths reworded; `docs/DASHBOARD.html` rebuilt; `MEMORY.md` re-rendered from `manifests/harness/lessons.json`. Gate code unchanged. |
| ENVIRONMENT | `test_environment_provenance` (shared venv imports the `_work` clone); collection errors `test_activity_balance*` (stray `tests` package in system site-packages), `test_v21_pv_microstate`, `test_whole_system_ei_native_state` (`scripts` shadowed by the jaxfne_v058 tree on PYTHONPATH); `test_jaxfne_vis` (shared venv has matplotlib; clean venv needs the `dev` extra) | Clean venv `D:/venvs/jomission_pin058`: system site-packages off, jaxfne 0.5.8 from pyproject, jomission editable from the integration checkout, `[dev]` extras. Tier 0 at 93a8157: `PROJECT_CHECK_PASS`, 258 passed, 0 failed. |
| ENVIRONMENT (open) | `test_allen_v1_units` x2 (WinError 1314, symlink privilege at `scripts/allen_v1_units.py:56`) | Not resolved. Needs a privileged host or WSL, or a TEST_DEFECT ruling (Hamm). |

Tier 0 receipt conflict (5 failed, 2 collection errors in the shared venv vs 258 passed in the old receipt): explained by the environment rows above. The 258 figure is now reproduced by a run at 93a8157, not only by the OPEN.md record.

Tier 2 (compatibility, `manifests/test_tiers.json` command) at 93a8157, fresh clone, clean venv: 657 passed, 3 skipped, 3 failed, 0 collection errors. Tracked `results/` unchanged. Failures:

- `test_allen_v1_units` x2: ENVIRONMENT (symlink privilege), open, see the table.
- `test_environment_provenance::test_jomission_import_resolves_to_this_checkout`: artifact of the run location. The clone's tier-2 run used the editable install that points at the integration checkout, and the test (correctly) rejects that. Run in the integration tree: 2 passed.

The OPEN-47 receipt numbers (626 passed, 7 failed, 3 skipped, 5 collection errors) are not reproduced; the clean venv has no collection errors.

## 7. Pending

- Allen pair: passes in WSL (6 of 6; control fails 2 of 6). Fails on Windows with WinError 1314 (symlink privilege, environment). Resolved for the candidate as in section 8.
- OPEN-53 closed; OPEN-55 closed (L11 retired 2026-11-14, owner ruling).
- Dev test-status record is OPEN-57 (formerly the second "OPEN-47" at `docs/OPEN.md:73`).
- Inventory row 14 (0.5.9 release-status conflict): resolved by owner ruling 2026-11-14, not by verification.

## 8. Gate status at candidate 05f9dec (2026-11-14)

Code identical to `05f9dec` in this commit; the commit that adds this section changes docs only. Check: `git diff 05f9dec HEAD -- jomission scripts tests manifests` is empty.

| Gate | Environment | Result | Record |
|---|---|---|---|
| Tier 0 (`project_check.py --offline`) | Windows, pinned venv `jomission_pin058`, integration checkout | PROJECT_CHECK_PASS; 258 passed, 5 warnings | `scratchpad/t0_05f9dec.log` |
| Tier 2 (manifest command) | Windows, `jomission_pin058` | 658 passed, 3 skipped, 2 failed: `test_allen_v1_units` x2, `[WinError 1314]` symlink privilege (environment) | `scratchpad/tier2_win.log` |
| Tier 2 (manifest command) | WSL Ubuntu, venv with jaxfne 0.5.8 from PyPI | 645 passed, 3 skipped, 8 failed, 7 errors; the Allen pair passed. Failure messages point to commits and the worktree being unreadable from WSL (`fatal: not a git repository` on the worktree); not re-run individually | `scratchpad/tier2_wsl.log` |
| Allen pair (6 tests) | WSL Ubuntu, clone of the bundle | 6 passed. Control (symlink write removed in a throwaway clone): 2 failed, 4 passed | `~/qual/*_allen.log`, `~/qual/control_allen.log` |

Counts reconcile (H3): Windows 658 + 3 + 2 = 663; WSL 645 + 3 + 8 + 7 = 663; same collected set.

Independent review (reviewer agent, read-only, 2026-11-14): ACCEPT. Four minor doc findings, fixed in this commit: OPEN-53 and OPEN-47 wording in section 7, the inventory mapping, the OPEN-47 line cite (105 dev, 106 candidate), and the 93a8157 receipt wording in OPEN-47.

Integration rule (owner, 2026-11-14): the Allen pair passes on the candidate in a symlink-capable environment, tier 0 and tier 2 pass on the candidate apart from that environment's own failures, and the review is ACCEPT. The keeper merges to `dev`; this branch is not pushed to `dev`.

Not verified: WSL tier 2 failures individually (attributed by message, not re-run); WSL run used a clone whose HEAD was unreliable in this session, so the Allen result is attributed to the code-identical tree.

## 9. Checkpoint seal (2026-11-14)

- Remote branch: `_temp_open47_integrate_20261114`. Its code is `05f9dec` (tier 0 and tier 2 record in section 8). `9737a5b` adds the section 8 gate record, retires L11 (OPEN-55), and corrects OPEN-47 wording. The commit that adds this section is the sealed head; `git ls-remote origin _temp_open47_integrate_20261114` reports it.
- Verified: remote equal to local before this commit; a fresh clone of the remote branch resolves to the sealed head.
- Remaining work, in order:
  1. Keeper merges `_temp_open47_integrate_20261114` into `dev`. Only the keeper pushes `dev`. After the merge, verify `dev` and delete the merged branch locally and on origin.
  2. OPEN-56: re-point the shared venv to the integration checkout once the `_work` lane on `_temp_p2_jomission` is retired or confirmed unused. Owner decision.
  3. OPEN-57: close after the dev merge; the gate record in section 8 is the evidence.
  4. Lane `_temp_ocma_17-143` at `6a11755a` (on `7897f381`), from session ocma-jnwb-omission. Not part of this branch; unpushed; reviewed ACCEPT by two reviewers. Owner direction needed before any push or merge.
  5. `_temp_review_rulings_20261114` (`a7b4fd7`) is superseded by this branch; delete it when its merge state is confirmed (merged-branch rule).
- Next check after the keeper's merge: `scripts/project_check.py --offline` on `dev` in the pinned venv (`PROJECT_CHECK_PASS`) and `tests/test_environment_provenance.py` passing from that checkout.

