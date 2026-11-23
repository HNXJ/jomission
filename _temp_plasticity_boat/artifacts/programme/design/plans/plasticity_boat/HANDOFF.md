# Plasticity Boat: handoff (2026-11-22)

Entry point for a fresh session on any PC. Repo: jomission, branch `_temp_plasticity_boat` (pushed to origin). The tip is the commit that adds this file: `git log -1 origin/_temp_plasticity_boat`. Paths below are relative to `_temp_plasticity_boat/` in the repo root (the branch tracks the programme under that folder). Measured checks: `receipts/seal_checks_2026-11-22.md`.

## Start

1. `git fetch origin && git switch _temp_plasticity_boat`. `git status` must be clean.
2. From the repo root run `python scripts/project_check.py` (full: needs network and tier-0; quick: add `--offline --no-tests`). It must end `PROJECT_CHECK_PASS`. Quick check measured `PROJECT_CHECK_PASS` at c1a5b73.
   - If a global `jomission` install points at another checkout, `test_jomission_import_resolves_to_this_checkout` fails (environment failure, not a model result). Use an isolated venv with `pip install -e .` in this checkout. Do not change a shared install.
3. `manifests/current_state.json` names `SCI-AGSDR-CALIBRATION` as the next authorized task. It is not this programme. Do not run it for Plasticity Boat.
4. Read this file, then the open decisions below. Ask the owner (Hamm) before any decision in that list.

## Scope and rules (owner rulings)

- 2026-11-19: development starts only. `default_rng(2025)` (validation, 20 starts) and `default_rng(2027)` (calibration, x10). Held-out `default_rng(2026)` unused.
- Criteria v1 (`criteria_v1_frozen.md`) and v2 (`criteria_v2.md`) are frozen. Any change is a new labelled criteria version with owner approval.
- The 10-neuron fixture is frozen.
- No publication, no metadata before manuscript stage 13, no merge to `dev` or `main` until independent review. Owner: after review, integrate useful work into `dev` and retire this branch.
- 2026-11-21: D3 ratified with refinements (`d3_dispositions.md`).
- 2026-11-22: feasibility bound uses sqrt(2) fm (option a, componentwise clip, plant unchanged). Option b (norm clip, a plant change) is not adopted and needs a new ruling.

## Where the work stands (observed, 2026-11-22)

- DMS acceptance tests, `code/test_dms_task.py`: 19 passed at c1a5b73, no skips. T09 needs a JaxFNE checkout whose `jaxfne/paradigm.py` has `SailorConfig` (JaxFNE `origin/dev` has it). With the installed JaxFNE 0.5.8 (no `SailorConfig`), T09 is skipped.
- Tier-0 (33 files, `manifests/test_tiers.json` tier 0) at c1a5b73: 258 passed, 0 failed, rc 0. DMS tests are not in tier-0; their evidence is the DMS run above.
- Open minors (from the D3 record): B2 proprioception is required by its docstring but not enforced; the first-CUE latch rule is only partly exercised by T12 (a latch-every-CUE-step mutant survives); T05 known-positive is near trivial; T07 validity is checked only through the exact bound value; the seeds for DMS development and validation are not named (D3.3); radius and timings are provisional, not calibrated (D3.4).
- Dev-100 track (`PB-DEV100-V2-VALIDATION` in `manifests/todo.json`, blocked on O-8): V0 regression is exact under JaxFNE source 0.5.9 and fails under pinned 0.5.8 (no `SailorConfig`). Gate record: R2 passes only at g_s <= 4.5; R4 fails at every g_s tested at g_d = 24. C3 stops at step 3 under v2. Diagnostic g_d = 4 at g_s = 3 passes R2 to R4 and is not selection-eligible.
- Option A (signed-position population) is the owner's choice for the 100-neuron directional encoder (recorded in the todo trigger).

## Decisions waiting on the owner (Hamm)

| ID | Question | Blocks | Packet |
|---|---|---|---|
| O-8 | Directional gain g_d: (1) criteria v3 with a G_D grid in (4, 6) sealed before calibration, (2) keep v2 and stop at C3 step 3, (3) drop the dev-100 track | PB-DEV100-V2-VALIDATION | `O8_decision_packet.md` |
| O-9 | Decoder: sum or mean over each 5-neuron motor group (sum is about 5x the mean's force gain) | P1 reporting | `O8_decision_packet.md` section 8 |
| O-6 | Eight-strategy list and primary metric (`protocol_v0.md` sections 3 to 5) | confirmatory runs | `protocol_v0.md` |
| O-7 | Owner's 15-stage manuscript workflow (not in repo or jarness; stage numbers provisional) | manuscript stages | `manuscript/plasticity_boat/main.md` |

## Evidence location

- Tracked: `receipts/` (about 0.5 MB): V0 regression, dev-100 v2 resting sweep, dev-100 v1 summaries and `start_00.json`, and `seal_checks_2026-11-22.md`.
- Not in git (gitignored, on the Mac that ran the work): `artifacts/_temp/plasticity_boat/`, 123 files, 9,848,386 bytes (earlier notes said 9.6 MB; measured 9.8 MB). It holds the dev-100 v1 trace folders (`P1_pos_vel/`, `N1_pos_only/`, `P1_tau05/`), the N1 log, gate runs, and smoke runs.
- Manifest, tracked: `receipts/gitignored_evidence_manifest.tsv` (sha256, bytes, relative path per file). To verify a copy, run this inside the copied `artifacts/_temp/plasticity_boat/` folder; it must print only `OK` lines:

  ```bash
  tail -n +2 receipts/gitignored_evidence_manifest.tsv | awk -F'\t' '{print $1"  "$3}' | shasum -a 256 -c
  ```

  (The manifest path is relative to this `_temp_plasticity_boat/` folder, so copy the manifest into the evidence folder first, or run the check with the manifest's path.)
- Figures and results that cite these files must not cite `_temp/` (JaxFNE todo EX-28).

## JaxFNE side (Crazy Sailor; not on this branch)

- The Crazy Sailor branch `_temp_crazy_sailor_m0m1_b` was merged into JaxFNE `dev` (merge `05657514`, 2026-11-20). The remote branch was deleted. Its commits are ancestors of `origin/dev`.
- The Crazy Sailor stacks (`state.md`, `open_stack.md`, `todo_stack.md`) are not on JaxFNE `dev`; they exist only on the local branch and in history. `dev` has `artifacts/programme/design/plans/crazy_sailor/m0_contract.md` and the sailor API `jaxfne/paradigm.py`.
- Record integrity: `dev` passes (`check_record_integrity.py --check`, rc 0, 0 violations). The local branch fails (1 violation: `budget_amendment_2026-10-23.md`).
- The JaxFNE decisions are open with Hamm: O-10, O-11, O-12 and EX-32 in the JaxFNE stacks (where the stacks live; whether the 2026-11-20 merge stands; the integrity repin).

## Not run or not examined

- `release_artifacts` gate: not examined (dist absent).
- Full `project_check.py` (with network): not run; the quick check was run.
- JaxFNE §13.7 critic verdict REVISE (9 findings): unresolved on the JaxFNE side (EX-22).

## Do not

- Do not repin any JaxFNE record without a SHA-specific go from Hamm.
- Do not change g_d, the criteria, the gain grid, the decoder or the start sets without owner approval.
- Do not use the held-out start set (`default_rng(2026)`).
- Do not change the shared editable install.
- Do not merge this branch into `dev` or `main` before independent review.
- Do not run `SCI-AGSDR-CALIBRATION` for this programme.
