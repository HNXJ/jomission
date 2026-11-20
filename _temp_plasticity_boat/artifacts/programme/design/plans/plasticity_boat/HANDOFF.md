# Plasticity Boat: handoff (2026-11-19)

Entry point for a fresh session. Repo: jomission, branch `_temp_plasticity_boat` (pushed; the tip is the commit that adds this file, `git log -1 origin/_temp_plasticity_boat`). Paths below are relative to `_temp_plasticity_boat/` in that repo.

## Start

1. `git fetch && git switch _temp_plasticity_boat`.
2. Run `python scripts/project_check.py` from the repo root. It must end `PROJECT_CHECK_PASS`. If jomission is not installed editable from this checkout, use an isolated venv with `pip install -e .` in it; do not change a shared install.
3. jomission's `manifests/current_state.json` names `SCI-AGSDR-CALIBRATION` as the next authorized task. It is not this programme. Do not run it for Plasticity Boat.
4. Read this file, then the open decisions below. Ask the owner (Hamm) before any decision in that list.

## Scope and rules (owner rulings, 2026-11-19)

- Development starts only: `default_rng(2025)` (validation, 20 starts) and `default_rng(2027)` (calibration, x10). Held-out `default_rng(2026)` unused.
- Criteria v1 (`criteria_v1_frozen.md`) and v2 (`criteria_v2.md`) are frozen. Any change is a new labelled criteria version with owner approval.
- The 10-neuron fixture is frozen.
- No publication, no metadata before manuscript stage 13, no merge to `dev` or `main` until independent review. Owner: after review, integrate useful work into `dev` and retire this branch.

## Results (observed)

- V0 regression, JaxFNE source 0.5.9: exact, max abs diff 0.0 (`receipts/dev100_v2/regression/V0_source_0.5.9.log`).
- V0 regression, pinned `jaxfne==0.5.8`: fails, `AttributeError: module 'jaxfne.paradigm' has no attribute 'SailorConfig'` (`receipts/dev100_v2/regression/V0_pinned_0.5.8.log`). The pin lacks the Sailor API that `code/circuit100.py` uses.
- dev-100 v1 gate: R2 and R4 fail (`dev100_gate_record.md`).
- dev-100 v2 resting sweep (`receipts/dev100_v2/resting/`): R2 passes only at g_s <= 4.5; R4 fails at every g_s with directional gain g_d = 24. C3 stops at step 3.
- Diagnostic (not selection-eligible): g_d = 4 at g_s = 3 gives R3 diff 53.5 Hz and R4 max 82.2 Hz, both passing. The R4 threshold lies in (4, 6).

## Decisions waiting on the owner

| ID | Question | Blocks | Packet |
|---|---|---|---|
| O-8 | Directional gain g_d: (1) criteria v3 with a G_D grid in (4, 6) sealed before calibration, (2) keep v2 and stop at C3 step 3, (3) drop the dev-100 track | PB-DEV100-V2-VALIDATION | `O8_decision_packet.md` |
| O-9 | Decoder: sum or mean over each 5-neuron motor group (sum is about 5x the mean's force gain) | P1 reporting | `O8_decision_packet.md` section 8 |
| O-6 | Eight-strategy list and primary metric (`protocol_v0.md` sections 3 to 5) | confirmatory runs | `protocol_v0.md` |
| O-7 | Owner's 15-stage manuscript workflow (not in repo or jarness; stage numbers provisional) | manuscript stages | `manuscript/plasticity_boat/main.md` |

The owner has chosen Option A for the 100-neuron directional encoder (signed-position population). The choice is recorded in the trigger of `PB-DEV100-V2-VALIDATION` in `manifests/todo.json`, and in the JaxFNE branch's `open_stack.md` (O-5, now closed).

## Evidence location

- Tracked: `receipts/` (about 0.5 MB): V0 regression, dev-100 v2 resting sweep, dev-100 v1 summaries and `start_00.json`.
- Not in git (gitignored, on the Mac that ran the work): `artifacts/_temp/plasticity_boat/` (9.6 MB). Includes the dev-100 v1 trace folders (`P1_pos_vel/`, `N1_pos_only/`, `P1_tau05/`), the N1 log, gate runs and smoke runs. Obtain by copy from that Mac, or regenerate. Figures and results that cite them must not cite `_temp/` (JaxFNE todo EX-28).

## JaxFNE side (local only, not pushed)

- Repo `jaxfne`, branch `_temp_crazy_sailor_m0m1_b`, worktree `jaxfne-crazy-sailor` on this Mac. It holds the Crazy Sailor programme state (`artifacts/state.md`, `open_stack.md`, `todo_stack.md`), and its commit `4b9e58e2` removes the 13 migrated Plasticity Boat files. The branch has no upstream.
- Record-integrity FAIL: `artifacts/release/budget_amendment_2026-10-23.md` is pinned at blob `c52e830b`; live content is `cdc17e41` after commits `46293fcc` (adds the Option B section; tree-size allowlist 2797 to 2798) and `03e24a16` (text only). Neither commit repinned. Option B's authority is cited as a 2026-11-19 chat from the owner, which I could not verify.
- Owner ruling: no waiver, no automatic repin to `cdc17e41`, JaxFNE push blocked until integrity passes and required checks pass.
- Until that branch is pushed, a fresh session elsewhere has no Crazy Sailor programme state, only this file and the jomission branch.

## Not run or not examined

- jomission tier-0 suite (33 files): not run. The suite lock needs write access to `~/.claude/suite.lock`, which the sandbox denies. Run it in the background under the lock with an owner-approved exception.
- `release_artifacts` gate: NOT EXAMINED (dist absent).
- Critic on the JaxFNE section 13.7 amendment (O-2): verdict REVISE. Findings not yet resolved.

## Do not

- Do not repin the JaxFNE record to `cdc17e41` without the owner's go on that SHA.
- Do not change g_d, the criteria, the gain grid, the decoder or the start sets without owner approval.
- Do not use the held-out start set.
- Do not change the shared editable install.
- Do not merge this branch into `dev` or `main` before independent review.
