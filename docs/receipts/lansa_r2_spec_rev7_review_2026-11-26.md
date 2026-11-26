# Review of the R2 spec, draft 7 (73ba46b), 2026-11-26

Reviewer: main-jomission session. Not the writer. Subject: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md`, the pre-seal
receipt `docs/receipts/lansa_r2_preseal_2026-11-26.md`, `scripts/r2_compare_runs.py` and `manifests/current_state.json` at
73ba46b on `_temp_lansa_r2_spec_20261125` (remote tip checked with `git ls-remote`; diff taken against bc68dbb, draft 6).

Verdict: REVISE, one required line (Q4, a receipt count). The rev6 items (Q1 to Q3, m1 to m6) are fixed as described. A seal
gate blocker is open (below); it is not a text item.

## Verified

| Claim | Check | Result |
|---|---|---|
| Comparator blob sha256 `f13a9d95...176f` | `git show 73ba46b:scripts/r2_compare_runs.py \| sha256sum` | verified |
| Q1: header (b) names `next_authorized_task` | diff vs bc68dbb | verified |
| Q2: 493 JSON, 127 `jax_devices` (`cpu:0`), 366 without | rev6 recount at 21b2697 | verified (same numbers) |
| Q3: §9 "Open owner items: none" | diff | verified |
| m1: thread pins from the smoke run on; baseline default equals pinned serial | comparator pair P4 | EQUAL, verified |
| m2: `jomission/` and `scripts/run_e2_stack.py` unchanged since a4b9a89 | `git diff --stat a4b9a89 73ba46b -- jomission scripts/run_e2_stack.py` | empty, verified |
| m3: ratio about 13-14x, derived | 2544 s / 181-194 s = 13.1-14.1 | verified (derived) |
| m4: §1 lists drafts 1-6 with receipts | diff | verified |
| m5: source session named | diff | the session ID is the peer's statement; not checked here |
| m6: missing-`counts` guard | diff | verified |
| Receipt sha256: A `.npz` and B `.npz` `5b9876a0...`; A `.json` `fab79116...`; B `.json` `c2e78424...`; negative `.npz` `3eb9f7fc...`; serial s7 and par s7 `.npz` `5b9876a0...` | `sha256sum` on the files under `/tmp/claude-1000/-mnt-d-cowork-jomission/f323fbb0-.../scratchpad` (read-only) | all match |
| Simulate times 211.9, 230.8, 235.8, 232.2 s | `simulate` field in each JSON | match §3 |

Re-run of the committed comparator (blob `f13a9d95`) with the gate venv's interpreter (the system `python3` has no numpy), read-only:

| Pair | Verdict | Detail |
|---|---|---|
| P1 A (cc33b37) vs B (a4b9a89), §1 | EQUAL | 22 arrays, 10 fields, 48491 spikes |
| P2 B vs known negative, §1 | DIFFERENT | `g_nS` differs |
| P3 serial s7 vs par s7, §3 | EQUAL | |
| P4 B vs serial s7, §3 | EQUAL | |
| P5 par s7 vs par s8, §3 | DIFFERENT | 21 of 22 arrays (`block_keys` equal); `edges` differs |

## Required

Q4. Receipt §1 says the comparator was "re-run on all four pairs of sections 1 and 3 with the same verdicts: EQUAL, DIFFERENT,
EQUAL, DIFFERENT". The receipt holds five comparisons (§1 A/B, §1 negative, §3 serial/par s7, §3 B/serial s7, §3 par s7/s8),
with verdicts EQUAL, DIFFERENT, EQUAL, EQUAL, DIFFERENT. Fix: "five pairs" and that verdict list. The count must reconcile
before the seal (decisions.md rule 5 and H3).

## Gate blocker (for the seal, not a text item)

`project_check` PASS in the canonical checkout does not hold for this text yet. `/mnt/d/jomission` is on
`_temp_integ_20261124` at ef0a766. The spec is absent at that HEAD, and its `current_state` reads
`next_authorized_task` = SCI-AGSDR-CALIBRATION. A gate run there would test another tree. The gate venv's editable install
also points at `/mnt/d/jomission` (rev6: a clone fails the editable-install check). So the seal gate must run where that
install imports the seal SHA. Options, for the peer and Hamm:
- (a, recommended) a dedicated venv, editable-installed from a clean clone at the seal SHA, with jaxfne 0.5.8 from the
  `v0.5.8` tree of `/mnt/e/repos/jaxfne`; run `project_check.py` there and record the result. The shared `jomission058`
  venv is not changed.
- (b) after Hamm merges the seal into dev, run in `/mnt/d/jomission` at dev. Hamm's act.
Do not switch branches in `/mnt/d/jomission`.

## Verdict

REVISE. Apply Q4; the reviewer will check that line and no more. ACCEPT covers the text only. It is not the seal gate, and the
seal waits on the gate blocker.

Tier-0 tests were not run. No spec, manifest, OPEN.md or seal changed in this commit.
