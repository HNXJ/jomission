# Independent review 1 of the OPEN-38 status generator (2026-11-17)

Reviewed SHA: c9b9224 (`_temp_o38s-status`). Reviewer: an independent read-only reviewer agent (it did not write the
code). The owner's three points are recorded with the reviewer's verdict, then each defect and its repair.

**Verdict: ACCEPT-WITH-CHANGES.** The stage-1 logic held. One defect could produce a stage-2 PASS.

## Owner's points

1. **Tolerances.** Measured over all 260 frozen vs ruling-B rule pairs:
   - Centre shift is 0 to 1.0 edge. `binom.interval` returns integer bounds, so ±1 is expected.
   - Width ratio deviates −0.69 % to +0.16 %. The binomial-vs-normal effect at p = 0.14 is −0.003 % to −0.10 % for N
     between 3.7e4 and 3.5e6.
   - The −0.69 % comes from one rule (pv_pv_D, rhor arm). There `rule_interval` takes the union of the N(N−1) and N²
     intervals, and that union's centre offset does not scale with z. It is an artefact of how the interval is
     computed, not a construction difference.
   - Sizes are pinned by field equality and p·N by the centre check, so p, N and the pair set cannot move. The level
     could drift within the 1 % width tolerance (n_tests from about 230 to 260). Intervals at n = 230 still failed on
     pv_pv_S.
2. **Independent derivation.** The reviewer took the pair counts from the receipt sizes and recovered p from the
   frozen 99.9 % interval by grid fit, which reproduces it to 1 edge. It then computed `binom.interval` at
   1 − 0.05/260. The maximum discrepancy against the ruling-B receipt was 2 edges, the grid-fit limit. The family block
   matched.
3. **Adversarial false-PASS.** About 15 forgeries were committed in a throwaway clone, and none reached public PASS:
   - forged family;
   - intervals at a different level;
   - a count changed in one receipt;
   - a flipped `pass` or `ok`;
   - ignored keys in undeclared slots;
   - an extended `ignore_keys`;
   - a changed rule;
   - a wrong execution SHA;
   - a missing compat entry;
   - a mixed integrity or run SHA;
   - a non-boolean `pass`.

   An uncommitted edit showed local PASS and public UNKNOWN, as designed.

## Findings and repairs (repaired at 0f6258e)

| Severity | Finding | Repair | Test (fails on c9b9224) |
|---|---|---|---|
| MAJOR | stage 2 PASS with the smoke segment unfinished: earlier segments were evaluated as live (RUNNING), and only FAIL blocked PASS | PASS needs phases exactly smoke PASS, rest PASS; an unfinished non-final segment is UNKNOWN | `test_stage2_pass_needs_finished_smoke` |
| gap (a) | `identical_paths` such as `README.md` or a nonexistent path passed: `git diff --quiet` is 0 on a path absent at both SHAs | require `jomission` and `scripts/run_e2_stack.py`; every path must exist at both SHAs (`git cat-file -e`) | `test_identical_paths_must_cover_model_code_and_exist` |
| gap (b) | a count forged consistently in both receipts passed: the frozen `ok` was not recomputed | frozen `ok` and `pass` recomputed from its own counts and intervals | `test_consistent_forgery_in_both_receipts_is_not_pass` |
| gap (c) | a rule dropped from both receipts, with n_tests 259, passed | pairs = arms × seeds, one rule set across pairs, and n_tests = the entry's `expected_n_tests` (260) | same |
| MINOR | `WIDTH_TOL` could be 0.75 %, or an exact recompute | not applied. Kept at 1 %. Residual slack: 1 % of a half-width plus 2 edges (≈ 0.04 SD at \|z\| = 4), documented in `STATUS_SOURCE_O38S.md` | — |

Mutants killed after repair: `no_phase_order`, `no_frozen_shape`, `no_path_exist`, `no_min_paths`. Status tests: 17
pass. Real stage-1 receipts still give PASS with no consistency problems.

Remaining exposure: a forgery written consistently into both committed receipts that also stays inside the frozen
interval would pass. Only git history binds the receipts to their builds.

## Review 2 (2026-11-17): stage-2 closure at b9a2ee4 and the generator at 568e659

Independent read-only reviewer agent; it did not write the code. Overall verdict: **REVISE**.

**Part A, stage-2 closure: ACCEPT.** Each item was observed by the reviewer:

- SHA 0d71622 throughout: log HEADs, 30 run `git_sha`, `git_dirty` false, both integrity heads.
- `0d71622..b9a2ee4` adds evidence only and touches no frozen file.
- 30/30 rc=0 and 30 unique exec IDs equal to the logged ones; all 150 output hashes (`.npz` included) match.
- Construction: family 0.05/780; all 780 counts inside their intervals; `e_pv_S` recomputed with scipy for 30/30
  pairs, 0 edges of difference. The other rules' intervals were not recomputed (the fit was too slow).
- Freeze and export manifest `sha256sum -c`: 0 non-OK.
- The 30 table rows equal `_verdict.json` and `o38s_s2_values.json`; E rates recomputed from the `.npz` for 7 runs
  with 0 difference.
- No interpretation found.
- Not verified: the "64 passed" test count and the attempt-1 timeline beyond its HEAD and STOP lines.

**Part B, generator: ACCEPT-WITH-CHANGES.**

- 18 tests passed.
- The merge simulation (568e659 into b9a2ee4) gave stage 1 PASS and stage 2 PASS with no problems; execution SHAs
  stayed 53f8b89 and 0d71622.
- Review 1's repairs held under forgery.
- Owner checks (a) and (b) held for the stage-1 compat path.
- Stage-2 forgeries that still gave public PASS:

| Severity | Finding | Repair | Test |
|---|---|---|---|
| major | stage 2 never read its construction receipt (`pass` false, a forged family or a deleted file gave PASS) | `stage2_evidence`: pass true, ruling-B family over 780, `_receipt_shape` | `test_stage2_pass_is_rederived_from_its_receipts` |
| major | no check of 30/30 runs (a run removed from the log gave PASS, runs 29) | the logged rc=0 set must equal the 30 planned tags | same |
| minor | run `git_sha`/`git_dirty` not in the heads (one run at 584922c gave PASS) | the run `git_sha` set joins the heads; `git_dirty` must be false | same; `test_stage2_receipts_from_another_revision_are_not_pass` |
| minor | `_exec.json` rc, verdict files and the integrity `checks`/`runs` not read | each read and required | `test_stage2_pass_is_rederived_from_its_receipts` |

Mutants killed after repair: `no_s2_evidence`, `no_s2_rc_set`, `no_s2_run_sha`, `no_s2_construction`. Status tests:
19 pass. Real evidence (D:/jomission at b9a2ee4): stage 2 PASS, public PASS, no consistency problems.

The stage-2 receipt document is not read for status; it is linked as evidence with its sha256.

Open: an independent check of these repairs before the status branch merges.
