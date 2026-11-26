# Handoff: LANSA geometry lane, 2026-11-25

For: the Claude agent taking over this lane. Owner: Hamm. Read this file, then `docs/laminar_motif/DECISIONS_2026-11-25.md`
on the review branch, before doing anything.

## 0. Where things are (absolute paths)

- Workspace, the canonical repo checkout: `/mnt/d/jomission`. Its working tree is shared with other agents. It is on
  `_temp_integ_20261124` (commit ef0a766) and is currently clean. Do not switch branches or edit files there.
- Cowork checkout, which holds the T1 receipts (not in this repo): `/mnt/d/cowork/jomission`. HEAD 21b2697. 51 tracked T1
  paths are deleted in its working tree. Do not restore them; they belong to another agent's working tree.
- Session scratch, not authoritative and may be cleared: `/tmp/claude-1000/-mnt-d-jomission/f444cf63-3499-4b62-8fb3-022e4b659be3/scratchpad`.
  It holds analysis scripts (`_temp_*.py`), drafts and this handoff.

## 1. Branches and commits (remote SHAs checked 2026-11-25)

| Ref | Commit | Status |
|---|---|---|
| `origin/_temp_lansa_geom_20261125` | 53a3e45 | Checkpoint, unaccepted. Retained unchanged. Do not rewrite. |
| `origin/_temp_lansa_geom_review_20261125` | d4aa932 | Current head. Revision 3: wording fixes from the independent review, plus the review record. Built on revision 2 (1591c18), which answered the REVISE. The handoff itself is at e36d3bb, which is an ancestor of d4aa932. |
| `origin/dev` | 264c1e8 | Contains the original audit, uncorrected. Not the corrected version. |

The corrected audit (`docs/laminar_motif/EVIDENCE_AUDIT_2026-11-25.md`) exists only on the review branch.

Files on the review branch (contents as of d4aa932; unchanged from 1591c18 except where noted):
- `scripts/lansa_t1_anatomy_recount.py`: read-only recount of the 57 T1 motif receipts under four depth rules.
- `docs/receipts/lansa_t1_anatomy_recount_2026-11-25.tsv`: its output. Only the header differs from 53a3e45.
- `docs/laminar_motif/DECISIONS_2026-11-25.md`: the decision record (revision 3 wording). Read it first.
- `docs/receipts/lansa_geom_rev2_review_2026-11-25.md`: the independent review of 1591c18 (ACCEPT, two minor findings, both fixed in d4aa932).
- `docs/laminar_motif/GEOMETRY_PHASE1_PROTOCOL_2026-11-25.md`: the Phase 1 protocol as text. Not authorised.
- `docs/laminar_motif/EVIDENCE_AUDIT_2026-11-25.md`: the corrected audit.

## 2. Authority

Authorised by the 2026-11-25 ruling:
- review, verification, and freezing protocol text;
- read-only runs of the recount script;
- pushes to `_temp_` branches.

Not authorised:
- merge into `dev` or `main`, or any push to `main`;
- any scientific run, including Phase 1 execution and Phase 2;
- edits to sealed specs (`docs/laminar_motif/SEAL_*.md`, `manifests/sealed_artifacts.json`), T1 verdicts or receipts;
- restoring the cowork deletions;
- any abstract edit. D1 is deferred, so the manuscript story, methods and figures come first.

Owner decisions are listed in section 5. Do not make them yourself.

## 3. Done and verified

- Depth coordinate: LANSA bins are cortical-depth fractions about mid-L4, negative superficial. Verified from
  `docs/laminar_motif/data/lansa_density_X.json` units and `docs/laminar_motif/PLAN_PAPER_JOMISSION_LANSA.md` line 29.
- Layer depths: the arithmetic `z = (µm − 431)/779` is correct and matches `scripts/allen_v1_figure.py`. Result: L4 ±0.095,
  L2/3 mid −0.216, L5 mid +0.198, L6 mid +0.406.
- Layer source: `docs/laminar_motif/ALLEN_V1_REPLICA_SPEC.md` section 1, which the owner RETIRED on 2026-11-17. Its source is a
  mouse V1 model (`AllenInstitute/oscillations-biorealistic-mouse-v1-model`, per `OPEN38_ALT_ALLEN_INVENTORY.md`). The P2
  inventory does not repeat the depths. No macaque boundary is in the repo.
- Recount (descriptive): rule O reproduces the earlier 21 of 57, and the script stops if it does not. Rule A (L4 bottom
  +0.095): 15 of 57 meet the anatomical pattern, and 9 also pass crossing_2se. Rules A, B and C give identical counts, so
  they are threshold checks, not a sensitivity analysis. The observed Y passes rule A (a known-positive check).
- Independent review of 53a3e45: REVISE, no wrong numbers. All items addressed in 1591c18, except the T2–T4 test-tier label.
  Revision 2 has not been reviewed yet.
- T1 receipts: the 57 motif JSON files equal cowork HEAD (57 of 57, sha256). The reviewer recomputed the verdict statistics
  from the 57 verdict JSON files and they match the audit.
- Gamma criterion, T1 record and FAIL: unchanged.
- S_ab figures: they match `docs/receipts/lansa_M_audit_2026-11-25.tsv` (on `origin/dev` at 264c1e8).

## 4. Known problems: do not repeat these

1. Leakage. The existing unit spectra were selected against the observed prototype Y. sst 15 is the T1 axis-2 winner by
   r_proto over all 11 areas (`docs/laminar_motif/FINDINGS_2026-11-10.md` line 105). kd 1.41 and sstd 21 were chosen by the
   same rule; the frozen values are at FINDINGS line 137. No existing receipt set supports a leakage-free area-held-out claim.
2. Manuscript depth placement (`paper/jomission_lansa/sections/results.tex` lines 481-488). It was selected on the 11-area
   prototype. Its "held-out" runs are other finite-size runs scored against the same prototype. That is not a held-out test.
   The paper's UNRESOLVED comment (about line 485) flags a different problem, an untracked file.
3. Layer boundaries. The mouse-model depths are used only as a convention. Whether they apply to the macaque LANSA areas is
   unknown. A macaque source, such as the LANSA histology methods, is not in the repo.
4. Cowork working tree. 51 tracked T1 paths are deleted: 36 `.json`, 6 `.log`, 9 `.png`. Nine are reference symlinks; three
   were checked (mode 120000). The rest are not classified. None of the 57 search tags' run records is deleted.
5. Gate. `scripts/project_check.py --offline --no-tests` fails. In `/home/hamm/.venvs/jaxfne`, `jaxfne` 0.5.9 is installed
   and 0.5.8 is required, and `jomission` is not importable there. The manifest's next authorised task is
   `SCI-AGSDR-CALIBRATION`, and `V2_LOCAL_OPERATION` is FAIL. The checkpoint did not cause this, and nothing was changed to
   get past it.
6. Phase 1 is not in the manifest. Running it needs an authorising entry first.
7. The T2–T4 test tier is unknown. `manifests/test_tiers.json` lists test files but no tier label for T2–T4.
8. The 15/9 counts are descriptive. They are not confirmations of the empirical motif, and must not be reported as
   replications.
9. Do not describe the depth placement as held-out, and do not describe rules A, B and C as a sensitivity analysis.

## 5. Owner decisions outstanding (ask Hamm, do not decide)

1. Phase 1 route. R1: run on existing spectra and label it exploratory, with no held-out claim. R2: generate new spectra
   whose parameters are fixed before any comparison with Y. R2 is a new lineage and needs a run. Only R2 supports a
   held-out claim.
2. Confirm the applied values from the ruling, whose numbers were blank in transmission: deep threshold +0.095; L5 mid
   +0.198 (primary); L6 mid +0.406 (sensitivity).
3. `sigma_alt` for the 2 x 2 design. The protocol proposes 0.25, which has no data basis.
4. A macaque source for the L4 and L5 boundaries.
5. The cowork deletions: restore them, or commit the records.
6. The T2–T4 test tier, and whether the sealed T1 reference stays as the reference (the ruling says keep it).
7. Whether Phase 1 is registered in the manifest.
8. The manuscript geometry paragraph: reword it as exploratory, or hold it until a held-out design exists.

## 6. Next steps, in order

1. Wait for the owner's answer on route R1 or R2 (item 5.1). Do not start Phase 1 before it is registered.
2. Get a fresh independent review of revision 2 (1591c18). The reviewer must not be the writer.
3. Owner or environment task: fix the gate (`jaxfne` 0.5.8, editable install of `jomission`), then run `project_check`.
4. Only then, and only if the owner decides, consider integration into `dev`. That is the owner's call.

## 7. Commands

Recount (read-only; from the repo root; writes a TSV to stdout):

```bash
cd /mnt/d/jomission && /usr/bin/env python3 scripts/lansa_t1_anatomy_recount.py --y docs/laminar_motif/data/lansa_lfp_Y.json
```

Check a remote ref:

```bash
git -C /mnt/d/jomission ls-remote origin refs/heads/_temp_lansa_geom_review_20261125
```

Commit method used for 1591c18 (no change to the shared working tree). Write the message file, then:
- `GIT_INDEX_FILE=<scratch>/idx git -C /mnt/d/jomission read-tree <parent>`;
- `git hash-object -w <file>` for each file, then `git update-index --add --cacheinfo 100644,<blob>,<path>` (with the same `GIT_INDEX_FILE`);
- `git update-index --force-remove <path>` for deletions;
- `git write-tree`, then `git commit-tree <tree> -p <parent> -F <msg>`;
- `git push origin <sha>:refs/heads/<literal-branch-name>`.

Write the branch name literally in the push. The `git_guard` hook refuses a branch name held in a shell variable.
End each commit message with `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`, as the session requires.

## 8. Do not

- Use `git add .` or `-A`. Do not switch branches or edit files in `/mnt/d/jomission`.
- Force-push, rewrite `53a3e45` or `1591c18`, or push to `main` or `dev`.
- Edit sealed specs, T1 verdicts, receipts or manifests.
- Restore or delete anything in `/mnt/d/cowork/jomission`.
- Start Phase 1 or Phase 2, or any simulation.
- Call the 15/9 counts replications, or the depth placement held-out.
