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
| `_temp_lansa_geom_20261125` (deleted after merge) | 53a3e45 | Geometry checkpoint, unaccepted. Ancestor of dev; the branch was deleted because its tip is contained in dev. Do not rewrite. |
| `_temp_lansa_geom_review_20261125` (deleted after merge) | e52b950 | Merged into dev; e52b950 is an ancestor of dev. Revision 3 (d4aa932): wording fixes from the independent review, plus the review record. Built on revision 2 (1591c18), which answered the REVISE. The handoff itself is at e36d3bb, which is an ancestor of d4aa932. |
| `_temp_jaxfne059_ruling_20261125` (deleted after merge) | c2e6470 | Merged into dev; c2e6470 is an ancestor of dev. Accepted ruling on JaxFNE admissibility (review ACCEPT at 3739b75; registration at d057609, review ACCEPT with one minor finding). Adds `docs/AUTHORITY_RULING_2026-11-25_JAXFNE059.md` and `docs/receipts/gate_jomission058_2026-11-25.md`; registered in `manifests/current_state.json` (`jaxfne_family_admission_ruling`) and `docs/OPEN.md` row 8. |
| `origin/_temp_lansa_r2_spec_20261125` | a4b9a89 | Draft 4 of the R2 spec with the three review wording fixes applied (on d2df9d5), based on dev 54048dc (draft 1 at 6b95a85, draft 2 at 6185106, draft 3 at 5d4acf5). Not sealed; nothing authorised to run. Draft 1 reviewed REVISE (`docs/receipts/lansa_r2_spec_review_2026-11-25.md`); draft 2 reviewed REVISE (`docs/receipts/lansa_r2_spec_rev2_review_2026-11-25.md`); draft 3 reviewed REVISE (`docs/receipts/lansa_r2_spec_rev3_review_2026-11-26.md`); draft 4 reviewed ACCEPT with three minor wording fixes (`docs/receipts/lansa_r2_spec_rev4_review_2026-11-26.md`, on this branch). |
| `origin/dev` | 54048dc (c2e6470 before the repoint) | Fast-forward from 264c1e8 on Hamm's order (2026-11-25): "merge the ruling branch into dev", then "Push without tests". Carries the geometry lane (53a3e45 through e52b950) and the ruling chain (44d46f1 through c2e6470); every commit has an independent ACCEPT. Tier-0 tests NOT run; the quick gate check fails only on the editable install, which points at `/mnt/d/jomission`. |

The corrected audit (`docs/laminar_motif/EVIDENCE_AUDIT_2026-11-25.md`) is on dev, merged from the review branch at c2e6470.

Files on dev, merged from the review branch (contents as of d4aa932; unchanged from 1591c18 except where noted):
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
- merge into `main`, or any push to `main`. (The merge into `dev` was done on Hamm's order, 2026-11-25; section 1.)
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
  Revision 2 (1591c18) was reviewed: ACCEPT, two minor findings, both fixed in d4aa932 (`docs/receipts/lansa_geom_rev2_review_2026-11-25.md`).
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
4. Cowork working tree. 51 tracked T1 paths are deleted: 36 `.json`, 6 `.log`, 9 `.png`. All 51 are symlinks at HEAD
   21b2697 (mode 120000, checked for all 51). 33 targets exist in the working tree; the other 18 are `t1_sst15_*` links,
   themselves deleted, which resolve through a second hop to `sw_n4000_dens_k1_kd1_sst15_*` files at HEAD. None of the 57
   search receipts references a deleted path. This is an alias-naming question, not data loss; restoring or committing the
   links is owner item 5.5 (DECISIONS section 9, decision 6). Whether any script resolves the `t1_*` names is UNKNOWN (not grepped).
5. Gate. `scripts/project_check.py --offline --no-tests` failed in the shared venv (`jaxfne` 0.5.9 installed, 0.5.8 required;
   `jomission` not importable there). Fixed by option A' in a separate venv, `/home/hamm/.venvs/jomission058`: jaxfne 0.5.8
   from tag `v0.5.8` (11957afa), `jomission` editable. jxm-dev reports rc 0 at ef0a766, and the reviewer re-ran it with the
   same output as `docs/receipts/gate_jomission058_2026-11-25.md`. This is the quick contract only. `V2_LOCAL_OPERATION` is
   still FAIL, independent of the environment. The shared venv is unchanged. A clean run at c2e6470 (dev) has not been made from `/mnt/d/jomission`: the clone run fails only on
   the editable install, which points at `/mnt/d/jomission` (expected). A clean run needs dev checked out there, which is
   Hamm's call. Tier-0 tests were not run on dev.
6. Phase 1 is not in the manifest. Running it needs an authorising entry first.
7. The T2–T4 test tier is open. `manifests/test_tiers.json` (about lines 98-116) places `tests/test_t2_readiness.py` and `tests/test_t4_t5_analysis.py` under a tier-2 heading; the tier of the experiments themselves is not stated.
8. The 15/9 counts are descriptive. They are not confirmations of the empirical motif, and must not be reported as
   replications.
9. Do not describe the depth placement as held-out, and do not describe rules A, B and C as a sensitivity analysis.
10. JaxFNE admissibility, now resolved by a family-scoped ruling. The per-run `jaxfne_version` counts (by script; reviewer
   ACCEPT at ee61a8d) are: T1 axis-1 and axis-2 search runs (36 records) 0.5.8; T1 axis-3 (sstd) search runs (18), the three
   confirmation runs (seeds 4-6, the T1 FAIL), the reference `sw_n4000_dens_k1_kd1_sst15_*` records (7), all 47 O38 records
   on dev (45 accepted, 2 smoke) and E1's per-run records (15) 0.5.9. `docs/AUTHORITY_RULING_2026-11-25_JAXFNE059.md` (on
   `_temp_jaxfne059_ruling_20261125` at 3739b75, reviewer ACCEPT) admits 0.5.9 for exactly these families. It grants no
   scientific acceptance: the T1 FAIL stands, each run keeps its exact version, and T8 and anything above 0.5.9 stay NOT
   COVERED. OPEN-37 and OPEN-41 stay open.
   Conditions the ruling keeps, and the review findings F1 and F2. (a) The axis-1 reference level mixes versions:
   `t1_kd1_n4000_s1_T8` resolves to `sw_n4000_dens_k1_kd1_sst15_s1_T8` (0.5.9); seeds 2 and 3 resolve to `lfs_n4000_s2_T8` and
   `lfs_n4000_s3_T8` (0.5.8). Any statistic pooled over that level must state the mix; whether it matters scientifically is
   open. (b) The E1 seal (`docs/laminar_motif/E1_SEAL.md` lines 5 and 13) and `docs/laminar_motif/HANDOUT_GEMINI.md` line 25
   ("JaxFNE 0.5.9 for E0 and E1") name 0.5.9 for E1. The 2026-11-24 ruling did not admit E1; the 2026-11-25 ruling admits it
   under its own seal pin only. The seal and handout wording are unchanged. (c) The counts come from cowork HEAD 21b2697
   (T1, `sw_*`, E1) and `origin/dev` 264c1e8 (O38); `/mnt/d/jomission` holds no `t1_*` records. The owner word is relayed from
   jxm-dev's session: ruling section 1 quotes "confirmed, register the ruling", and that session registered the ruling at
   d057609 (section 5, item 9). Hamm's words were not seen in this lane.

## 5. Owner decisions outstanding (ask Hamm, do not decide)

1. Phase 1 route: decided R2 (Hamm, 2026-11-25, by direct answer in jxm-dev's session; relayed here, not seen). R2 is a new
   lineage and needs a run. Its draft spec is at a4b9a89 on `_temp_lansa_r2_spec_20261125` (draft 4 with the review wording fixes applied):
   not sealed, nothing authorised to run. Draft 1 was reviewed REVISE (`docs/receipts/lansa_r2_spec_review_2026-11-25.md`);
   draft 2 was reviewed REVISE (`docs/receipts/lansa_r2_spec_rev2_review_2026-11-25.md`); draft 3 was reviewed REVISE (`docs/receipts/lansa_r2_spec_rev3_review_2026-11-26.md`); draft 4 was reviewed ACCEPT with three minor wording fixes (`docs/receipts/lansa_r2_spec_rev4_review_2026-11-26.md`, on this branch). The GPU venv `/home/hamm/.venvs/jomission058gpu` is built (`pip show` checked 2026-11-26). Owner choices in
   draft 2, relayed from jxm-dev's session: P0, sigma_alt 0.25, the depths as listed, a 0.5.8 GPU venv. Open: confirm the
   support and falsification rule and the manifest entry `SCI-LANSA-GEOM-R2` (owner); the bitwise P0 baseline (Major A in
   the draft 2 review) and the device choice D1 or D2 (owner, draft 4 section 5). The GPU venv is built, so it is no longer open.
2. Confirm the applied values from the ruling, whose numbers were blank in transmission: deep threshold +0.095; L5 mid
   +0.198 (primary); L6 mid +0.406 (sensitivity).
3. `sigma_alt` for the 2 x 2 design. The protocol proposes 0.25, which has no data basis.
4. A macaque source for the L4 and L5 boundaries.
5. The cowork deletions: restore them, or commit the records.
6. The T2–T4 test tier, and whether the sealed T1 reference stays as the reference (the ruling says keep it).
7. Whether Phase 1 is registered in the manifest.
8. The manuscript geometry paragraph: reword it as exploratory, or hold it until a held-out design exists.
9. Gate at dev c2e6470: a clean `project_check` run from `/mnt/d/jomission` (dev checked out there), with tier-0 tests not yet run. That is
   Hamm's call. Registration of the ruling (`manifests/current_state.json`, `docs/OPEN.md` row 8) is done at d057609; the
   owner word for it is relayed, not seen here. Also open: whether the mixed axis-1 reference level matters scientifically,
   and whether the E1 seal and handout wording need correcting.

## 6. Next steps, in order

1. Route R2 is decided (section 5, item 1). Do not run anything before the R2 spec is revised, reviewed, sealed and authorised by a manifest entry.
2. Done: the JaxFNE record was reviewed (ACCEPT at ee61a8d; F1 and F2 applied at e52b950); the ruling was reviewed (REVISE at 44d46f1, ACCEPT at 3739b75).
3. Done: gate fixed in `/home/hamm/.venvs/jomission058` (option A'; jxm-dev reports quick-contract PASS). `V2_LOCAL_OPERATION` is still FAIL.
4. Done: integration into dev at c2e6470, on Hamm's order (2026-11-25). Tier-0 tests were not run.

## 7. Commands

Recount (read-only; from the repo root; writes a TSV to stdout):

```bash
cd /mnt/d/jomission && /usr/bin/env python3 scripts/lansa_t1_anatomy_recount.py --y docs/laminar_motif/data/lansa_lfp_Y.json
```

Check a remote ref:

```bash
git -C /mnt/d/jomission ls-remote origin refs/heads/dev
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
- Force-push, rewrite `53a3e45` or `1591c18`, or push to `main`. Push to `dev` only on Hamm's order.
- Edit sealed specs, T1 verdicts, receipts or manifests.
- Restore or delete anything in `/mnt/d/cowork/jomission`.
- Start Phase 1 or Phase 2, or any simulation.
- Call the 15/9 counts replications, or the depth placement held-out.
