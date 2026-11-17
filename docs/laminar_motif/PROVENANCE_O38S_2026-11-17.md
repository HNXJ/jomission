# OPEN-38 result provenance, 2026-11-17: manifest, retrospective V0-V3 and lineage record SCI-OPEN38-O38S done

## Outcome (supersedes "Blocked" below)

Owner 2026-11-17 authorized option (a), the retrospective V0-V3; no contract exemption.

- `scripts/open38_viz.py` drew `results/viz/SCI-OPEN38-O38S/` at dc2acf7. Nothing was simulated.
  - V0 rebuilds the ref seed-1 construction and is accepted only when its edge table equals the run record. Its rule
    table covers all 12 737 617 edges.
  - V1 and V2 are the recorded raster samples, 0-1000 ms from simulation start, which is the start-up transient. V2
    compares the arms with the reference at seed 1.
  - V3 is the atlas: the scorer maps from the recorded `.npz`, plus the verdict metrics and the recorded counts.
- Checks before drawing:
  - each of the 45 `.npz` files matched its stage integrity receipt's `npz_sha256`;
  - each run record carried its stage's execution SHA.

  Hashes are in `provenance.json`.
- The lineage record `SCI-OPEN38-O38S` (`manifests/lineage_registry.json`):
  - the record's commit is dc2acf7, and it lists 257 artifacts, among them the 234 results as exact entries;
  - the spec is at 4478e9a, with the amendments 462c078, 584922c and 8fed7f2;
  - verdict label `O38S_GATES_PASS_REGISTERED`, with a note that it adds no validation and seals nothing;
  - `validate_lineage` returns [] and `results_coverage` returns [];
  - the record was inserted as text, so the 34 other records are unchanged.
- Independent review (read-only reviewer, at 193df09): ACCEPT-WITH-CHANGES. The reviewer's own code confirmed:
  - all 45 `.npz` and run-JSON hashes and SHAs;
  - re-derived raster picks: V1 has 5194 spikes, and flat 4375 and mst 5725 in V2;
  - the 33 rule counts;
  - the map r_ab means against the verdict files and §6;
  - the record fields and commit ancestry.

  Changes applied at dc2acf7 and in the following commit:
  - the start-up transient window and arm-against-arm V2 are labelled;
  - V0's unit/area wording is fixed, and its rule table is named as the full topology;
  - the verdict label was renamed.
- Gates at the record commit: `project_check.py` with `JOMISSION_REMOTE` set to https gives PROJECT_CHECK_PASS, tier 0
  258 passed, in the venv `D:/venvs/_temp_jomission_lansa` (editable install → D:/jomission). The OPEN-38 test files
  give 39 passed.
- E0/E1 exemption precedent: recorded as OPEN-51, not changed.

Authorization: owner 2026-11-17, option 1. Register all 234 tracked `results/laminar_e2/o38s_*.json` in one OPEN-38 lineage
record, with complete file-level provenance. Registration is not validation and not sealing.

## Done

- `scripts/open38_provenance.py` writes `manifests/o38s_provenance.json`. Each file carries:
  - its group, evidence commit, git blob and the SHA-256 of its committed bytes;
  - its execution SHA, checked against every `git_sha` or `head` field in the file, or UNRECORDED.

  `--check` recomputes at the recorded revision and compares the working tree.
- Groups and execution SHAs:

  | Group | Files | Execution SHA |
  |---|---|---|
  | smoke | 9 | 3294090 (`git_dirty` true) |
  | stage 1 | 63 | 53f8b89 |
  | ruling-B re-check | 1 | checker at 8fed7f2 |
  | stage-2 attempt 1 (STOP) | 2 | 584922c |
  | stage 2 | 156 | 0d71622 |
  | §6 | 2 | UNRECORDED |
  | status config | 1 | UNRECORDED |

- No listed file is in `manifests/sealed_artifacts.json`.
- `tests/test_open38_provenance.py` has 10 tests:
  - coverage equals `git ls-files`;
  - no file is sealed;
  - a changed working-tree file fails `--check`;
  - each of 5 changed manifest fields fails `--check`;
  - an unknown name has no group.

  A mutant that skips the working-tree comparison fails its test.
- Independent review (read-only reviewer, at 716db2f): ACCEPT-WITH-CHANGES. The reviewer's own code confirmed coverage,
  bytes, blobs and evidence commits for 234/234, with no contradictory SHA. Changes applied:
  - attempt 1 records 584922c, from `STAGE2_O38S_2026-11-17.md` line 5;
  - the smoke group's `git_dirty` is noted;
  - manifest-field mutation tests were added.

  Gap left: heads are read only from the `git_sha` and `head` keys. The reviewer scanned all string values and found no
  contradiction.

## Blocked: the lineage record

`manifests/visualization_contract.json` requires every executed, non-grandfathered model lineage to carry V0-V3
artifacts: construction schematic, initial 1000 ms raster, final raster and atlas. Enforcement is in
`jomission/harness/validate.py`, `validate_visualization`.
- The `NOT_A_MODEL_LINEAGE` exemption covers only lineages that "construct no model and run no simulation". OPEN-38 did
  both, so the exemption would be false, and so would `executed: false`.
- OPEN-38 has no V0-V3 artifacts. Raster samples exist in the local untracked `.npz` run outputs.
- The reviewer independently found no third truthful route.
- Seals: the registry does not treat lineage-named files as sealed. Seals live only in `sealed_artifacts.json`, and the
  write guard is `jomission/harness/seals.py`.

Owner options: (a) authorize post-hoc V0-V3 for OPEN-38, labelled as made after interpretation; (b) amend the contract,
for example by adding OPEN-38 to `grandfathered`.

Precedent: `SCI-LAMINAR-MOTIF-1` (E0) and `SCI-LAMINAR-E1` both ran simulations, yet both carry the
`NOT_A_MODEL_LINEAGE` exemption. Not corrected here.

## Promotion receipt: `dev` at 520d5fb (owner-authorized 2026-11-17)

| Item | Value |
|---|---|
| Authorization | owner 2026-11-17: promote `_temp_lansa-provenance` at 520d5fb to `dev`; nothing sealed; OPEN-51 stays open |
| Before | `origin/dev` fcd8155c0ef7a01dc8c4f6d7a721e706da1b3b16 |
| After | `origin/dev` 520d5fb4c249e66cd8a69177b4581a6a28e6f2a0; `git ls-remote` agrees; pushed 2026-11-17 about 23:10Z |
| Mode | fast-forward: fcd8155 is an ancestor of 520d5fb, 66 commits; the promoted tree is the gated tree |
| Gates on that tree | `project_check.py` (https remote via `JOMISSION_REMOTE`, accepted by the owner for this promotion) PROJECT_CHECK_PASS, tier 0 258 passed; `--no-tests` PASS at 520d5fb; OPEN-38 tests 39 passed; `test_harness_state.py` 6 passed |
| CI | `project-check.yml` runs only on pull requests or manual dispatch, so the push started none. `pages.yml` builds on a push to `dev` and deploys from `main` only (ruling Q23), so nothing was published |
| Delivered | `dev` SHA sent to main-vwin (status-generator dependency) |
| Semantics | `SCI-OPEN38-O38S` is labelled `O38S_GATES_PASS_REGISTERED`: provenance and coverage registered, not scientific validation, nothing sealed |
