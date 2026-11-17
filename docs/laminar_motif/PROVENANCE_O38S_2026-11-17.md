# OPEN-38 result provenance, 2026-11-17: manifest done; lineage record blocked by the visualization contract

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
