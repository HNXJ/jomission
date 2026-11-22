---
name: jomission-lineage-review
description: Review a proposed jomission scientific result or branch against its lineage authority, detecting criterion weakening, retuning, stale evidence, unsealed parameter choices, cross-lineage contamination, and unsupported PASS. Use before accepting, merging, or citing a result.
---

# Lineage review

Inputs:
- the result branch
- its lineage record (or proposed record)
- the parent record
- the TODO item

Each check yields a finding with a receipt, or "clear".
Commands: [references/review_checks.md](references/review_checks.md).

| defect | question | how to detect |
|---|---|---|
| criterion weakening | Are the applied thresholds, windows, and estimators those sealed in the spec and gate file? | Diff the spec at the pre-execution commit against what the analysis used. The gate-file blob recorded in the result equals the blob at the spec commit. |
| retuning | Did any parameter, bracket, seed, or rule change after outputs existed? | `git log` on spec and parameter files between spec and result commits must be empty |
| stale evidence | Is every cited artifact the current version, from the current engine and model? | Receipts resolve at the result commit. The engine version in the result equals `current_state.jaxfne_version`. No cited file is marked superseded. |
| unsealed parameter choice | Is any value in the result absent from the spec? | List numeric parameters in the result config. Each appears in the spec or a frozen parent artifact. |
| cross-lineage contamination | Did code, data, or state from another unmerged lineage enter? | `git diff --stat main...BR` touches only its own paths. Inputs list no other branch. |
| unsupported PASS | Does the evidence scope match the claim scope? | PASS requires every gate check true on every declared window and cell. Component readings are not gate verdicts. OBSERVED claims have receipts. INFERRED stays INFERRED. |

Also check:
- One principal delta. The qualification stages exist for a new mechanism.
- The verdict label in the result equals the status proposed for `manifests/current_state.json`. Otherwise, a `review_override` records why not.
- Negatives stay recorded, and retired axes (`docs/project-sources/SUBSTRATE_PROGRAM.md`) are not reopened without new authority.

Output: findings ranked by consequence. Each finding has an evidence class and a receipt.
Then a disposition: accept, accept with additive correction, or reject (new lineage required).
The reviewer decides. This skill does not merge.
