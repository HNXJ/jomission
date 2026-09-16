---
name: jomission-lineage-review
description: Review a proposed jomission scientific result or branch against its lineage authority, detecting criterion weakening, retuning, stale evidence, unsealed parameter choices, cross-lineage contamination, and unsupported PASS. Use before accepting, merging, or citing a result.
---

# Lineage review

Inputs: the result branch, its lineage record (or proposed record), the parent
record, and the TODO item. Each check yields a finding with a receipt, or
"clear". Commands: [references/review_checks.md](references/review_checks.md).

| defect | question | how to detect |
|---|---|---|
| criterion weakening | Are the applied thresholds, windows, and estimators those sealed in the spec and gate file? | diff spec at the pre-execution commit against what the analysis used; gate-file blob recorded in the result equals the blob at the spec commit |
| retuning | Did any parameter, bracket, seed, or rule change after outputs existed? | `git log` on spec and parameter files between spec and result commits must be empty |
| stale evidence | Is every cited artifact the current version, from the current engine and plant? | receipts resolve at the result commit; engine version in the result equals `current_state.jaxfne_version`; no cited file is marked superseded |
| unsealed parameter choice | Is any value in the result absent from the spec? | list numeric parameters in the result config; each appears in the spec or a frozen parent artifact |
| cross-lineage contamination | Did code, data, or state from another unmerged lineage enter? | `git log --first-parent` of the branch touches only its own paths; inputs list no other branch |
| unsupported PASS | Does the evidence scope match the claim scope? | PASS requires every gate check true on every declared window and cell; component readings are not gate verdicts; OBSERVED claims have receipts; INFERRED stays INFERRED |

Also check:
- one principal delta; the qualification stages exist for a new mechanism;
- the verdict label in the result equals the status proposed for
  `manifests/current_state.json`, or a `review_override` records why not;
- negatives stay recorded, and retired axes (`docs/project-sources/SUBSTRATE_PROGRAM.md`)
  are not reopened without new authority.

Output: findings ranked by consequence, each with evidence class and receipt,
then a disposition: accept, accept with additive correction, or reject
(new lineage required). The reviewer decides; this skill does not merge.
