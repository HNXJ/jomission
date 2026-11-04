# SCI-AGSDR-CALIBRATION amendment 3 (DRAFT, not in effect)

Status: DRAFT for owner review. Not in effect until the owner approves and it is committed as a dated amendment. Nothing here changes a sealed file in place.
Owner ruling used: Q2 (2026-11-04), H08 informational, as in the sealed design.

## Problem
- The objectives manifest (manifests/agsdr_local_objectives.json, H08 entry) lists H08 (pseudogenome motif-gain ratios within plus or minus 30 percent) as a hard_threshold.
- The sealed design (docs/designs/SCI-AGSDR-CALIBRATION_design.md, line 18) lists nine hard gates, without H08.
- The runner (scripts/agsdr_local_sweep.py) records H08 as informational (box 0.7 to 1.3 of the candidate multiplier). 0 of 24 candidates fall inside that box.
- Under the objectives reading, no candidate is feasible and the design stop rule (line 38) applies. Under the design reading, H08 does not gate.

## Proposed change (for owner decision)
1. H08 is informational for stage 0. It is reported per candidate and does not gate feasibility. This matches the sealed design and the runner.
2. The objectives manifest is not edited in place. This amendment records the reconciliation; the manifest keeps its original text as history.
3. H07 and H11 are listed hard in the objectives manifest, are not in the sealed design's nine gates, and are not computed by the runner. The amendment must state whether they are (a) out of the stage-0 hard set, or (b) hard and to be computed under a new spec. This draft does not choose; the owner must.

## What this amendment does not do
- It does not change any threshold, weight, sampling rule or candidate parameter.
- It does not select candidates, authorize stage 1, or approve a rerun.
- It does not decide the viable-regime condition (OPEN-25).

## Consequence to record if approved
- Stage-0 hard feasibility under the runner's gates is unchanged, subject to OPEN-26 (H05 denominator, ruled all E cells: 2 of 24 feasible).

## Approval
- Owner decision on items 1 and 3 is required before commit.
