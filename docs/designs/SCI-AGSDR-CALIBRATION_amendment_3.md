# SCI-AGSDR-CALIBRATION amendment 3 (in effect on commit; 2026-11-04)

Owner ruling (Hamm, Q10, 2026-11-04): H08 informational; H07 and H11 remain hard gates. Supersedes `SCI-AGSDR-CALIBRATION_amendment_3_DRAFT.md` (removed). Nothing here changes a sealed file in place.

## Problem (unchanged from the draft)
- Objectives manifest (`manifests/agsdr_local_objectives.json`) lists H08 (motif-gain ratios within plus or minus 30 percent) as hard_threshold; the sealed design (`SCI-AGSDR-CALIBRATION_design.md`, line 18) lists nine hard gates without H08; the runner records H08 as informational (box 0.7 to 1.3). 0 of 24 candidates fall inside the box.

## Ruling
1. H08 is informational for stage 0: reported per candidate, does not gate feasibility. Matches the sealed design and the runner.
2. The objectives manifest is not edited in place; it keeps its original text as history; this amendment is the reconciliation record.
3. H07 and H11 remain **hard** gates (option (b) of the draft). The runner does not compute them. They must be computed under a new sealed spec before stage-0 feasibility can be called final.

## Consequence
- Stage-0 hard feasibility under the runner's gates is 2 of 24 (OPEN-26, H05 over all E cells: AGS_S0_06, AGS_S0_20). Because H07 and H11 are not yet computed, this is an upper bound: the final feasible set is at most these 2.
- Stage 1 stays blocked (OPEN-25 viable-regime verdict; shortfall below 8 feasible).

## Does not
Change any threshold, weight, sampling rule or candidate parameter; select candidates; authorize stage 1 or a rerun; decide OPEN-25.
