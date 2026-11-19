# LANSA stage 10, round 2: verification report (2026-11-19)

Scored with `quality_rubric.md` (fixed at 6f0e51c before this round). Manuscript state scored: d289583 on
`_temp_lansa-stage10-r2`. Items per criterion are the item groups in the rubric's Items column; an item is PASS only
when checked against its source in this round.

## Gates

| Gate | State | Evidence |
|---|---|---|
| M1 build | PASS | latexmk rc 0, 0 LaTeX warnings, 35 pages at d289583 |
| M2 claims within verdicts | PASS | reviewer B: every abstract sentence supported at no greater strength (after the E1 scope fix, Hamm's ruling) |
| M3 every number traced | UNKNOWN | all table cells traced to primary records (437 cells, 0 DEFECT); 23 text numbers traced; the rest of the text and caption numbers not traced |
| M4 cited sources hashed | PASS | D6 212 entries; 41 added this round, re-hashed by the reviewer 41/41; the 122 absent earlier all found byte-identical |
| M5 skeptical review | PASS | 3 majors closed per Hamm's rulings and checked independently (A, B, C PASS); minors open |

Cap: 89 (M3 UNKNOWN).

## Criteria

| # | Weight | Items PASS / total | Score | DEFECT or UNKNOWN items |
|---|---|---|---|---|
| C1 numerical accuracy | 25 | 1/3 | 8.3 | text numbers UNKNOWN (23 traced), caption numbers UNKNOWN |
| C2 calibration | 20 | 3/3 | 20.0 | none |
| C3 statistics | 15 | 2/4 | 7.5 | sidedness UNKNOWN; effect size with interval DEFECT (T8 rule at n = 3, ranges without intervals) |
| C4 reproducibility | 15 | 2/4 | 7.5 | code commits DEFECT (C15, C16 cite "this revision" without a hash); data access UNKNOWN (archive and DOI, stage 13) |
| C5 consistency | 10 | 3/5 | 6.0 | figures UNKNOWN; text DEFECT (results.tex:238-239 and discussion.tex:11-12 attribute the loss differently) |
| C6 figures | 5 | 0/1 | 0.0 | UNKNOWN until the C6 items below are fixed (inspected 2026-11-19: 6 DEFECTs) |
| C7 methods completeness | 5 | 0/1 | 0.0 | UNKNOWN until the C7 items below are fixed (inspected 2026-11-19: 4 DEFECTs, 3 minors) |
| C8 prose | 5 | 1/3 | 1.7 | unasked contrasts DEFECT (style check: 6 in discussion, 1 in introduction); parentheses rule UNKNOWN |

Score 51 of 100 (cap 89 not binding). Coverage, the weight whose items were all checked: 20 of 100 (C2).

## Open minors from the skeptical review

methods.tex:52-54 dirty-tree flag wording; C15 and C16 without hashes; methods.tex:313 T8 rule at n = 3,
uncorrected, and the "reference higher" wording; results.tex:238-239 against discussion.tex:11-12 (the D 21 ms
level, not fresh seeds); results.tex:53-54 exact permutation count (13 824).

## New work that bears on the lock

E1b (`docs/laminar_motif/E1B_SEAL.md`, sealed 2026-11-19) is running: 27 runs, seeds 6-8, arm A band-floor
sensitivity and arm B fixed PV. Its result enters Results, Methods and possibly the abstract sentence on E1, so the
methodological lock waits for it.

## M3 trace (after this report's score)

Claim-level trace at d289583, tables in `docs/laminar_motif/m3_trace_2026-11-19/` (1057 rows). Six DEFECTs found
and fixed at c1f320c to 4340019 (Fig S1 label sizes, M/D power, probe-control sentence, dirty-flag count, permutation
wording, untraced seed negative control); C15-C18 records pinned. Eight UNKNOWN remain, none holdable in a file:
three figure readings (35 Hz, 40 Hz, "one unit"), four literature values (Mazzoni 90 % three times, LANSA 1 s), one
process statement. M3 stays UNKNOWN; the score is not recomputed until E1b is folded in.

## C6 and C7 inspection (2026-11-19, pages from the d289583-plus-minors build, 35 pages)

Inspected: all 23 figure files exist (no `pendingbox` in the build). Rendered and checked: Figs 1-8, S1-S4 against
their captions and the text values. Figs 1-7 and S2 caption ranges match the text (Fig 4 legend n sums to 93; Fig 6
titles match the T8 table; Fig 5 settings match the text; delivery labels in Figs 3 and 5 match methods.tex:43-44).

**C6 DEFECTs (figures)**

| # | Where | Defect |
|---|---|---|
| C6-1 | supplement.tex:14, Fig S1 | Figure shows internal text "(kd: agent choice)" and a repo filename (`..._details.md`). Remove both from the figure. |
| C6-2 | supplement.tex:27-32, Fig S2 | Caption omits the colour meaning (difference of relative power, seed A minus seed B; scale -1 to 1). |
| C6-3 | results.tex:508-519, Fig 8 | Caption says "top" and "bottom" but not which panel is phi (left) and CSD (right); panels unlettered. |
| C6-4 | results.tex:515-516, Fig 8 | Caption gives sigma_e = 0.3 S/m, lateral distance 0.05, and offsets z_u-0.15 / z_u-0.10 and "both assumed; Methods". Methods states none of them (see C7-1). |
| C6-5 | supplement.tex:57-66, Fig S4 | Caption claim "dim above 35 Hz (read from the figure)" is an unmeasured visual claim. Measure it or delete it. |
| C6-6 | supplement.tex:22, 54 | Fig S1 (fig:s-reference) and Fig S3 (fig:s-ladder) are never cited in the main text; S3 is cited only in Table C17. |

Minor, not counted: Fig 7 caption does not say theta = 90 delta_r is undefined (figure title does).

**C7 DEFECTs (methods)**

| # | Where | Defect |
|---|---|---|
| C7-1 | methods.tex:142-146 (CSD) | sigma_e, lateral distance, the z_u offsets and the trough-detection band-pass are not stated. The Fig 8 caption points here for them. |
| C7-2 | methods.tex:145-146 | Trigger contact rule ("contact with most 10-30 Hz band power", which makes the trace circular) is stated only in Fig 8. |
| C7-3 | methods.tex:144 vs Fig 8 caption | Naming: Methods says "SST distal apical", figures say "SOM". Methods says "AMPA basal and apical", Fig 8 says "AMPA half at z_u, half at z_u-0.10". One term; state the placement. |
| C7-4 | methods.tex:266-278 (T1) | T1 winner is chosen from 3 axes x 7 levels = 21 levels (63 level-seed scores). No statement of search size or of multiplicity for the winner's fresh-seed result. |

Minor, not counted: jackknife window count not stated (derived: 6 s retained at 1 s windows = 6 windows per run,
methods.tex:153-158); "image 10-30 Hz" row label undefined (methods.tex:198); the dirty-tree wording (methods.tex:52-55) is
already an open minor.

**C7 items checked and complete (PASS):** seeds per test (E0 1-5, E1 1-3, T1 selection 1-3 and fresh 4-6, T8 1-3,
ladder 1-3); durations and discards; dt; software versions and per-run authority (methods.tex:42-51); E1 permutation
definition and p bound (methods.tex:80-86); nulls and controls (methods.tex:164-170); pass rule (172-183); ladder rule
(254-258); T8 decision rule and deviations (313-321); event vs dense per test (43-44, 59-68).

**Still open, not C6/C7:** C3 intervals (T8 and fresh-seed confirmation, n = 3); the remaining C8 contrasts; the
permutation count (no decision made, see the Monte Carlo note); Data availability section is the stage-13 decision
(backmatter.tex:1).

## Resume (if this session is lost)

1. Batch: detached, launched 2026-11-19T17:56Z from a clone pinned at 4c99e05 (`D:/tmp/_temp_e1b_clone`), output
   `D:/tmp/_temp_e1b_runs`, log `batch.log` there. Rerunning `scripts/e1b_batch.py --out D:/tmp/_temp_e1b_runs` from
   that clone resumes it (completed receipts are skipped).
2. Score: `python -B scripts/e1b_score.py --runs D:/tmp/_temp_e1b_runs --out results/laminar_e1b/e1b_result.json`
   (gates 3 and 4 run first; exit 2 on a gate failure or a missing run).
3. Copy receipts and counts to `results/laminar_e1b/`, hash them into D6, write the E1b result into Results and
   Methods (E1 stays INCONCLUSIVE), review, rescore with the rubric, present the lock decision to Hamm.
