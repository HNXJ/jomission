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
| C6 figures | 5 | 0/1 | 0.0 | UNKNOWN, not inspected this round |
| C7 methods completeness | 5 | 0/1 | 0.0 | UNKNOWN, not inspected this round |
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

## Resume (if this session is lost)

1. Batch: detached, launched 2026-11-19T17:56Z from a clone pinned at 4c99e05 (`D:/tmp/_temp_e1b_clone`), output
   `D:/tmp/_temp_e1b_runs`, log `batch.log` there. Rerunning `scripts/e1b_batch.py --out D:/tmp/_temp_e1b_runs` from
   that clone resumes it (completed receipts are skipped).
2. Score: `python -B scripts/e1b_score.py --runs D:/tmp/_temp_e1b_runs --out results/laminar_e1b/e1b_result.json`
   (gates 3 and 4 run first; exit 2 on a gate failure or a missing run).
3. Copy receipts and counts to `results/laminar_e1b/`, hash them into D6, write the E1b result into Results and
   Methods (E1 stays INCONCLUSIVE), review, rescore with the rubric, present the lock decision to Hamm.
