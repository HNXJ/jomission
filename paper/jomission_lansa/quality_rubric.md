# LANSA manuscript quality rubric (fixed 2026-11-19, before stage-10 round 2)

Scores manuscript quality, which is separate from workflow progress (`manuscript-stages.md`). The rubric was fixed
before the round-2 inspection, at the commit that adds this file; changing it after the inspection needs Hamm's
word. Negative scientific results stated accurately are findings, not defects. Author metadata is out of scope
until stage 13.

## Evidence states

Each criterion item is PASS (checked against its source, no defect), DEFECT (checked, wrong), or UNKNOWN (not
checked, or its source is unavailable). UNKNOWN scores 0, the same as a DEFECT, so incomplete inspection never
raises the score.

## Mandatory gates

| Gate | Condition |
|---|---|
| M1 | The build is clean: 0 errors, 0 LaTeX warnings, every reference and citation resolved |
| M2 | No claim exceeds its verdict: INCONCLUSIVE, FAIL and untested stay so in abstract, results and discussion |
| M3 | Every number in the abstract, results, discussion, tables and captions traces to a source file holding that value |
| M4 | Every cited source file is verifiable by hash, or the text states that it is unavailable |
| M5 | A fresh independent skeptical review has no open major finding |

Any gate DEFECT caps the score at 49. Any gate UNKNOWN caps it at 89. Only all five gates PASS allow 90 or more.

## Weighted criteria (sum 100)

| # | Criterion | Weight | Items |
|---|---|---|---|
| C1 | Numerical accuracy | 25 | numbers in text, every results and supplement table, captions, against source files |
| C2 | Claim-evidence calibration | 20 | verbs match evidence; limitations stated; mechanism framed as tested, not validated |
| C3 | Statistics | 15 | unit of inference, sidedness, multiplicity, effect size with interval and n |
| C4 | Reproducibility | 15 | code commits, environment and jax dependence, run records and hashes, data access |
| C5 | Internal consistency | 10 | text, figures, captions, tables and supplement agree |
| C6 | Figures | 5 | each figure readable at print size, caption complete, traces to its producer |
| C7 | Methods completeness | 5 | a reader could rerun each analysis from Methods and the records table |
| C8 | Prose | 5 | match-writing-style rules; no slop; no hedge words |

Each criterion scores its weight times the fraction of its items that PASS. Report the score, the gate caps
applied, the coverage (the weight whose items were all checked, out of 100), every DEFECT and every UNKNOWN.
