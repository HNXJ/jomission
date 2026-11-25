# LANSA consolidated audit, provisional (2026-11-25)

Status: PROVISIONAL diagnostic. Not the frozen rubric's score and not an accepted result. Rubric: `paper/jomission_lansa/quality_rubric.md`
(fixed 2026-11-19, not modified). Item groups: the stage-10 groups, provisional and versioned here. Numbers review (M3, C1, C5)
still running at writing; its results will replace the rows marked PENDING.

## 1. Gates

| Gate | State | Basis | Source |
|---|---|---|---|
| M1 build clean | UNKNOWN | no TeX engine on this host; stage-10 record says PASS at d289583 (35 pages) | `STAGE10_R2_VERIFICATION.md` gates table |
| M2 claims within verdicts | DEFECT | claims review: abstract does not state E1 INCONCLUSIVE or T1 FAIL (`abstract.tex:7-9`); strict count 24 PASS, 5 DEFECT, 15 UNKNOWN of 44 | claims review (agent a8eda4d150ff85bfd) |
| M3 numbers traced | PENDING | numbers review running (agent a41c178ab7d4e8b12) | — |
| M4 sources hashed | DEFECT (provisional) | 77 of 267 cited sources verifiable; 190 missing. Caps at 49 unless the text states those 190 are unavailable (not yet checked) | hashes review (agent a217b0316cf468e37) |
| M5 skeptical review | DEFECT | the claims review is an independent review with an open major (`abstract.tex:7-9`) | claims review |

Cap: 49 (any gate DEFECT). The cap is set by M2 and M5 regardless of the pending numbers review.

**Reconciliation.** The 2026-11-19 stage-10 record scored M2 PASS ("reviewer B: every abstract sentence supported at no greater
strength, after the E1 scope fix"). The 2026-11-25 claims review finds the abstract still omits E1 INCONCLUSIVE and T1 FAIL. The two
records cannot both hold for the abstract at the same state. The 2026-11-25 review is on the current text; the stage-10 record
was scored at d289583. Resolve by checking the abstract at HEAD against both verdicts. Until then M2 stays DEFECT (the stricter claim).

## 2. Criteria (provisional, raw; each review's own item unit)

| # | Weight | Items PASS / total | Raw score | Unit and source | DEFECT or UNKNOWN |
|---|---|---|---|---|---|
| C1 numerical accuracy | 25 | PENDING | PENDING | numbers review | PENDING |
| C2 calibration | 20 | 24 / 44 | 10.9 | claims review, abstract/results/discussion sentences | 5 DEFECT, 15 UNKNOWN |
| C3 statistics | 15 | 31 / 45 | 10.3 | claims review, also reports C3 (cells) | 8 DEFECT, 6 UNKNOWN |
| C4 reproducibility | 15 | 2 / 6 | 5.0 | hashes review | C15, C16 cite "this revision" without a hash (DEFECT); data access UNKNOWN (stage 13) |
| C5 consistency | 10 | PENDING | PENDING | numbers review | PENDING |
| C6 figures | 5 | 5 / 12 | 2.1 | figure items with a producer | 7 without a producer; stage-10 inspection found 6 DEFECTs |
| C7 methods | 5 | 9 / 16 | 2.8 | methods items | 7 not PASS |
| C8 prose | 5 | not reviewed this round | — | `match-writing-style` pass not run | stage-10: 6 unasked contrasts in discussion, 1 in introduction (DEFECT); parentheses rule UNKNOWN |

Raw sum of the known rows: 30.3 (C2, C3, C4, C6, C7). Not a score: C1 (25) and C5 (10) are pending, C8 has no current count,
and the item units differ by review. The capped reading is at most 49 whatever the pending rows are.

Coverage (weight whose items were all checked): not computable yet. C2 is 66% checked (29 of 44 items resolved as PASS/DEFECT);
the others are not fully checked.

## 3. Every DEFECT (from the reviews, with source)

| # | Gate or criterion | Finding | Location | Repair |
|---|---|---|---|---|
| D1 | M2, M5, C2 (major) | Abstract omits E1 INCONCLUSIVE and T1 FAIL | `abstract.tex:7-9` | State both verdicts in the abstract. Needs owner ruling on wording (open decision). |
| D2 | C2 | Causal wording for rho | `results.tex:45`, `:53` | Reword to the tested association |
| D3 | C2, C5 | Crossing reported as all seeds; it holds in 2 of 3 | `results.tex:99` | State "2 of 3 seeds" |
| D4 | C3 | Unmatched comparison of r_ab: 8 s vs 12 s, different configuration | `results.tex:249` | Match the configuration or state the mismatch |
| D5 | C2, C8 | Hedge missing where the claim is inferred | `results.tex:401-402` | Add the hedge |
| D6 | C3 | T8 rule at n = 3, uncorrected; "reference higher" wording | `methods.tex:313` | Report n, correct or state uncorrected |
| D7 | C4 | C15, C16 cite "this revision" without a hash | `methods.tex`, records table | Insert the commit hash |
| D8 | M4, C4 | 190 of 267 sources unverifiable by hash | records table | Hash them, or state unavailability in the text (check first) |
| D9 | C5 | Loss attributed differently in two places | `results.tex:238-239` vs `discussion.tex:11-12` | Make the attribution agree (D 21 ms level, not fresh seeds) |
| D10 | C6 | 6 figure DEFECTs at stage-10 inspection | figure producers | Repair per stage-10 list (not re-inspected here) |
| D11 | C7 | 4 methods DEFECTs, 3 minors (stage-10 inspection) | `methods.tex` | Repair per stage-10 list (not re-inspected here) |
| D12 | C8 | Unasked contrasts: 6 in discussion, 1 in introduction | discussion, introduction | Remove (style rule) |
| D13 | C3 | Sidedness UNKNOWN; effect sizes without intervals (ranges) | `results.tex` | Add intervals and sidedness |
| D14 | C1 | Permutation count 13 824 (`results.tex:53-54`) | — | Check against the source (numbers review) |

## 4. Prioritized repairs

1. D1: abstract verdicts (gate M2, cap 49). Owner decision on wording first; the agent does not edit the abstract unasked.
2. D8 and M4: hash the 190 sources, or add the unavailability statement (gate M4).
3. D7: commit hashes for C15 and C16 (C4).
4. D2 to D6, D9 (C2, C3, C5): text repairs, each checked against its source.
5. D10, D11: figure and methods repairs (C6, C7), re-inspected after repair.
6. D12: prose (C8).
7. D14 and the pending numbers review (C1, C5, M3).

## 5. Open items

- Owner: abstract wording (D1); whether the agent may edit the abstract.
- Verify: M1 requires a TeX run; not available on this host.
- Reconcile: M2 PASS (2026-11-19) vs DEFECT (2026-11-25), section 1.
- Pending: numbers review (C1, C5, M3); its results decide the rows marked PENDING.
- Not checked this round: C8 counts, D10, D11 (carried from stage-10, not re-inspected).
