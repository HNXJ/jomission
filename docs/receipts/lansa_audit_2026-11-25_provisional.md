# LANSA consolidated audit, provisional (2026-11-25)

Status: PROVISIONAL diagnostic. Not the frozen rubric's score and not an accepted result. Rubric: `paper/jomission_lansa/quality_rubric.md`
(fixed 2026-11-19, not modified). Item units: stated per row; the frozen rubric has no item list, so units are a choice, flagged here.
Reviews: claims (agent a8eda4d150ff85bfd), hashes (a217b0316cf468e37), numbers (a41c178ab7d4e8b12). All three completed.

## 1. Gates

| Gate | State | Basis | Source |
|---|---|---|---|
| M1 build clean | UNKNOWN | no TeX engine on this host; stage-10 record says PASS at d289583 (35 pages) | `STAGE10_R2_VERIFICATION.md` gates table |
| M2 claims within verdicts | DEFECT | abstract does not state E1 INCONCLUSIVE or T1 FAIL (`abstract.tex:7-9`); strict 24 PASS, 5 DEFECT, 15 UNKNOWN of 44 | claims review |
| M3 numbers traced | UNKNOWN | 1343 of 1391 data-bearing tokens PASS, 48 UNKNOWN, 0 numeric DEFECT by reviewer count. About 480 PASS tokens trace only to records absent from this repo at HEAD (section 2, D16) | numbers review |
| M4 sources hashed | DEFECT | (a) HEAD: 77 of 267 verifiable, 190 missing (hashes review). (b) Canonical checkout `/mnt/d/cowork/jomission`: 198 of 267 match, 66 missing, 3 differ. (c) `details.json` recorded as `9b4f66ec…`, on-disk `e3a7b697…` (observed) | hashes review; numbers review |
| M5 skeptical review | DEFECT | independent claims review has an open major (`abstract.tex:7-9`) | claims review |

Cap: **49** (M2, M4, M5 are DEFECT). M3 UNKNOWN alone would cap at 89. The cap is 49 whatever the other rows say.

**Reconciliation 1 (M2).** The 2026-11-19 stage-10 record scored M2 PASS ("reviewer B: every abstract sentence supported at no greater
strength, after the E1 scope fix"). The 2026-11-25 claims review finds the abstract still omits E1 INCONCLUSIVE and T1 FAIL. Both cannot hold
for the abstract at one state. Check the abstract at HEAD against both verdicts; until then M2 stays DEFECT (the stricter reading).

**Reconciliation 2 (M4 counts).** 77 of 267 (HEAD, hashes review) and 198 of 267 (cowork, numbers review) are not contradictory: they
measure different trees. HEAD 3f12e22 and cowork 21b2697 differ, and cowork holds 51 dirty paths. A reader of this repo alone cannot
verify the 198.

## 2. Criteria (provisional, raw)

| # | Weight | Items PASS / total | Raw | Unit | DEFECT or UNKNOWN |
|---|---|---|---|---|---|
| C1 numerical accuracy | 25 | 1343 / 1391 tokens | 24.1 | numeric tokens (numbers review; token unit, not the stage-10 group unit) | 0 DEFECT by reviewer count; 48 UNKNOWN; D16 (1 strict DEFECT) |
| C2 calibration | 20 | 24 / 44 | 10.9 | claims sentences (claims review) | 5 DEFECT, 15 UNKNOWN |
| C3 statistics | 15 | 31 / 45 | 10.3 | cells (claims review) | 8 DEFECT, 6 UNKNOWN |
| C4 reproducibility | 15 | 2 / 6 | 5.0 | items (hashes review) | C15, C16 no hash (DEFECT); data access UNKNOWN (stage 13); records absent at HEAD (D16) |
| C5 consistency | 10 | 5 / 6 groups | 8.3 | groups (numbers review C5 table) | 1 DEFECT (D15); C9, C17 defined but not cited (minor) |
| C6 figures | 5 | 5 / 12 | 2.1 | figure items with a producer (hashes review) | 7 not traced to a producer; stage-10 found 6 DEFECTs |
| C7 methods | 5 | 9 / 16 | 2.8 | methods items (hashes review) | 7 not PASS |
| C8 prose | 5 | not scored | — | `match-writing-style` pass not run this round | stage-10: 6 unasked contrasts in discussion, 1 in introduction |

Raw sum of scored rows: **63.5** (C8 excluded). Capped reading: **49**. These raw values use different units and are not a publishable score.

**Coverage** (weight whose items were all checked): C5 only (10 of 100). C1 has 48 UNKNOWN tokens; C2, C3, C4, C6, C7 have UNKNOWN or unchecked items; C8 is unchecked.

## 3. Every DEFECT and UNKNOWN (with source)

| # | Gate or criterion | Finding | Location | Repair |
|---|---|---|---|---|
| D1 | M2, M5, C2 (major) | Abstract omits E1 INCONCLUSIVE and T1 FAIL | `abstract.tex:7-9` | State both verdicts. Owner ruling on wording first. |
| D2 | C2 | Causal wording for rho | `results.tex:45`, `:53` | Reword to the tested association |
| D3 | C2, C5 | Crossing reported as all seeds; holds in 2 of 3 | `results.tex:99` | State "2 of 3 seeds" |
| D4 | C3 | Unmatched r_ab comparison: 8 s vs 12 s, different configuration | `results.tex:249` | Match configuration or state the mismatch |
| D5 | C2, C8 | Hedge missing where claim is inferred | `results.tex:401-402` | Add the hedge |
| D6 | C3 | T8 rule at n = 3, uncorrected; "reference higher" wording | `methods.tex:313` | Report n; correct or state uncorrected |
| D7 | C4 | C15, C16 cite "this revision" without a hash | methods, records table | Insert commit hash |
| D8 | M4, C4 | 190 of 267 unverifiable at HEAD; 66 missing and 3 differ at cowork | records table | Hash the sources in this repo, or state unavailability (check first) |
| D9 | C5 | Loss attributed differently | `results.tex:238-239` vs `discussion.tex:11-12` | Make them agree (D 21 ms level, not fresh seeds) |
| D10 | C6 | 6 figure DEFECTs at stage-10 inspection | figure producers | Repair; re-inspect (not re-inspected this round) |
| D11 | C7 | 4 methods DEFECTs, 3 minors at stage-10 | `methods.tex` | Repair; re-inspect (not re-inspected this round) |
| D12 | C8 | Unasked contrasts: 6 discussion, 1 introduction | discussion, introduction | Remove |
| D13 | C3 | Sidedness UNKNOWN; effect sizes without intervals | `results.tex` | Add intervals and sidedness |
| D14 | C1 | Permutation count 13 824 | `results.tex:53-54` | Not rechecked this round; check against source |
| D15 | C5 (DEFECT) | Text says "level maps in Fig.~\ref{fig:s-ladder}"; that label is the finite-size ladder maps. The T1 level maps are `fig:t1` | `results.tex:236`; `supplement.tex:54`, `:65` | Change the reference to `fig:t1`. Then `fig:s-ladder` has its referrer |
| D16 | M3, C1, C4 (source) | About 480 PASS tokens (T1, ladder, weights, depth, probe, seeds, S2) depend on run records R3–R10, R12 absent from HEAD. Traced only to `/mnt/d/cowork/jomission` | run records | Decision: commit the records to this repo or seal them. Owner call |
| D17 | M4 (DEFECT) | `details.json` hash in `sources_sha256.txt` is `9b4f66ec…`; on-disk file in cowork is `e3a7b697…`. V0 and block PNGs also differ. S1–S4 numbers still match `details.md` | `sources_sha256.txt`; cowork `details.json` | Re-hash after deciding which file is authoritative |
| D18 | C1 (minor) | "about 356": 0.140008 × 2547 = 356.6, which rounds to 357 | `results.tex:98` | Use 357 or "about 357" (strict DEFECT; reviewer counted PASS) |
| D19 | C1 UNKNOWN | Dominance 87–100%, 100%, 69–89%: no per-frequency unit spectra on disk | `results.tex:332` (5 tokens) | Recompute from T8 `.npz` or add a record |
| D20 | C1 UNKNOWN | "identity checked to 10^-9": no file holds it | `results.tex:117` (1 token) | Add a record or remove the number |
| D21 | C1 UNKNOWN | Label sizes 6.5–7.0 pt and 2.6 pt | `supplement.tex:15` (3 tokens) | Trace to the figure source |
| D22 | C1 UNKNOWN | S4 neuron table: JaxFNE ADEX defaults, not traced to the pinned source | `details` S4 table (31 tokens) | Trace to the pinned `jaxfne` source |
| D23 | C1 UNKNOWN | 8 tokens in FINDINGS only (12.9x M/D power ratio, M/S gamma 0.25, M E 0.25 Hz, CSD consistency 0.979, 108 troughs, rank 9 of 9). The CSD run has no JSON in `results/viz/paper_w4` | `docs/laminar_motif/FINDINGS_2026-11-10.md` | Record the run or drop the numbers from the paper |
| D24 | C5 minor | `FINDINGS_2026-11-10.md:166` says "M E 36 Hz in a 2 s smoke"; paper (`results.tex:277`) says 54. Paper is right (54.48 Hz over 0–2 s; 36.4 Hz over 0.5–2 s, per commit 8464e1f) | findings note | Correct the findings note. Not a manuscript defect |
| D25 | C5 | 93-run and 90-distinct counts in scatter caption are arithmetic only (9+54+3+12+15 = 93, minus 3 T8 duplicates = 90); no record lists them | scatter caption | Add a record or cite the arithmetic |

Items not rechecked this round: D10, D11, D14, C8 counts.

## 4. Prioritized repairs

1. D1: abstract verdicts (M2, cap 49). Owner wording decision first; the agent does not edit the abstract unasked.
2. D17 and D8: settle the authoritative `details.json`; re-hash; add unavailability statements where a source is missing (M4).
3. D16: owner decision on committing or sealing the run records R3–R10, R12. Without this M3 and C4 cannot pass from this repo.
4. D15 and D18: the figure reference and the rounding (C5, C1). Both are one-line fixes.
5. D7: commit hashes for C15, C16 (C4).
6. D2 to D6, D9, D13: text repairs, each checked against source (C2, C3, C5).
7. D19 to D23: trace or drop the UNKNOWN numbers (C1).
8. D10, D11: figure and methods repairs (C6, C7), re-inspected after repair.
9. D12, C8 pass: prose.

## 5. Open items

- Owner: abstract wording (D1); whether the agent may edit the abstract.
- Owner: commit or seal the run records R3–R10, R12 (D16).
- Owner: which `details.json` is authoritative (D17).
- Verify: M1 requires a TeX run; not on this host.
- Reconcile: M2 PASS (2026-11-19) vs DEFECT (2026-11-25), section 1.
- Repo note: `docs/laminar_motif/FINDINGS_2026-11-10.md:166` is stale (D24); not a manuscript defect.
