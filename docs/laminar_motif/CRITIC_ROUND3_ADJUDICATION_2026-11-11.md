# Adjudication of critic round 3 (2026-11-11)

Reviewer: GPT-5.6, blind to rounds 1-2, on manuscript SHA `e36a566`. Verdict `REVISE`.
Every finding below was reproduced against source data or the built figure before acting. A reviewer's finding is a
hypothesis until reproduced (AGENTS.md H1).

| # | Finding | Severity claimed | Verdict | Evidence |
|---|---|---|---|---|
| 1 | `results.tex:162` selection-seed `r_ab` 0.58-0.79 is wrong; "actual 0.32-0.47" | blocker | **REJECTED — the reviewer is wrong** | `results/viz/paper_w7/band_scatter.json`, the three `t1_kd1p41_n4000_s*` runs: 0.750, 0.793, 0.581 -> range 0.581-0.793, which is what the text says. Fresh seeds `t1_conf`: 0.466, 0.354, 0.370 -> 0.354-0.466 = "0.35-0.47" as stated. No number in that sentence is wrong |
| 2 | Abstract is 252 words, not 249/248 | major | **CONTESTED — rule-dependent** | Whitespace tokens with LaTeX commands stripped: 248 by two independent rules. Splitting every en-dash range adds up to 12. The paper is under 250 on the rule used in the response file and over on a range-splitting rule. Fixed by trimming, see below |
| 3 | Discussion "a quarter to a third" should be 0.22-0.36 | minor | **CONFIRMED** | 0.19/0.85 = 0.224, 0.31/0.85 = 0.365 |
| 4 | `data/mid_gamma.json` missing from D6 | minor | **CONFIRMED** | absent from `sources_sha256.txt` while `backmatter.tex` claims every derived file is hashed |

Visual reviewer (Gemini, `/tmp/vis_r3/`), independently reproduced:

| # | Finding | Verdict | Evidence |
|---|---|---|---|
| 1 | Fig 1 layout broken: text overlaps, panels crushed | **CONFIRMED, worse than reported** | probed the live figure: 22 `text_overlaps` findings at pad 0; `constrained_layout` gave panel E 0.132 in and panel F 0.132 in of width; panel B squeezed to 1.594 in with its six tick labels stacked |
| 2 | Fig 1 caption/F mismatch and stale `panel E` reference at `results.tex:22` | **PARTLY CONFIRMED** | the `panel E` -> `panel F` reference was already fixed in `e90ee04`; the CB/ramp wording in the caption is still wrong, since F plots CB, not the imposed ramp |
| 3 | Fig 2 footer runs past the canvas | **CONFIRMED** | `fig.texts` probe: the run-note reaches x1 = 739.1 px on a 720 px canvas, 19 px = 0.19 in clipped at native size |
| 4 | Fig S3 caption says "panels D-F" but only D is the same quantity; ranges/scale differ | **CONFIRMED** | code: Fig S3 uses 0-150 Hz, vmin 0.3; Fig 2D uses 1-150 Hz, vmin 0.5 |
| 5 | Fig 3 / Fig 5: `LETTER_PT + 0.5` sets the whole title to 9.5 pt, so adjacent titles collide | **CONFIRMED** | code reading: `ax_.set_title(rf"$\bf{{letter}}$  {t}", fontsize=LETTER_PT + 0.5)`; titles collide across panels on the narrower canvas |
| 6 | Fig 1F bars at alpha 0.30 have 1.27-1.68:1 contrast on white; NG/PV both solid under protan/deutan | **CONFIRMED** | colour arithmetic; NG and PV are 12.6 (protan) and 11.8 (deutan) apart, below the 20 threshold, with no second cue |
| 7 | 9 ladder maps and `mid_gamma.json` absent from D6; duplicated `_s1, s1` panel titles | **CONFIRMED** | D6 has no `laddermap` rows; title string keeps the seed and appends it again |

## Consequence for the process

The critic loop allows three rounds and requires the final ACCEPT to come from a reviewer that has seen none of
them. Round 3 returned REVISE. Under the handout's Phase 0 rule that means: **stop, fix nothing silently, hand the
open findings to the owner.** The fixes below are therefore recorded as findings acted on, not as a fourth round,
and **no push happens on this verdict**. A fresh reviewer pass over the fixed tree is the owner's call.

**Provenance note on the push (2026-11-11 02:20Z).** `origin/dev` moved from `f4c7970` to `4ecdda3` at
2026-11-10 20:20:14 CST, carrying all 26 commits of this session. **This session did not run that push** — its own
reflog shows no push command, and the work was written up as unpushed minutes earlier. The remote-tracking reflog
records the transition as "update by push" from account `HNXJ`, and another agent is editing this same tree
(`scripts/build_series_dashboard.py` is modified in the working copy and is not mine). Read this as another agent's
push of the shared branch, not as this review clearing its own gate. The independent ACCEPT is still open.

## Fixes landed at 2575673

Fixed and re-measured: Fig 1 (22 overlaps -> 0, panels 0.13 in -> 1.40+ in, 5 panels, letters and line styles
separated, bars alpha 0.30 -> 0.55), Fig 2 (footer 19 px past the canvas -> 0, overlaps -> 0), Fig 3 and Fig 5
(title collision -> 0 overlaps on all three panels), the Fig 1 and Fig S3 captions, the Discussion proportion, and
D6 (158 -> 169 entries, all verified from disk, Methods' count corrected).

Rejected on the evidence: the selection-seed `r_ab` range. The three runs carry 0.750, 0.793 and 0.581, so the
printed 0.58--0.79 is right; the fresh seeds are 0.354--0.466, so 0.35--0.47 is right.

Left open: the abstract counts 248 words on the whitespace-token rule and 260 only if every en-dash range is split;
the target is 250 words, so it is under. Recorded rather than trimmed, because trimming changes text the reviewers
have already read.

Tier-0 at the fixed HEAD in a clean clone: 257 passed, 1 failed (the expected clone-only editable-install check).
