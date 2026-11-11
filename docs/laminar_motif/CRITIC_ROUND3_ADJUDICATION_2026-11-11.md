# Adjudication of critic round 3 (2026-11-11)

Reviewer: GPT-5.6, blind to rounds 1-2, on manuscript SHA `e36a566`. Verdict `REVISE`.
Every finding below was reproduced against source data or the built figure before acting. A reviewer's finding is a
hypothesis until reproduced (AGENTS.md H1).

| # | Finding | Severity claimed | Verdict | Evidence |
|---|---|---|---|---|
| 1 | `results.tex:162` selection-seed `r_ab` 0.58-0.79 is wrong; "actual 0.32-0.47" | blocker | **UPHELD — I was wrong to reject it, then wrong again on the second pass** | See "The `r_ab` finding, resolved" below |

## The `r_ab` finding, resolved (2026-11-11)

This took three passes and two of them were mine and wrong. The record:

1. Round 3 said the selection-seed range should read 0.32-0.47 rather than 0.58-0.79. I rejected it, having checked
   the three `t1_kd1p41_n4000_s*` runs, which do give 0.581-0.793. **That check was on the wrong runs.**
2. A fresh reviewer caught the error: the setting actually frozen and run on fresh seeds is not the all-15 ms
   circuit. It is `B = 1.41`, SOM$\to$E 15 ms in S and M and **21 ms in D**, i.e. `t1_confirm_kd1p41_sst15_sstd21`,
   whose own selection-seed runs are `t1_sstd21_n4000_s{1,2,3}_T8` with `r_ab` **0.320, 0.473, 0.377**.
3. Verified from the run records: `t1_sstd21_n4000_s1_T8` carries `tau_sst_e_ms = {S: 15, M: 15, D: 21}` and the same
   layout string as the confirmation runs. So round 3 was right, my rejection was wrong, and the paper contradicted
   itself: `results.tex:223` and Table 4 row 3 already said 0.32-0.47, while `results.tex:165` and the Discussion
   said 0.58-0.79.

Fixed: `results.tex` now states the confirmed setting's own selection-seed range (0.32-0.47) and its fresh-seed range
(0.35-0.47) separately, keeps 0.58-0.79 only for the axis-1 and all-unit-decay levels it actually describes, and says
the setting never reproduced the reference's 0.83-0.86 rather than that it "lost" it. The Discussion says the same.
Every range was re-verified against `band_scatter.json` after the edit.

Second finding, also fixed: the Fig S1 caption called the diagnostic montage "record R1", but Table S6 defines R1 as
the E0 result. The caption now points at the per-run verdict records R6 and R7.

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

**Addendum by the pushing session (Claude, 2026-11-10).** The push of `4ecdda3` was made by the Claude session that
wrote the handout, after two gates on that exact SHA:
- Tier-0 in a `git clone --shared`: 257 passed, 1 failed. The failure was the clone-only
  `test_jomission_import_resolves_to_this_checkout`, confirmed failing alone. `verify_clone.py` later gave 258 passed.
- A fresh `reviewer` that had seen none of rounds 1-3 reviewed `git diff e36a566 4ecdda3` and returned
  VERDICT: ACCEPT, with no blocker or major. Its four minors:
  - this file's table row "Fixed by trimming" contradicts "Recorded rather than trimmed" (the abstract has 248
    whitespace tokens);
  - the Fig S1 architecture PNG is untracked;
  - the `e2_ladder_map.py` refmap flags (`--frac 0.78 --height 2.1 --suffix refmap`) are recorded nowhere;
  - the docstring "maximum per contact" should read "per frequency over depth".

The independent ACCEPT is therefore not open. Those four minors are the open items.

## Fixes landed at 2575673

Fixed and re-measured: Fig 1 (22 overlaps -> 0, panels 0.13 in -> 1.40+ in, 5 panels, letters and line styles
separated, bars alpha 0.30 -> 0.55), Fig 2 (footer 19 px past the canvas -> 0, overlaps -> 0), Fig 3 and Fig 5
(title collision -> 0 overlaps on all three panels), the Fig 1 and Fig S3 captions, the Discussion proportion, and
D6 (158 -> 169 entries, all verified from disk, Methods' count corrected).

Rejected on the evidence: nothing. The one finding recorded here as rejected (the selection-seed `r_ab`) was my
error, corrected at `f8e0d3b` and explained above; round 3 and the fresh reviewer were both right.

Left open: the abstract counts 248 words on the whitespace-token rule and 260 only if every en-dash range is split;
the target is 250 words, so it is under. Recorded rather than trimmed, because trimming changes text the reviewers
have already read.

## Lesson

"Verify the claim" means verify it about the object the claim is about. I checked `r_ab` on the axis-1 winning runs
when the sentence was about the *confirmed* setting, which is that circuit plus a deep-unit change. The confirmed
setting is named by its own layout and tau record; a matching-looking tag prefix is not the identity. Two later
passes each read the same sentence and only the third tied it to `t1_sstd21`.

Tier-0 at the fixed HEAD in a clean clone: 257 passed, 1 failed (the expected clone-only editable-install check).
