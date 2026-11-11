# Handout: jomission-LANSA to 100/100 (2026-11-10)

From: Claude session (owner Hamm) to the Cursor agent that continues the work. Supersedes
`HANDOUT_GEMINI.md` (2026-11-07, Windows paths, pre-E1 state) for this lineage.

Read `AGENTS.md`, then this file, then the files it names. Every statement here is a pointer, not evidence:
re-read the source before you build on it. Status marks: OBSERVED (file or run you can open), INFERRED,
PROPOSED (needs Hamm).

## 1. Where things are

| What | Where |
|---|---|
| Checkout | `/Users/hamednejat/workspace/dev/jomission`, branch `dev`, remote `origin` (SSH). Other agents work in this tree: never stage, stash or revert their files (e.g. `paper/jomission_lansa/deck/` is another session's) |
| Python | `/usr/local/bin/python3` (3.13); JaxFNE 0.5.8 installed (execution authority); T1 axis 3, confirmation, weights and T8 ran on 0.5.9 at 351f6048 (tag v0.5.9, Zenodo 10.5281/zenodo.21448155) |
| Manuscript | `paper/jomission_lansa/`: `main.tex`, `sections/*.tex`, `refs.bib`, plan `main.md`, build `latexmk -pdf -interaction=nonstopmode main.tex` (gate: 0 undefined, 0 overfull, no "Float too large") |
| Evidence | Table S6 in `sections/supplement.tex` maps every C/S/D/R/N id to a file; `data/sources_sha256.txt` (D6) hashes every cited file (159 at `e6b33ac` and later; re-count it, the number moves as figures are added); findings log `docs/laminar_motif/FINDINGS_2026-11-10.md` |
| Seals | `docs/laminar_motif/SEAL_*.md` (E0, E1, LFS, T1, T2_T3, T4, T8). Sealed-not-run: T2, T3, T4 |
| Reviews | `paper/jomission_lansa/response_sol_2026-11-10.md`, `response_review4_2026-11-10.md`; `docs/reviews/JOMISSION_LANSA_PREFLIGHT_2026-11-10.md` |
| Proposal, not authorized | `docs/designs/OMISSION_SPECTROLAMINAR_HDP_PROPOSAL_2026-11-10.md` (unsealed; run nothing from it) |
| Project state | `manifests/todo.json` (lineage `SCI-LAMINAR-MOTIF-1`, OPEN), `manifests/current_state.json`, `docs/OPEN.md` (OPEN-36..42 are the laminar rows) |

## 2. Rules that bind you (from AGENTS.md and the owner's global rules)

- Start: `python3 scripts/project_check.py` (full needs network; quick: `--offline --no-tests`).
- Seal before running: question, estimand, arms, seeds, outcomes, decision rule and stop states committed and pushed
  before the first run. No retuning after results; a changed criterion is a new lineage.
- Visualization contract per model lineage: V0 schematic, V1 1 s raster before any score is read; V2, V3 before
  calling a lineage complete (`manifests/visualization_contract.json`).
- Tier-0 before a code push: run `project_check.py` in a `git clone --shared` of HEAD through
  `python3 ~/.claude/bin/suite_lock.py --label jomission -- python3 scripts/project_check.py`, in the background.
  In a clone, `test_jomission_import_resolves_to_this_checkout` fails by construction (editable install points at
  the main tree); every other failure is real. Tracked `results/**/*.json` must be registered
  (`test_results_coverage_registered`): keep derived plot JSONs untracked or register them.
- Push `dev` after the gates pass and an independent reviewer (another model, never the writer) accepts. Never push
  `main`, tag, publish, deposit on Zenodo or post to bioRxiv without Hamm's explicit go for that SHA.
- Stage exact paths; no `git add .`, no force push, no hard reset. Commit trailer per the repo's history.
- Manuscript: never generate declarations (authors, funding, conflicts, contributions, data and code availability,
  AI use): mark `[MISSING: ...]`. Cite by DOI or Table S6 id, never by path in the text. Abstract <= 250 words.
- Critic loop for the manuscript: at most 3 rounds; the final ACCEPT from a reviewer that has not seen earlier rounds.
- Shell: write scripts to files and run them by path; no `sed -i`; kill by PID only.

## 3. State on 2026-11-10 (OBSERVED unless marked)

- Handoff point, 2026-11-11T01:03Z (2026-11-10 19:03 CST): `dev` is 4 commits ahead of origin/dev (f4c7970), nothing pushed: 12fc805
  (revision), 2c51ab3 (critic round 1 fixes), 5aa809f (critic round 2 fixes and this handout), e36a566 (the state
  section this commit corrects). **Manuscript content to review = `e36a566`**; this doc-only update does not change
  it. Run `git log origin/dev..dev` and `git diff e36a566 HEAD -- paper/` before acting: an empty paper diff means
  the round-3 review still binds `e36a566`.
- **Scope warning for the round-3 verdict (2026-11-11).** Round 3 was dispatched against `e36a566`, but A3 work
  landed while it ran, so the paper has moved: `git diff e36a566 HEAD -- paper/` shows `results.tex` (two captions),
  `supplement.tex` (Fig S3 sources and caption), `data/sources_sha256.txt` (six re-hashed PNGs) and the response
  file. An ACCEPT therefore covers the text as of `e36a566`; the caption and figure-source edits after it are
  mechanical (panel description, figure path, hash) and are named in their commits, but a strictly clean ACCEPT for
  the shipped SHA needs either a reviewer pass over that diff or Hamm's acceptance of the split. Do not push on the
  round-3 ACCEPT alone without recording this.
- Critic rounds: 1 on 12fc805 REVISE (fixed), 2 on 2c51ab3 REVISE (fixed: dense runs' engine 0.5.8 / ref s1 0.5.9;
  selection vs fresh-seed r_ab wording; r_gm bound 0.1776 vs 0.1778). **Round 3 on `e36a566` returned REVISE**, by a
  reviewer that saw neither earlier round; the findings were reproduced and 8 of 9 fixed at `2575673`, one rejected
  on the evidence, and the abstract count left as is (see `CRITIC_ROUND3_ADJUDICATION_2026-11-11.md`). The loop's
  three rounds are spent, so the next ACCEPT must come from a fresh reviewer pass on the fixed tree, which is the
  owner's call. **Nothing is pushed.**
- Tier-0 at 2c51ab3 in a clone: 257 passed, 1 failed (the expected clone import check). 5aa809f, e36a566 and every
  later commit changed only tex, docs, figure PNGs and figure scripts, so that result still binds HEAD; rerun tier-0
  before pushing if code changes. Rerun twice anyway this session (at `458fb2c` and at `2575673` after the round-3
  fixes): **257 passed, 1 failed each time**, the failure being
  `test_environment_provenance.py::test_jomission_import_resolves_to_this_checkout`, which fails in a clone by
  construction.
- Manuscript build at HEAD: `latexmk -pdf` clean, 0 undefined references or citations, 0 overfull boxes, 0
  "Float too large", 35 pages (`main.pdf` 18:58:09 postdates the last tex write at 18:58:02; both untracked).
- Results: no run passes the six-part scorer (ladder 9, T1 54 + 3 fresh-seed, weights 12, T8 15). Reference column:
  r_ab 0.83-0.86 on seeds 1-3 (never rerun on fresh seeds); mid-depth relative gamma 0.19-0.31 vs 0.85 observed.
  T1 fails on fresh seeds. T8: density changes let D dominate, SOM/PV changes let S dominate; no band-specific effect.
- Readout: capacity 0.853 / 0.657 at sigma 0.15; joint optimum r 0.845 at mid 0.828 is a numerical solution
  (known-answer and rank-one grid checks, `data/readout_joint_check.json`); circuit puts 17-19% of gamma power in M
  vs 49% at the optimum. Depth placement alone raises r_proto 0.38 -> 0.72, null not rejected (p ~ 0.19).
- Manuscript: 35 pages, review-4 revision (Results reordered, Fig 4 scatter, Discussion 941 words). Reviewer
  editorial scores for the previous draft: rigor 88, clarity 79, naturalness 76, bioRxiv readiness 82.
- Owner items outstanding: all declarations, LANSA access terms, run-record archive, Zenodo for jomission.

## 4. What 100/100 means here (PROPOSED: Hamm confirms or edits the criteria)

Ten criteria, 10 points each: met 10, partial 5, open 0. Current status is the writer's assessment (INFERRED);
this scale is for the research programme, not the bioRxiv-readiness score above.

| # | Criterion | Now | Closes with |
|---|---|---|---|
| Q1 | Every number traces to a hashed receipt; figures and text agree | met (10) | keep: D6 regenerated per change |
| Q2 | Each headline claim confirmed on fresh seeds under a seal | partial (5) | phase B1 (reference on fresh seeds), B2 |
| Q3 | The leading mechanism (fast mode in M, PV kinetics) tested causally with a rate-matched control | open (0) | phase B2 (T9) |
| Q4 | Drive matched to the target (peri-stimulus, layer-specific input) | open (0) | phase C1 |
| Q5 | Readout checked against an independent forward model | partial (5) | phase C2 |
| Q6 | Comparisons powered (seeds per arm, multiplicity correction declared) | partial (5) | phase B3 |
| Q7 | Density and SOM/PV separated from rates and in-degrees; more depth units | partial (5) | phase C3, C4 |
| Q8 | Public, frozen archive (code, run records, environment, DOI, checksums verified from the archive) | partial (5) | phase A2 |
| Q9 | Manuscript complete: declarations, figures legible at print size, critic ACCEPT | partial (5) | phase A1, A3 |
| Q10 | Posted, externally reviewed, reviews answered with evidence | partial (5) | phase A4, D |

Total now 45/100 (INFERRED). 100 = every row met.

## 5. Plan, in order (top = next). Each step: what, closing check, blocked by

### Phase 0: close the review-4 round (this session may already have done it)
0. **DONE to the limit the loop allows.** Round 3 ran on `e36a566` under a reviewer that saw neither earlier round
   and returned REVISE. Its findings were reproduced: 8 were real and are fixed at `2575673` (Fig 1 layout and 22
   overlaps, Fig 2 footer clipping, Fig 3 caption/F mismatch, Fig S3 caption mismatch, Fig 3/5 title collision,
   Fig 1 CB-vs-ramp wording, the Discussion proportion, and two D6 omissions); 1 was rejected on the evidence (the
   selection-seed `r_ab`, see the adjudication file). The round-3 findings that remain and the fact that the loop's
   three rounds are spent mean **the next ACCEPT needs a fresh reviewer over the fixed tree, and that is Hamm's
   call, not the agent's.** Do not push on the round-3 verdict.
   Check at the point this is picked up: `git log origin/dev..dev` and `python3 scripts/project_check.py` in a
   clone; the build is 35 pages with 0 undefined, 0 overfull.

### Phase A: preprint (bioRxiv), no new simulation
A1. Declarations and metadata from Hamm into `main.tex` and `sections/backmatter.tex`. Check: `grep -c MISSING
    sections/*.tex main.tex` = 0. Blocked by: Hamm.
A2. Archive: run records R1-R11 plus code at a frozen SHA to Zenodo (or Hamm's choice), DOI into Data and Code
    availability; verify every D6 hash from the downloaded archive. Check: script re-hashes the archive, 0 mismatches.
    Blocked by: Hamm's go to publish; LANSA access terms from Hamm.
A3. Figures: redraw Fig 1E (split built counts from LANSA profiles) and Fig 2C (fewer traces, larger band labels);
    supplement layout (Table S6 font, ladder maps). Check: render each page at 150 dpi, every label >= 6 pt at print
    size, independent visual review ACCEPT.
    Print-scale floors: DONE 2026-11-11 at `5a89597`. Every label now clears its floor, measured at the 489.10 pt
    text width (6.7677 in) and confirmed by rasterizing the figure pages of the rebuilt PDF: Fig 1 canvas 7.2 ->
    6.7677 in (scale 1.000) giving ticks 7.00, legend 6.50, letters 10.00 pt; Fig 3 canvas 6.8 -> 6.7677 in with
    panel D 6.0 -> 6.5 pt and letters 8.0 -> 9.5 pt, giving 7.00 / 6.50 / 9.50; Fig 2 keeps its 7.2 in canvas
    (absolute-inch layout, COL_X + AX_W = 6.9 in would overflow) and carries FONT_SCALE = 7.2/6.7677 in
    `TEXT_PT`, `LABEL_PT`, `LETTER_PT`, with the SOM in-cell label at 6.9 pt native, giving 6.11 / 6.58 / 6.10 /
    9.02 pt; Fig 4 unchanged at 6.0-7.0 pt. Table S6 was measured, not assumed: it sets `\scriptsize` and the PDF
    page 35 embeds LMRoman8 and LMMono8, i.e. 8 pt, clearing the floor, so it needs no change.
    Still open in A3: the Fig 1E *layout* change (split built counts from LANSA profiles into two panels) and the
    Fig 2C *content* change (fewer traces, larger band labels) were NOT done; only their font sizes were fixed.
    Ladder maps (Fig S3) unmeasured. Visual review by a second reviewer: not yet run.
A4. Final PDF review, push `main` for the named SHA, post to bioRxiv. Check: Hamm's go per SHA; bioRxiv DOI recorded
    in `main.md`. Blocked by: A1-A3, Hamm.

### Phase B: confirm and test the leading mechanism (each a new seal)
B1. Reference on fresh seeds 4-6 (12 s), same scorer; seal first. Check: per-seed verdicts; text updated whether
    r_ab holds or not. Cost: 3 runs (~140 s each on the event route at 8 s; 12 s longer).
B2. T9, rate-matched middle-unit gamma: raise M's fast mode by PV$\to$E kinetics (decay, as Herrera 2026 for L4) and,
    separately, PV$\to$PV coupling, with M's E rate held within a declared band by drive; control arm: rate change
    without the kinetic change. Primary outcome mid-depth relative gamma; secondary r_gm, r_ab, crossing. Prediction
    (Discussion): r_gm rises without breaking the crossing; the rate-only control does not. Check: sealed rule
    applied, V0-V3 produced, result reported whatever its sign.
B3. Power: rerun the T8 arms and B2 at >= 6 seeds with the correction declared in the seal (e.g. Holm over primary
    outcomes). Check: decision table from the sealed rule.
B4. Run the sealed-not-run T2 (E/I), T3 (SOM/PV scaling), T4 (silencing) only if B2 leaves the mechanism open;
    re-read their seals first (they predate T8). Blocked by: Hamm confirms they are still wanted.

### Phase C: remove the main confounds
C1. Peri-stimulus drive: time-locked input to M (thalamic) and to S/D (feedback) with declared latencies; score
    against LANSA peri-stimulus windows with matched windowing. Check: sealed comparison of stationary vs driven
    maps on the same column. Blocked by: Hamm on the input model (a scientific choice).
C2. Independent readout: score saved runs with population-rate spectra and with a multicompartment forward model
    on a subset (e.g. LFPy-style line sources per unit), identical spectral processing; report where the RWS-kernel
    map departs. Check: per-run agreement table; the mid-depth deficit present or not under each readout.
C3. Decouple construction: density change with E rates and in-degrees matched (drive or weight compensation), and
    SOM/PV change at fixed SOM$\to$E in-degree. Check: sealed arms; dominance and band outcomes reported.
C4. More units: 5-7 depth units from the same LANSA bins, so depth resolution comes from the model, not the kernel.
    Check: ladder-style rule for unit count; scorer unchanged.

### Phase D: journal version
D1. Merge phases B-C into a revised manuscript; per-area fits (11 areas) as generalization; respond to bioRxiv
    comments. Check: critic loop ACCEPT; Hamm chooses the journal (instructions read on the date of submission).

## 6. Decisions only Hamm makes (ask one at a time, options ranked)

1. Confirm or edit the 10 criteria in section 4.
2. Declarations and metadata (A1); LANSA access statement; archive host and publish go (A2); `main` push and
   bioRxiv go per SHA (A4).
3. Input model for C1; whether T2-T4 still run (B4); JaxFNE version authority (OPEN-37: 0.5.8 vs 0.5.9).
4. Whether the HDP proposal becomes a lineage (it is unsealed; nothing runs from it).

## 7. How to work

Loop per step: re-read sources -> smallest change -> its check -> independent review -> commit exact paths ->
push `dev` -> delete the finished row here and update `manifests/todo.json` progress. Report to Hamm: result first,
blockers and open items in a table. Surprising numbers are resolved before they are reported.
