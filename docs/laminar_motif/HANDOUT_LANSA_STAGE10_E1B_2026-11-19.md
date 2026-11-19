# Handout: LANSA stage 10 round 2 and E1b (2026-11-19, 20:07Z)

Repo HNXJ/jomission, clone `D:/jomission`, branch `_temp_e1b` (contains `_temp_lansa-stage10-r2`). Remote is the
recovery point; `git ls-remote origin _temp_e1b` must equal the local HEAD before work starts.

## Where it stands

- Stage 10 of the 15-stage workflow (jarness `rules/reference/manuscript-stages.md`), before the methodological
  lock. Hamm has not authorized the lock; do not ask for it until E1b is folded in.
- Round-2 report: `paper/jomission_lansa/STAGE10_R2_VERIFICATION.md` (rubric `quality_rubric.md`: score 51,
  coverage 20, cap 89 because M3 is UNKNOWN; M3 trace done since, 6 DEFECTs fixed, 8 UNKNOWN not holdable in a file).
- Done this round: Majors 1-3 per Hamm's rulings; the abstract E1 sentence scoped to E1 (Hamm's ruling); D6 at 212
  hashes; records C15-C18 pinned; claim trace tables in `docs/laminar_motif/m3_trace_2026-11-19/`.
- E1 stays INCONCLUSIVE (frozen). E1b (`docs/laminar_motif/E1B_SEAL.md`, SEALED 2026-11-19 r2) tests the rho = 4
  band floor (arm A, 10-80 Hz band) and SOM at fixed PV 750 (arm B), seeds 6-8, 27 runs.

## E1b batch

Detached (pid 22788), launched 17:56Z from the clone `D:/tmp/_temp_e1b_clone` pinned at 4c99e05, 3 at a time,
about 30 min per round. At 20:07Z: 12 of 27 complete, 0 failed; arm B (larger builds) still to run. Expected end
about 22:30-23:00Z. Log `D:/tmp/_temp_e1b_runs/batch.log`. No result has been read.

If the process died: from the clone, `D:/venvs/_temp_o51_jax0102/Scripts/python.exe -B -u scripts/e1b_batch.py
--out D:/tmp/_temp_e1b_runs --jobs 3` resumes (complete receipts are skipped). Launch detached (Win32_Process.Create,
CreateFlags `[uint32](0x200 -bor 0x08000000)`).

## Next steps, in order

1. When `batch.log` shows `finish: 27 of 27 complete`: score in `D:/jomission`
   `D:/venvs/_temp_jomission_lansa/Scripts/python.exe -B scripts/e1b_score.py --runs D:/tmp/_temp_e1b_runs --out results/laminar_e1b/e1b_result.json`.
   Gates 3 (15/15 E1 calls reproduced) and 4 (null) run first; exit 2 on a gate failure or a missing run.
2. Copy the receipts and counts into `results/laminar_e1b/`, add their SHA-256 lines to
   `paper/jomission_lansa/data/sources_sha256.txt` (D6; bytes, LF; update the count in methods.tex), add an R record.
3. Write the E1b result into Results and Methods: arm A as a sensitivity analysis of E1's band floor, arm B
   restricted to SOM at fixed PV (total inhibition co-varies). State that E1b was designed after the E1 re-read,
   once, in Methods. E1's verdict and the abstract's E1 sentence change only with Hamm's word.
4. Rebuild (MiKTeX on PATH: `export PATH="$(cygpath -u "$LOCALAPPDATA")/Programs/MiKTeX/miktex/bin/x64:$PATH"`;
   `latexmk -pdf -interaction=nonstopmode -outdir=/d/tmp/_temp_o51_pdfbuild/out_r2b main.tex` in
   `paper/jomission_lansa`): 0 warnings required. Gate: `JOMISSION_REMOTE=https://github.com/HNXJ/jomission.git
   D:/venvs/_temp_jomission_lansa/Scripts/python.exe scripts/project_check.py --allow-dirty` must print
   PROJECT_CHECK_PASS.
5. Independent reviewer check of the patch (numbers, abstract-Results alignment, unrelated content unchanged).
6. Rescore with `quality_rubric.md` (UNKNOWN scores 0; report coverage, DEFECTs, UNKNOWNs) and present the lock
   decision to Hamm with a short verification report.

## Open minors (not blocking)

T8 rule at n = 3 without intervals (C3); results.tex:238-239 against discussion.tex:11-12 on what lost the deep
alpha-beta (C5); 7 unasked contrasts (C8); results.tex:53-54 exact permutation count 13 824; data archive and DOI
(stage 13). Run records exist as single untracked copies in `D:/cowork/jomission` (read-only): a stage-13
data-availability decision.

## Rules that bit this session

No `python -c` / `python -` (write a script, run by path); no filler commands in chains; no `sed -i`; stage exact
paths; never touch other agents' ` D` deletions or `results/laminar_e2/*.png` in `D:/jomission`; `D:/cowork/jomission`
is read-only; push with `git push -q origin <branch> </dev/null` and check `ls-remote`; Haiku workers stop on any
baseline diff, so state the baseline ruling in the packet; seal before running any new test.
