# OPEN-51 re-run protocol, amendment A1: fixed jax 0.10.2 environment (DRAFT; reviewed ACCEPT-WITH-CHANGES, changes applied; not authorized)

Status: drafted on the owner's instruction of 2026-11-18 ("Draft protocol amendment"). It does not edit the frozen
protocol `OPEN51_RERUN_PROTOCOL_2026-11-17.md`@5d586da, and it applies only after an owner ruling. Nothing has run
under A1. Evidence: `OPEN51_DIAGNOSIS_2026-11-18.md`.

## Reason

- Under the frozen candidate rule (venvs that existed on 2026-11-06 to 2026-11-08), the only candidate was jax 0.10.1.
  R1 then failed: the first differing step was 8088.
- The diagnosis traced the divergence to the compiled executable:
  - the recovered original executable reproduces the original counts on the re-run's inputs;
  - jax/jaxlib 0.10.2 with the re-run's other pins compiles E0 seed 1 to the original's optimized HLO;
  - a fresh 0.10.2 run, diagnostic and outside the protocol, reproduced the E0 seed-1 counts at all 120 000 steps,
    and the committed estimands matched.
- The environment that built the originals was deleted. A1 replaces the candidate rule with one fixed, reconstructed
  environment.

## Selection

- A1's environment was chosen after it matched E0 seed 1. E0 seed 1 is therefore the selection case, and an A1 R1 is
  a confirmation, not a test of the environment choice.
- R1 still tests compile-to-compile variation, because it compiles fresh and may produce a new variant of the two
  varying kernels.
- The out-of-sample tests are R3 (E0 seeds 2-5) and R4a/R4 (E1, a different code path and different shapes).
- If R3 or R4 fails, the classification of E0 seed 1 is reported as conditional on the selection, not as an
  independent pass.

## Changes to the frozen protocol

| # | Section | Frozen | A1 |
|---|---|---|---|
| A1.1 | Environment, candidate rule | candidate jax versions from venvs that existed on 2026-11-06 to 2026-11-08; the first to pass R1 is fixed | exactly one environment: the tested venv `_temp_o51_jax0102` (Python 3.14.3; `jax==0.10.2 jaxlib==0.10.2 numpy==2.4.6 scipy==1.18.0 ml_dtypes==0.5.4 opt_einsum==3.4.0`; JaxFNE `e0160dfc`, all `--no-deps`). No other candidate: an R1 failure stops everything |
| A1.2 | R0, batch-level checks | `open51_rerun.py` records packages per run | the batch driver checks, before R0c: `pip freeze` equals the A1.1 list exactly; `pip check` is clean; the JaxFNE identity check (each of the 84 modules in `E:/repos/_wt/jaxfne-0.5.9-site` compiles equal to the installed `e0160dfc`) passes for this venv. Any failure stops everything |
| A1.3 | R0c, new compile gate | none | a batch step runs `scripts/open51_compile.py` for E0 seed 1 in the A1 venv (no simulation). Its captured `jit_scan` must have YNN fusion count 7, `dynamic_variables` 4, and HLO equal to `open51_diag/original_e0s1.hlo.txt.gz` under the strict comparison `open51_diag/hlodiff_strict.py` (`identical True`). The 2026-11-18 capture meets this condition. On failure, stop. E1 has no compile gate; it is gated by R4a |
| A1.4 | R1 | E0 seed 1 | unchanged, re-run under A1. The 2026-11-18 diagnostic run is not reused (Decision 2) |
| A1.5 | R2 self-determinism | E0 seed 1 again, same environment | the same, but with a fresh, empty JAX cache, so the repeat also recompiles. Pass condition unchanged: raster and edge table byte-identical to R1. A failure stops everything, with no retry (frozen rule). The receipt records the suspected cause: compile variation if the two varying kernels differ between R1's and R2's cached executables, else execution |
| A1.6 | R4a | E1 rho 4 seed 1, twice | the second run also uses a fresh JAX cache, with A1.5's failure rule |
| A1.7 | Drivers and labels | `scripts/open51_rerun.py`; batch `scripts/open51_rerun_batch.py` | the run driver is unchanged (f7f2b5f content), so each per-run receipt keeps its hard-coded `protocol` field `...@5d586da`. The A1 identity goes in a sidecar `<step>.a1.json` per run and in `batch_state.json`: protocol `...@5d586da+A1@<sha>`, environment "reconstructed jax 0.10.2 environment (A1)". A new batch driver `scripts/open51_rerun_batch_a1.py` differs from the frozen batch only in: the interpreter path; the A1.2 checks; the R0c step; fresh caches for R2 and the second R4a; and the sidecars. Its diff against the frozen batch driver is independently reviewed before launch |
| A1.8 | Classification | as frozen | unchanged classes and labels. The 2026-11-18 R1 under jax 0.10.1 stays recorded as NOT REPRODUCED under the frozen candidate rule; A1 classifications are new, separate records. The Selection rule above applies |
| A1.9 | Remaining uncertainty | as frozen | add: the A1 environment is a reconstruction; the deleted venv's other packages and XLA flags are unknown. A count match makes a material difference unlikely (inferred) |

Every other part of the frozen protocol is unchanged, including:
- worktrees 3257ffc and e381154, clean;
- the import-origin and comparator checks;
- the order R1, R2, R3, R4a, R4 and the stop rules;
- the JSON estimand check;
- the "count-validated reconstructed raster" label;
- one run at a time under `suite_lock.py`;
- outputs uncommitted until classified.

## Cost

- **Time:** R0c about 2 min. The full order takes about 38 100 s (10.6 h) if all pass, as frozen; the 2026-11-18
  run took 1236 s against a frozen basis of 2128 s per E0 run, so this estimate is conservative (inferred). A1.5 and
  A1.6 add two compiles, under 2 min each.
- **Setup:** the batch driver and its review, about 1 h.
- **Resources:** about 6.6 GB working set per run, and about 200 MB on D: per run (edge table 200 MB, raster 2 MB,
  counts 1.4 MB), so about 3.8 GB for 19 runs (R1, R2, 4 in R3, 2 in R4a, 11 in R4).

## Decisions for the owner

1. Adopt A1 as written, including the Selection rule (a conditional classification of E0 seed 1 if R3 or R4 fails).
2. R1 evidence: re-run R1 under A1 (recommended; about 22 min, keeps the protocol's own record), or accept the
   2026-11-18 diagnostic 0.10.2 run as R1. That run used a different lane head, ran outside the batch and carries
   the frozen label.
3. Authorize the A1 re-run (about 10.6 h). Registration and promotion stay separate decisions, as frozen.
