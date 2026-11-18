# OPEN-51 re-run result, 2026-11-18: R1 NOT REPRODUCED; batch stopped (independently verified)

Authorization: owner 2026-11-17, execution of the frozen protocol `OPEN51_RERUN_PROTOCOL_2026-11-17.md` at 5d586da.
Protocol, criteria and driver were not changed during execution. Registration and promotion are deferred.

## Setup

- Driver `scripts/open51_rerun.py` and batch `scripts/open51_rerun_batch.py` at f7f2b5f.
- Fresh venv `D:/venvs/_temp_o51_rerun`: Python 3.14.3, jax and jaxlib 0.10.1, numpy 2.4.6, scipy 1.18.0, and
  JaxFNE `e0160dfc`.
  - The 84 side-install modules compile equal to the install (82 byte-identical, 2 structurally).
  - jax 0.10.1 was the only candidate under the protocol's rule. The global C:/Python314 is the only environment that
    existed on 2026-11-06 to 2026-11-08 with jax installed. The original handoff venv is gone.
- Worktrees: 3257ffc (E0) and e381154 (E1), both clean.
- Receipts and hashes are in `open51_rerun/`. The run outputs (counts, raster, edges) stay in
  `D:/tmp/_temp_o51_rerun/batch/R1/`; their sha256 values are in `open51_rerun/R1_output_files.sha256`.
- Documented deviation: the JAX compilation cache pointed to an empty directory on D: (the original used `~/.cache` on
  C:). That this has no numerical effect is a hypothesis; it was not tested.
- Smoke runs (0.3 s; E0 seed 1, E1 rho 4 seed 1) checked the pipeline only. They are not evidence and were not
  classified.

## R1 (E0 seed 1, 12 s), 2026-11-18T01:25:56Z to 01:47:36Z

- R0 gates passed: clean worktree, no PYTHONPATH, jomission from the worktree, jaxfne from the fresh venv, the
  comparator sha256 equal to the table (ac5a4415...), and the driver committed.
- Counts: shape and dtype equal, values not equal (exit 3).
  - Equal for steps 0-8087; the first differing step is 8088 (808.8 ms).
  - 91 201 of 120 000 steps differ, with a maximum absolute difference of 31 spikes per step.
  - Total spikes, new against original: E 110 867 vs 110 714, PV 104 187 vs 103 964, SOM 67 493 vs 66 667.
- Committed-JSON estimands: also different.

  | Estimand | New | Original |
  |---|---|---|
  | f_pk | 33.0 Hz | 34.0 Hz |
  | peak excess | 16.81 | 16.29 |
  | PV-SST lag | 7.0 ms | 6.9 ms |
  | spikes kept, E | 88 840 | 88 726 |

- Wall time 1293.6 s; peak working set 6.59 GB. The realized edge table has 6 236 657 edges, sha256 06f6c60f...
- Stop rule applied: R1 failed and no jax candidate is left, so the batch stopped. R2, R3, R4a and R4 did not run.

## Classification (protocol table)

- E0 seed 1: NOT REPRODUCED, observed.
- The other 19 runs: classified NOT REPRODUCED by the protocol's R1 rule (R1 failed, no candidate left). This is not
  observed; none of them was executed. The E1 rho 1 seeds inherit E0's class.
- V1/V2 for E0 and E1 stay blocked.

Evidence for the environment, committed in `open51_rerun/`:
- `jax_candidate_inventory.txt`: C:/Python314 is the only Python 3.14 environment with jax that existed on
  2026-11-06 to 2026-11-08. The other jax venvs were created 2026-11-14 or later. Python312 is the wrong interpreter.
  The original handoff venv is absent.
- `jaxfne_identity_check.txt`, with the script `jaxfne_identity_check.py`: of the 84 modules, 82 are byte-identical
  and 2 structurally identical; 0 differ.

## Observations, not acceptance evidence

- The realized edge table of R1 has the same array sha256 as the E0 seed-1 smoke run in a separate process. The
  construction was therefore identical across those two processes, which is observed for this one seed only.
- The counts match exactly for 808.8 ms and then diverge. The trajectory is the same up to a point and then separates.
  This is consistent with a small numerical difference in the simulation path amplified by the dynamics, and
  inconsistent with a different construction or seed (inferred). The candidate sources are the jax/jaxlib/XLA build,
  the CPU code path, and the compilation cache. None was tested.

## Independent verification (read-only reviewer at c85cd99, 2026-11-18): ACCEPT

The reviewer's own code recomputed, from the files:
- the counts comparison: first differing step 8088, 91 201 steps differ, max 31, totals as above;
- all output sha256 values;
- the counts implied by the saved raster, which equal the new counts;
- the analysis: `analyze_run` on the original counts reproduces the committed JSON exactly, so the mismatch is in the
  simulation, not the comparator or the analysis path.

It also confirmed:
- the executed driver's sha256 equals `git show f7f2b5f:scripts/open51_rerun.py`;
- the run body is line-identical in sequence to `caminos_e0.run()` at 3257ffc, plus the protocol's two saves;
- the order, stop rules and classification follow the protocol at 5d586da, and nothing was retried;
- the protocol file is unchanged since 5d586da, and the scripts are unchanged since f7f2b5f.

Its minor findings were applied: the unrun runs are now worded as rule-classified; the inventory and identity-check
outputs are committed. The cache deviation stays disclosed and untested.

| Dimension | Verified status |
|---|---|
| Reproducibility | FAIL: the re-run did not reproduce the original counts |
| Determinism | UNKNOWN: R2 did not run; construction identical across two processes for E0 seed 1 only |
| Provenance integrity | PASS |
| Protocol compliance | PASS |

## Preliminary assessment (main session, before verification)

| Dimension | Status |
|---|---|
| Reproducibility | FAIL: R1 counts differ from step 8088 |
| Determinism | UNKNOWN: R2 did not run (stop rule); same-process repeatability was not tested |
| Provenance integrity | PASS (preliminary): every R0 assertion recorded and passed, comparator hash matched, outputs hashed |
| Protocol compliance | PASS (preliminary): order and stop rule followed, nothing retried, criteria unchanged; the cache deviation is recorded |

The environment is a reconstruction. Equal JaxFNE modules and jax versions do not establish runtime equivalence with
the original runs.
