# OPEN-51: identity-checked re-run protocol for E0/E1 rasters (DRAFT, not authorized)

Status: prepared on the owner's instruction of 2026-11-17 ("prepare, but do not execute"). Nothing here has run.
Execution needs a separate owner authorization for new simulations. Findings it builds on:
`OPEN51_E0_E1_VIZ_FEASIBILITY_2026-11-17.md`.

## Claim the protocol can support

The map from per-cell spike trains to per-step population counts is many-to-one. A re-run whose counts equal the
original counts at every step establishes aggregate observational equivalence. It does not establish that the
original per-cell raster has been recovered. Any raster produced here is therefore a **count-validated reconstructed
raster**, never a recovered original raster. V1 and V2 drawn from it carry that label in the title, in the
`visualization_note` and in `provenance.json`.

## Environment (gate R0; a failure stops everything)

| Item | Value | Status |
|---|---|---|
| Python | 3.14.3 | observed (side-install `.pyc` tag cpython-314; current venvs 3.14.3) |
| JaxFNE | `e0160dfc` from `D:/repos/_ref/jaxfne.git`, installed into a fresh venv | declared for E0, observed for E1 |
| JaxFNE identity check | each of the 84 modules in `E:/repos/_wt/jaxfne-0.5.9-site` compiles equal to the new install (`_temp_o51_marshal.py` method) | must pass |
| jax, jaxlib | unknown: the run records hold only `jax_devices: ["cpu:0"]`; pyproject at 3257ffc allows `>=0.4.25,<0.11` | the main risk to equality |
| numpy | unknown (pyproject: `>=1.24`) | affects RNG only if JaxFNE draws with numpy |
| Device, dtype | CPU, float32 (`run()` refuses other dtypes) | recorded |
| jomission code | E0: tree at 3257ffc; E1: tree at e381154. One `git worktree` per commit, never the live tree; `git status --porcelain` empty in each worktree before every run | inferred from timing; clean tree must pass |
| Import origin | the driver asserts `caminos_e0.__file__` lies under its worktree and `jaxfne.__file__` under the fresh venv, and refuses a side-install or another editable install | must pass |
| Comparator integrity | each F: counts file re-hashed and equal to the sha256 table in `scripts/open51_viz.py` before it is compared. That table was computed from the same salvage copy, so it detects later change only | must pass |
| JAX and RNG configuration | full `jax.config` dump, including `jax_default_prng_impl`, `jax_threefry_partitionable` (default changed in jax 0.5), `jax_enable_x64` and `jax_default_matmul_precision`; `XLA_FLAGS`; `PYTHONHASHSEED` (set ordering during construction); numpy version | to record; the values are unknown for the original runs |
| Threads | single-process; record `OMP_NUM_THREADS`, the XLA CPU thread count and the CPU model | to record |
| Device | CPU. Host rule `host.md` sends heavy compute to the GPU; identity with CPU-run originals requires CPU, recorded as the exception | exception noted |

Candidate jax versions are the ones installed in venvs that existed on 2026-11-06 to 2026-11-08 (to inventory before
R1). Each candidate is tried only through R1; the first one that passes R1 is fixed for the rest.

## Driver

A new driver `scripts/open51_rerun.py`, run inside each worktree, imports that commit's `caminos_e0.build_spec`,
`build_sim` and `spikes_to_counts` and repeats `run()` (lines 121-130) unchanged, with two additions:

1. it saves the full spike raster sparsely (`(step, cell)` int32 pairs; about 10^5 to 10^6 spikes per run) next to the
   counts it derives with `spikes_to_counts`;
2. it saves the realized edge table (`model.static["adex"]` pre, post, weight_nS, receptor) and its sha256, which gives
   V0's realized level the receipt it lacks.

Outputs go to `D:/tmp/_temp_o51_rerun/` and are not committed until a classification exists. F: stays read-only:
the original counts are read there as the comparator.

## Order, comparison and stop rules

| Step | Run | Pass condition | On failure |
|---|---|---|---|
| R1 pilot | E0 seed 1 | `np.array_equal` of the new counts and `e0_seed1_counts.npy` (int32, shape (120000, 3)); file sha256 recorded, not required, because the `.npy` header can differ across numpy versions | try the next jax candidate; when none is left, stop: every class below becomes NOT REPRODUCED |
| R2 self-determinism | E0 seed 1 again, same environment | raster and edge table byte-identical to R1 | stop; the environment is nondeterministic, and no raster is classified |
| R3 E0 | seeds 2-5 | counts equal per seed | stop the batch at the first mismatch; R4 does not start |
| R4a E1 determinism | rho 4 seed 1 at e381154, twice (the E1 code path with the `n_sst` change; about 1476 s each) | first run's counts equal the original; second run's raster and edge table byte-identical to the first | stop; no E1 raster is classified |
| R4 E1 | rho 0.25, 0.5, 2 × seeds 1-3 and rho 4 seeds 2-3 at e381154 | counts equal per run | stop the batch at the first mismatch |

E1 rho 1 seeds 1-3 are E0 seeds 1-3 (byte-identical counts), so they take E0's classification from R1 and R3 and
are never re-run.

Every run also recomputes `analyze_run` and must reproduce its committed JSON (the check in `scripts/open51_viz.py`).

## Classification per run

| Class | Condition | Use |
|---|---|---|
| COUNT-VALIDATED RECONSTRUCTION | R0, R1 and R2 pass, and all 120 000 × 3 counts equal | V1/V2 drawn, labelled "count-validated reconstructed raster" |
| NOT REPRODUCED | any count differs | the raster is a new realization; it is not used for this lineage's V1/V2 |
| INCOMPLETE | run missing, short or crashed | none |

A lineage receives V1/V2 only when every run its V1/V2 draws on is COUNT-VALIDATED RECONSTRUCTION. The
lineage-record change (adding `visualization` paths, removing the false exemption) is a separate owner decision after
classification.

## Cost

The 17 distinct original runs took 34 509 s (9.6 h) of recorded `wall_s` on CPU. The repeats in R2 (about 2130 s) and
R4a (about 1476 s) bring the total to about 38 100 s (10.6 h) if all pass, run one at a time under `suite_lock.py`. The pilot (R1) alone is about 35 min per jax
candidate.

## Remaining uncertainty after a full pass

- E0's JaxFNE build is declared, not observed; a count match at `e0160dfc` makes it consistent.
- Per-cell identity stays unproven: equal counts plus self-determinism make the reconstruction reproducible, not
  identical to the lost original raster.
- The jomission working tree at run time may have been dirty; a count match makes a material difference unlikely
  (inferred).
