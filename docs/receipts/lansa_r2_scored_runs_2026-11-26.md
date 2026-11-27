# LANSA R2 scored runs: simulation receipt (seeds 7, 8, 9, T 12 s)

Spec: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md` (sealed text; code SHA 46899e2 = seal commit).
Layout `e2_3unit_n4000_dens_k1`, parameter set P0, `sigma_alt` 0.25 (sealed), default delivery route.

## 1. Run records (observed, read from each `<tag>.json`)

| tag | seed | T_s | edge_delivery | jaxfne_version | jax_devices | git_sha | git_dirty | simulate_s | total_s |
|---|---|---|---|---|---|---|---|---|---|
| r2_p0_n4000_s7_T12 | 7 | 12.0 | default | 0.5.8 | ["cpu:0"] | 46899e29f5f4b6571248a18983cb6efa8f9935a0 | false | 2762.03 | 2783.95 |
| r2_p0_n4000_s8_T12 | 8 | 12.0 | default | 0.5.8 | ["cpu:0"] | 46899e29f5f4b6571248a18983cb6efa8f9935a0 | false | 2806.06 | 2825.43 |
| r2_p0_n4000_s9_T12 | 9 | 12.0 | default | 0.5.8 | ["cpu:0"] | 46899e29f5f4b6571248a18983cb6efa8f9935a0 | false | 2790.59 | 2810.51 |

All three records meet the sealed section 7 requirement (`edge_delivery` default, `jaxfne_version` 0.5.8, `jax_devices` `["cpu:0"]`).

## 2. Output checks (observed)

`scratchpad/_temp_r2_check_scored.py`, run with `/home/hamm/.venvs/jomission058gpu/bin/python -I`, `JAX_PLATFORMS=cpu`. Exit 0.

| tag | npz sha256 | float arrays | nonfinite values | counts total (spikes) |
|---|---|---|---|---|
| r2_p0_n4000_s7_T12 | 7758e519b0ca8821921ad9aadae9ad6064d85e6e231148dd296b097dba9e1aca | 2 (`g_nS`, `lfp_pA`) | 0 | 422734 |
| r2_p0_n4000_s8_T12 | 93f4e1df9909279418454b93bae45c7f4cc128a4dbaec2e33ae5a54675fa4594 | 2 | 0 | 426732 |
| r2_p0_n4000_s9_T12 | 2d9c139b9aed08ba078b098c90dd382fd748e58d72ead6192427faa815236b4b | 2 | 0 | 425795 |

Each npz holds 18 spike-train arrays (`{S,M,D}_{E,PV,SOM}_{t,c}`), `counts` (120000 x 9), `block_keys`, `lfp_pA` (120000 x 3) and `g_nS` (120000 x 6).

## 3. Post-processing chain and sealed scorer (observed)

Chain as `scripts/run_e2_scored.sh` after the run: `e2_motif_figure.py TAG 2`, `e2_motif_similarity.py TAG`, `e2_scorer.py TAG`.
Run with `venv_seal` (python 3.14, jaxfne 0.5.8, `JAX_PLATFORMS=cpu`, `PYTHONPATH` = this clone). All nine steps exit 0 (seed 9 rerun in the foreground after its background log was incomplete; the rerun is the one reported).

| tag | r_proto | r_ab | r_gm | crossing_2se | perm_rank | smooth_p95 | identical_control | pass |
|---|---|---|---|---|---|---|---|---|
| r2_p0_n4000_s7_T12 | -0.3495 | 0.8878 | -0.8393 | false | 5 | 0.3801 | -0.0773 | false |
| r2_p0_n4000_s8_T12 | -0.3408 | 0.9066 | -0.8338 | false | 5 | 0.3801 | -0.0773 | false |
| r2_p0_n4000_s9_T12 | -0.3658 | 0.8774 | -0.8446 | false | 5 | 0.3801 | -0.0773 | false |

`reason` is null on all three (no nonfinite value, all 150 frequencies used). `halves.agree` is true for s7 and s9 and false for s8. The pass rule is the sealed scorer's, unchanged. The verdict files are `results/laminar_e2/<tag>_verdict.json` (untracked).

## 4. Not in this receipt

- Readout conditions G0, G1, G2, G3, G1', K1, K2 (`GEOMETRY_PHASE1_PROTOCOL_2026-11-25.md` sections 4-6): not run.
- Readout conditions G0, G1, G2, G3, G1', K1, K2 (`GEOMETRY_PHASE1_PROTOCOL_2026-11-25.md` sections 4-6): not run.
- V2 final raster and V3 atlas: not built.
- Raw outputs are untracked (`results/` is outside git); archive is a later step.
- Not claimed: the sealed scorer's FAIL is the result for this unit-spectrum map against the 11-area prototype (the sealed rule). It does not by itself settle the section 7 readout test.
