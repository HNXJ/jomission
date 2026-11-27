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

## 4. Readout conditions (script `scripts/e2_r2_readout.py`, observed output `results/laminar_e2/r2_readout_<tag>.json`)

Per held-out area a (11 areas, 21-contact Pearson against `Y[bin, freq, a]`), median over the 11 areas. Conditions have no selection step, so the LOAO fold does not change them. Cross-check: G0 against the area mean reproduces the sealed scorer's r_gm and r_ab to 1e-9 for all three seeds (`crosscheck_G0_areamean`).

| tag | G0 | G1 | G2 | G3 | G1' | K1 | G1 r_ab | areas with G1 r_gm >= 0.6 | G1 max area r_gm |
|---|---|---|---|---|---|---|---|---|---|
| s7 | -0.7958 | -0.4655 | -0.8349 | -0.4873 | -0.5120 | 0.1598 | 0.7632 | 0 of 11 | -0.142 |
| s8 | -0.8043 | -0.4277 | -0.7969 | -0.4320 | -0.4871 | 0.1598 | 0.7533 | 0 of 11 | -0.103 |
| s9 | -0.8036 | -0.5032 | -0.8482 | -0.5160 | -0.5595 | 0.1598 | 0.7653 | 0 of 11 | -0.183 |

Cells G0 to K1 are median r_gm over the 11 areas. The G1 r_ab column is the median r_ab over the 11 areas. Sealed section 7 rule (support: G1 median r_gm >= 0.6, G1 exceeds K1, G1 median r_ab >= 0.6; falsification: G1 median r_gm < 0.6, or G1 does not exceed K1): on all three seeds G1 median r_gm is below 0.6 and below K1, so the falsification condition holds. The r_ab condition holds (0.75-0.77). Derived: G1 - K1 = -0.625, -0.588, -0.663; G1 - G0 = +0.330, +0.377, +0.300.

## 5. Status (owner decision, Option 2, 2026-11-26)

- **G1 FALSIFIED under the sealed readout criteria, reported per seed (seeds 7, 8, 9; not pooled).** Each seed: G1 median r_gm below 0.6, and G1 below K1. Falsification condition met on every seed.
- **K2: NOT_RUN.** Its selection procedure is UNDEFINED in the protocol. No definition is made after observing results, and no K2 simulation is authorised.
- **Full protocol: not complete** (K2 missing).
- This falsifies G1 under the specified operational test. It does not establish that the broader biological mechanism is impossible.
- **Status of the falsification:** independent non-writer review (reviewer agent, read-only) returned ACCEPT. It reproduced all 18 per-seed medians (G0, G1, G2, G3, G1', K1) and the G1 max-area values from `sc.unit_spectra` and Y directly, and confirmed the manifest (`sha256sum -c`, 38 OK). It flagged one minor receipt defect, fixed in this section.

## 6. Visualization (contract, lineage field `visualization`)

Built by `scripts/e2_r2_viz.py` from saved arrays only (no re-simulation). Paths resolve at this commit under `results/viz/SCI-LANSA-GEOM-R2/`:

| stage | path | what it is | status |
|---|---|---|---|
| V0 | `results/viz/SCI-LANSA-GEOM-R2/V0_rule_based_schematic.png` | rule-based block schematic, labelled rule-based; not an HSPICE schematic (the HSPICE renderer needs `params["edge_list"]`, which the default route does not store); no edge list drawn | present |
| V1 | `results/viz/SCI-LANSA-GEOM-R2/V1_raster_initial_1000ms_s7.png` | initial 1 s raster, seed 7 (smoke run, 0-1 s) | present, sampled cells |
| V2 | `results/viz/SCI-LANSA-GEOM-R2/V2_raster_first1s_s{7,8,9}.png` and `V2_raster_last1s_s{7,8,9}.png` | final rasters, scored seeds: first 1 s (same window as V1) and last 1 s (11-12 s) | present, sampled cells |
| V3 | `results/viz/SCI-LANSA-GEOM-R2/atlas/index.html` | atlas | PROVISIONAL: due at lineage completion, which K2 blocks |

Identity check (observed, `e2_r2_viz.py`): the scored seed-7 run's first 1 s equals the smoke run's sample block by block (9 of 9 blocks) and in per-bin counts. V1 and V2 first-second panels therefore show the same spikes (5211 points each).

Deviations from the contract, stated: the rasters show the driver's fixed cell sample (E 200, PV 50, SOM 50 per block, `default_rng(0)`), not all 14251 units. The V1 raster here replaces the earlier full-population V1 image of the smoke run. The V3 atlas marks membrane dynamics, I_net and H/A/m(H) as not recorded or not applicable; the driver does not store them. The LFP proxy is qualified as current-based, not a potential. The first 2 s are excluded from the LFP spectra, as the scorer's `discard_s` does.

## 7. Status of remaining items

- Readout conditions G0, G1, G2, G3, G1' and K1: run, section 4. K2: NOT_RUN (section 5).
- The sealed scorer's FAIL (section 3) is the unit-spectrum map against the 11-area prototype under the sealed rule. It is separate from the section 7 readout test in section 4.
- V2 final raster and V3 atlas: not built. Their purpose is to be decided after this review.
- Raw outputs, logs and scripts are preserved in `/mnt/d/cowork/jomission/results/laminar_e2_r2/` with `MANIFEST.sha256` (38 files, verified with `sha256sum -c`). The sealed-scorer `_verdict.json` files are in that directory under `outputs/`.
- Raw outputs are untracked (`results/` is outside git); the archive is a later step.
