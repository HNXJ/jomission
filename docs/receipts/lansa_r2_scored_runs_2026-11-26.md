# LANSA R2 scored runs: simulation receipt (seeds 7, 8, 9, T 12 s)

Spec: `docs/laminar_motif/GEOMETRY_R2_SPEC_2026-11-25.md` (sealed text; code SHA 46899e2 = seal commit).
Layout `e2_3unit_n4000_dens_k1`, parameter set P0, `sigma_alt` 0.25 (sealed), default delivery route.
Lineage: SCI-LANSA-GEOM-R2 (`manifests/todo.json`). Status: K2 NOT_RUN, so the protocol is not complete.

## 1. Run records (observed, read from each `<tag>.json`)

| tag | seed | T_s | edge_delivery | jaxfne_version | jax_devices | git_sha | git_dirty | simulate_s | total_s |
|---|---|---|---|---|---|---|---|---|---|
| r2_p0_n4000_s7_T12 | 7 | 12.0 | default | 0.5.8 | ["cpu:0"] | 46899e29f5f4b6571248a18983cb6efa8f9935a0 | false | 2762.03 | 2783.95 |
| r2_p0_n4000_s8_T12 | 8 | 12.0 | default | 0.5.8 | ["cpu:0"] | 46899e29f5f4b6571248a18983cb6efa8f9935a0 | false | 2806.06 | 2825.43 |
| r2_p0_n4000_s9_T12 | 9 | 12.0 | default | 0.5.8 | ["cpu:0"] | 46899e29f5f4b6571248a18983cb6efa8f9935a0 | false | 2790.59 | 2810.51 |

All three records meet the sealed section 7 requirement (`edge_delivery` default, `jaxfne_version` 0.5.8, `jax_devices` `["cpu:0"]`).

## 2. Output checks (observed)

Scratch script `_temp_r2_check_scored.py` (durable copy in `results/laminar_e2_r2/scripts/`), run with `/home/hamm/.venvs/jomission058gpu/bin/python -I`, `JAX_PLATFORMS=cpu`. Exit 0.

| tag | npz sha256 | float arrays | nonfinite values | counts total (spikes) |
|---|---|---|---|---|
| r2_p0_n4000_s7_T12 | 7758e519b0ca8821921ad9aadae9ad6064d85e6e231148dd296b097dba9e1aca | 2 (`g_nS`, `lfp_pA`) | 0 | 422734 |
| r2_p0_n4000_s8_T12 | 93f4e1df9909279418454b93bae45c7f4cc128a4dbaec2e33ae5a54675fa4594 | 2 | 0 | 426732 |
| r2_p0_n4000_s9_T12 | 2d9c139b9aed08ba078b098c90dd382fd748e58d72ead6192427faa815236b4b | 2 | 0 | 425795 |

Each npz holds 18 spike-train arrays (`{S,M,D}_{E,PV,SOM}_{t,c}`), `counts` (120000 x 9), `block_keys`, `lfp_pA` (120000 x 3) and `g_nS` (120000 x 6). `lfp_pA` is not used by the scored analysis; the scored proxy is the RWS proxy computed from `g_nS` (section 6).

## 3. Post-processing chain and sealed scorer (observed)

Chain as `scripts/run_e2_scored.sh` after the run: `e2_motif_figure.py TAG 2`, `e2_motif_similarity.py TAG`, `e2_scorer.py TAG`.
Run with `venv_seal` (python 3.14, jaxfne 0.5.8, `JAX_PLATFORMS=cpu`, `PYTHONPATH` = this clone). All nine steps exit 0 (seed 9 rerun in the foreground after its background log was incomplete; the rerun is the one reported).

| tag | r_proto | r_ab | r_gm | crossing_2se | perm_rank | smooth_p95 | identical_control | pass |
|---|---|---|---|---|---|---|---|---|
| r2_p0_n4000_s7_T12 | -0.3495 | 0.8878 | -0.8393 | false | 5 | 0.3801 | -0.0773 | false |
| r2_p0_n4000_s8_T12 | -0.3408 | 0.9066 | -0.8338 | false | 5 | 0.3801 | -0.0773 | false |
| r2_p0_n4000_s9_T12 | -0.3658 | 0.8774 | -0.8446 | false | 5 | 0.3801 | -0.0773 | false |

`reason` is null on all three (no nonfinite value, all 150 frequencies used). `halves.agree` is true for s7 and s9 and false for s8. The pass rule is the sealed scorer's, unchanged. The verdict files are `results/laminar_e2/<tag>_verdict.json` (untracked; copies in `results/laminar_e2_r2/outputs/`).

## 4. Readout conditions (observed; `scripts/e2_r2_readout.py`; `results/laminar_e2/r2_readout_<tag>.json`)

Estimand (protocol section 4): per held-out area a, the 21-contact Pearson between the model's band-mean profile and area a's own band-mean profile, `Y[bin, freq, a]`; median over the 11 areas. Conditions have no selection step, so the LOAO fold does not change them.

Cross-checks, asserted in the script on every run:
- G0 against the area mean reproduces the sealed scorer's r_gm and r_ab to 1e-9 (`crosscheck_G0_areamean`).
- G0's crossing, alpha-beta and gamma profiles, `diff_se` and contrast reproduce the saved motif json, `sum_over_cells`, image bands (`crosscheck_G0_motif`).
- Unit tests, `tests/test_e2_r2_readout.py` (7 tests: kernel value, flat-spectrum readout, negative control, per-area identity and negation, band selection, summary and threshold flags). Run with `python -m unittest tests/test_e2_r2_readout.py`: 7 OK.

Median r_gm and r_ab per condition (observed):

| tag | G0 | G1 | G2 | G3 | G1' | K1 | G1 r_ab | areas with G1 r_gm >= 0.6 | G1 max area r_gm |
|---|---|---|---|---|---|---|---|---|---|
| s7 | -0.7958 | -0.4655 | -0.8349 | -0.4873 | -0.5120 | 0.1598 | 0.7632 | 0 of 11 | -0.142 |
| s8 | -0.8043 | -0.4277 | -0.7969 | -0.4320 | -0.4871 | 0.1598 | 0.7533 | 0 of 11 | -0.103 |
| s9 | -0.8036 | -0.5032 | -0.8482 | -0.5160 | -0.5595 | 0.1598 | 0.7653 | 0 of 11 | -0.183 |

Sensitivity (sealed spec section 7: pass/fail of the continuous medians at 0.5 and 0.7 beside 0.6; descriptive only, only 0.6 carries the claim):

| condition | median r_gm >= 0.5 | >= 0.6 | >= 0.7 | median r_ab >= 0.6 | >= 0.7 |
|---|---|---|---|---|---|
| G0, G1, G2, G3, G1' (each seed) | no | no | no | yes | yes |
| K1 (each seed) | no | no | no | yes | no |

Crossing (`crossing_2se`, the scorer's test on the model profile, computed with `e2_motif_figure.score` on the same spectra): false for G0, G1, G2, G3 and G1' on every seed. K1 is not defined (no window variance, so no jackknife SE). Protocol section 4 names crossing_2se alongside r_ab(a), but it is a property of the model profile: it is the same for every held-out area, so it is reported once per condition.

Sealed section 7 rule, applied to G1 (support: G1 median r_gm >= 0.6, G1 exceeds K1, G1 median r_ab >= 0.6; falsification: G1 median r_gm < 0.6, or median r_ab < 0.6, or G1 does not exceed K1):
- G1 median r_gm is below 0.6 on every seed, and below K1 (0.16) on every seed. Both falsification conditions hold.
- G1 median r_ab is 0.75-0.77 on every seed, so the r_ab clause does not trigger.
- Derived: G1 - K1 = -0.625, -0.588, -0.663; G1 - G0 = +0.330, +0.377, +0.300.

## 5. Status (owner decision, Option 2, 2026-11-26)

- **G1 FALSIFIED under the sealed readout criteria, reported per seed (seeds 7, 8, 9; not pooled).** On each seed G1 median r_gm is below 0.6 and below K1.
- **K2: NOT_RUN.** Its selection procedure is UNDEFINED in the protocol. No definition has been made after observing results, and no K2 simulation is authorised.
- **Full protocol: not complete** (K2 missing).
- This falsifies G1 under the specified operational test. It does not establish that the broader biological mechanism is impossible.
- **Independent review (reviewer agent, read-only): ACCEPT.** It reproduced all 18 per-seed medians and the G1 max-area values from `sc.unit_spectra` and Y directly. Its one receipt defect was fixed then. Review findings and their disposition are in section 8.

## 6. Visualization (contract, lineage field `visualization`)

Built by `scripts/e2_r2_v0.py` (V0) and `scripts/e2_r2_viz.py` (V1, V2, V3) from saved arrays only (no re-simulation). Paths resolve at this commit under `results/viz/SCI-LANSA-GEOM-R2/`:

| stage | path | what it is | status |
|---|---|---|---|
| V0 | `results/viz/SCI-LANSA-GEOM-R2/V0_rule_based_schematic.png` | rule-based block schematic; units S, M, D; depths are fractions of cortical thickness relative to L4; not HSPICE; the contract elements it does not draw are printed on the figure | present, partial |
| V1 | `results/viz/SCI-LANSA-GEOM-R2/V1_raster_initial_1000ms_s7.png` | initial 1 s raster, seed 7 (smoke run, 0-1 s), sampled cells | present, sampled |
| V2 | `results/viz/SCI-LANSA-GEOM-R2/V2_raster_first1s_s{7,8,9}.png` and `V2_raster_last1s_s{7,8,9}.png` | final rasters, scored seeds: first 1 s (same window as V1) and last 1 s (11-12 s), sampled cells | present, sampled |
| V3 | `results/viz/SCI-LANSA-GEOM-R2/atlas/index.html` | atlas, with rates, the scored RWS proxy and its spectra, conductances, readout and crossing tables, motif maps | PROVISIONAL: due at lineage completion, which K2 blocks |
| archive | `results/viz/SCI-LANSA-GEOM-R2/archive/V1_fullpopulation_smoke_s7_archive.png` | the earlier full-population V1 image of the smoke run, unchanged (sha256 `80f0556f…3984c3`, identical to the original) | archive |

Identity check (observed, `e2_r2_viz.py`, now a gate): the scored seed-7 run's first 1 s equals the smoke run's sample block by block (9 of 9 blocks), in per-bin counts, and in unit sizes (14251 cells). V1 and V2 first-second panels therefore show the same spikes (5211 points each).

Deviations from the contract, stated:
- V0 is rule-based and not HSPICE. The default route stores no edge list, so the renderer cannot run. V0 does not draw x/y placement, O and X connections, FF/FB direction, E/I projection sign, or connection authority g or w.
- V1 and V2 show the driver's fixed cell sample (E 200, PV 50, SOM 50 per unit, `default_rng(0)`), not all 14251 units.
- The V1 shown is not the earlier full-population image, which is kept in `archive/`.
- V3 is provisional. Membrane dynamics, I_net and H/A/m(H) are not recorded by the driver or not applicable. Plastic weights are not applicable.
- The LFP panel is the scored RWS proxy (`e2_motif_figure.rws` on `g_nS`) with the scored estimator (`e2_scorer.unit_spectra`: DPSS 1 s windows, 2 Hz smoothing, first 2 s discarded). It is not `lfp_pA`. The RWS proxy is a current-based model proxy, not a calibrated potential.

## 7. Status of remaining items

- Readout conditions G0, G1, G2, G3, G1' and K1: run (section 4). Crossing computed (section 4). Sensitivity computed (section 4).
- K2: NOT_RUN (section 5).
- The sealed scorer's FAIL (section 3) is the unit-spectrum map against the 11-area prototype under the sealed rule. It is separate from the section 7 readout test (section 4).
- V2 and V3 are built (section 6). V3 is provisional.
- Raw outputs, logs, scripts and figures are preserved in `/mnt/d/cowork/jomission/results/laminar_e2_r2/` with `MANIFEST.sha256` (checked with `sha256sum -c`).
- `results/` is largely untracked in git. Only `results/viz/SCI-LANSA-GEOM-R2/` is tracked at this commit.

## 8. Review findings and disposition

Review 1: reviewer agent, read-only, 2026-11-26: ACCEPT (section 5).

Review 2: Gemini neon (agy, `gemini-4-neon-high`), read-only, 2026-11-26: SOUND WITH FIXES. Each item was verified against the repo; disposition below.

| # | severity | finding | disposition |
|---|---|---|---|
| 1 | major | section 7 said V2/V3 not built, contradicting section 6 | fixed (section 7) |
| 2 | major | falsification sentence omitted the sealed r_ab clause | fixed (section 4 quotes the rule) |
| 3 | major | V0 labelled depths in mm; they are fractions of thickness | fixed; V0 regenerated (`scripts/e2_r2_v0.py`) |
| 4 | major | LFP panel plotted `lfp_pA`, not the scored RWS proxy | fixed; panel uses `mf.rws` and `sc.unit_spectra` |
| 5 | major | the sealed 0.5/0.7 sensitivity was not reported | fixed (section 4) |
| 6 | major | protocol crossing_2se was not computed | fixed (section 4; a model property, reported once per condition) |
| 7 | major | repo manifest hash for the receipt was stale; manifest omitted V0 to V3 | fixed (regenerated; see section 9) |
| 8 | minor | V0 called S, M, D areas; the construction builds them as units of one area "C" (`caminos_stack.py:281`) | fixed (units) |
| 9 | minor | V0 omitted contract elements without saying so | fixed: printed on the figure and listed in section 6 |
| 10 | minor | identity gate did not require count equality | fixed: gate requires counts and sizes |
| 11 | minor | G0 cross-check did not test per-area, summary, K1 | fixed: 7 unit tests (section 4) |
| 12 | minor | receipt said `results/` is outside git while `results/viz` is tracked | fixed (section 7) |
| 13 | minor | atlas marked diagnostics "present" | fixed: "partial (K2 NOT_RUN)" |
| 14 | minor | receipt omitted plastic weights as not applicable | fixed (section 6) |
| 15 | minor | sealed spec header "draft 7 (not sealed)" | not edited (sealed artifact); erratum in section 9 |
| 16 | owner | K2 selection rule undefined | open; owner decision (section 5) |
| 17 | owner | +0.095 depth line in spec | erratum in section 9; sealed spec not edited |
| 18 | owner | seed pooling not specified | per-seed reporting kept (section 5) |
| 19 | owner | manuscript does not contain R2 | open; owner decision on stage 10 |

Reviewer statements checked and not accepted: (a) "the original full-population V1 was not preserved". It existed in the scratch clone, and is now archived with an identical hash. (b) "V1 replaced without preserving" is therefore resolved, not a defect in the lineage.

## 9. Errata and provenance

- **+0.095 (erratum to spec line 11-12).** The sealed spec's owner-decision summary lists "+0.095, +0.198 primary, +0.406 sensitivity" as the confirmed depths. `docs/laminar_motif/DECISIONS_2026-11-25.md` lines 12, 24 and 145 show that +0.095 is the mouse-derived L4 boundary (z = ±0.095; `(505 - 431)/779` lower boundary, `z > +0.095` for the deep threshold), used for the T1 anatomical recount, and that the owner ruling values were blank in transmission and marked "OWNER to confirm". The readout depths are S -0.216 (L2/3 mid), M 0.000 (L4 mid), D +0.198 (L5 mid, primary), and D +0.406 (L6 mid, sensitivity). +0.095 is not used in `scripts/e2_r2_readout.py`, in the R2 simulation layout, or in any R2 output. The sealed spec is not edited.
- **Depth provenance.** The readout depths come from mouse Allen V1 mid-depths `z = (µm - 431)/779` (`DECISIONS_2026-11-25.md:24`). No Y value enters them. Mouse-to-macaque transfer is assumed and unverified (`UNKNOWN`).
- **Manifest.** `docs/receipts/lansa_r2_scored_manifest_2026-11-26.sha256` lists the durable copy under `results/laminar_e2_r2/`. It was regenerated at this commit and includes the viz outputs, scripts, tests and this receipt.
