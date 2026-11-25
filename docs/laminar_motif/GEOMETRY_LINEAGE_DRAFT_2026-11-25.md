# Geometry lineage (draft, unsealed): unit depth and readout width

Status: DRAFT 2026-11-25. Not sealed. No run has been made from it. Per AGENTS.md a seal is committed and pushed before
any run that uses it. Items marked OWNER need Hamm's decision before the seal.

## 1. Question

Does readout geometry (unit depths and the volume-conduction width sigma) explain the mismatch between the model's
superficial gamma profile and the observed one? The observed profile of a held-out area is never used to choose the
geometry that scores it.

## 2. Fixed, not touched

- T1 record and FAIL (`docs/laminar_motif/SEAL_T1.md`; 57 of 57 verdicts `pass: false`). Confirmation seeds 4-6 stay as
  recorded.
- Scorer criteria, `scripts/e2_scorer.py`: r_gm >= 0.6 on the image band 50-150 Hz; r_ab >= 0.6 on 10-30 Hz; the other
  pass rules as written. Unchanged.
- Observed data: `docs/laminar_motif/data/lansa_lfp_Y.json`, 21 bins x 150 Hz x 11 areas (V1, V3, V4, MT, MST, LIP, TEO,
  PMD, 7A, 8A, PFC).
- Depth convention: cortical-depth fraction about mid-L4, negative superficial (verified; see
  `docs/laminar_motif/DECISIONS_2026-11-25.md` section 2).

## 3. Estimand and endpoints

Per held-out area a: r_gm(a) = Pearson over the 21 bins between the model 50-150 Hz band-mean profile and the observed
50-150 Hz band-mean profile of area a alone (not the 11-area mean). Secondary: r_ab(a), crossing_2se.
Primary summary: median over the 11 leave-one-area-out (LOAO) folds of r_gm(a).

## 4. Phases

- Phase 1, readout only, dynamics fixed. Re-project the unit cross-spectra that `e2_scorer.unit_spectra` builds from
  existing runs, varying only the readout depths and sigma. No model is re-simulated. The source `.npz` and `.json`
  receipts are in `/mnt/d/cowork/jomission/results/laminar_e2/`, not in this repo: reproducibility UNKNOWN until they
  are committed or hashed.
- Phase 2, dynamics. Only after a Phase 1 result. Unit depth also enters the density profile the model draws from, so
  moving a unit in the simulation changes its cells as well as the readout. A Phase 2 seal must hold each unit's
  density fixed while the readout geometry changes, or it is not a geometry-only test.

## 5. Conditions (Phase 1)

Anatomical reference (ALLEN_V1_REPLICA_SPEC section 1; mid-depths 263, 431, 585, 747 um; thickness 779 um;
rel = (um - 431) / 779): L2/3 mid -0.216, L4 mid 0.000, L5 mid +0.198, L6 mid +0.406.

| ID | Condition | Selection | Role |
|---|---|---|---|
| G0 | Sealed layout: S -0.40, M 0.00, D +0.30, sigma 0.15 (`jomission/laminar/caminos_stack.py` line 246; `scripts/e2_motif_figure.py` SIGMA) | none | baseline |
| G1 | Anatomical depths: S -0.216, M 0.000, D +0.198; sigma 0.15 | none | prespecified test |
| G1' | As G1 with D +0.406 (L6 mid) | none | sensitivity; OWNER chooses D |
| G2 | G1 depths; sigma in {0.10, 0.15, 0.20, 0.25} | sigma chosen on the 10 non-held-out areas per fold | width sensitivity |
| G3 | Full placement grid (S < M < D on 0.05 steps; sigma 0.15) | placement chosen per fold on the 10 non-held-out areas | replaces the 2026-11-10 search, which used the observed prototype |

## 6. Controls (Phase 1)

| ID | Control | Tests |
|---|---|---|
| K1 | Flat-spectrum readout: units with identical flat gamma power at the same depths and sigma | whether geometry alone generates the profile |
| K2 | Depth-shuffled null: random ordered placements scored under the same LOAO selection | base rate of passing by chance |
| K3 | Spectral-shift null, as in `scripts/e2_depth_placement.py` | spectral content versus geometry |
| K4 | Unit-order permutation | which unit's depth the fit needs |

## 7. Support and falsification (proposed; OWNER to confirm before seal)

- Support: G1, with no selection, gives median LOAO r_gm >= 0.6, exceeds K1 and K2 at the same geometry, and keeps
  median r_ab >= 0.6.
- Falsification: G1 median r_gm < 0.6, or G1 does not exceed K1. Then anatomical placement does not explain the gap.
- G2 and G3 are exploratory. They count only if their LOAO selection holds in the held-out area of each fold.
- All conditions are reported. No pass is chosen across them.

## 8. Prior result, not evidence for this lineage

`paper/jomission_lansa/sections/results.tex` (geometry paragraph, lines 481-488, HEAD): moving the three readout depths
raised r_proto to 0.720. The selected depths (S -0.25, M -0.05, D +0.20) gave r_gm 0.64-0.84 on 8 other runs. The
selection used the 11-area mean prototype. Those 8 runs are scored against that same prototype, so nothing is held out.
The paper marks this UNRESOLVED. `docs/laminar_motif/CLAIM_DEPENDENCY_MATRIX_2026-11-24.md` lists it as C12,
UNSUPPORTED-PROVENANCE. G3 is its leave-one-area-out replacement.

## 9. Limits

- The anatomical depths come from one V1 source (A-sheet). Transfer to the LANSA areas is unverified.
- The 11 areas share one pipeline and are not independent samples.
- sigma = 0.15 is labelled an assumption in code. The repo holds no source for its value.
- The unit-spectrum receipts are in cowork, not in this repo.

## 10. Before any run

- This file's final version is sealed: committed and pushed before the run.
- OWNER items settled: D target (L5 mid or L6 mid); the median r_gm >= 0.6 rule; whether G3 is included; whether
  Phase 2 is authorised at all.
- Each run record states SHA, config, seed and the held-out area.
