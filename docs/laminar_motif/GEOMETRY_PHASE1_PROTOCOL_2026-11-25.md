# Geometry Phase 1 protocol, v1 (text frozen; execution not authorised)

Status: protocol text frozen in this commit for owner review. Execution is not authorised. The 2026-11-25 ruling
authorises no scientific run, and the manifest's next authorised task is `SCI-AGSDR-CALIBRATION`. The eligibility
precondition (section 3) is open. Phase 2 is not authorised. Items marked OWNER need a decision before any go.

## 1. Question

Does readout geometry (unit depths and the volume-conduction width sigma) explain the gap between the model's gamma profile
and the observed one? Phase 1 changes only the readout; the unit spectra are fixed.

## 2. Fixed

- T1 record and FAIL (`docs/laminar_motif/SEAL_T1.md`; 57 of 57 verdicts `pass: false`).
- Gamma criterion: r_gm >= 0.6 on the 50-150 Hz image band, as in `scripts/e2_scorer.py`. Retained by ruling.
- Observed data: `docs/laminar_motif/data/lansa_lfp_Y.json` (21 bins, 150 Hz, 11 areas: V1, V3, V4, MT, MST, LIP, TEO, PMD,
  7A, 8A, PFC). Identical to its blob at 53a3e45.

## 3. Eligibility precondition (OPEN, OWNER)

Every existing spectrum family carries Y-based selection (DECISIONS section 4). Phase 1 therefore has one of two routes:

- R1: run on existing spectra and label the output exploratory. It makes no held-out claim.
- R2: generate new spectra whose parameters are fixed before any comparison with Y. This is a new lineage with its own seal
  and run. It is the only route to a held-out claim.

No execution starts until the owner picks a route.

## 4. Estimand

Per held-out area a: r_gm(a) is the Pearson correlation over the 21 bins between the model's 50-150 Hz band-mean profile and
area a's own 50-150 Hz band-mean profile (not the 11-area mean). Also r_ab(a) and crossing_2se.
Summary: the median over the 11 leave-one-area-out (LOAO) folds. In each fold, any geometry choice uses only the other 10 areas.

## 5. Conditions (2 x 2 with interaction)

Anatomical depths are on the LANSA scale (section 7): L2/3 mid -0.216, L4 mid 0.000, L5 mid +0.198 (primary), L6 mid +0.406
(sensitivity, per the ruling).

| ID | Placement (S, M, D) | sigma | Role |
|---|---|---|---|
| G0 | sealed: -0.40, 0.00, +0.30 | 0.15 | baseline |
| G1 | anatomical: -0.216, 0.000, +0.198 | 0.15 | prespecified test |
| G2 | sealed | sigma_alt | width effect at the sealed placement |
| G3 | anatomical | sigma_alt | width effect at the anatomical placement |
| G1' | anatomical with D = +0.406 | 0.15 | D sensitivity (ruling) |

- sigma_alt is OWNER's choice. The draft proposes 0.25 with no data basis.
- Effects: placement = G1 - G0; width = G2 - G0 (and G3 - G1); interaction = (G3 - G1) - (G2 - G0). Interaction is reported and
  not tested as a claim.
- No placement search. The earlier full grid (G3 of the 2026-11-25 draft) is dropped.

## 6. Controls

- K1, required. A flat-spectrum readout: units with identical flat gamma power at the same depths and sigma. It tests whether
  geometry alone produces the profile.
- K2, required. A depth-shuffled null: random ordered placements under the same LOAO selection. It gives the base rate of
  passing by chance.
- K3 and K4 (spectral-shift null; unit-order permutation, as in `scripts/e2_depth_placement.py`) are optional diagnostics.

## 7. Support and falsification (OWNER to confirm before any go)

- Support: G1 gives a median LOAO r_gm >= 0.6, exceeds K1 at the same geometry, and keeps median r_ab >= 0.6.
- Falsification: G1's median r_gm < 0.6, or G1 does not exceed K1. Then anatomical placement does not account for the gap.
- Depth source: the anatomical depths come from the layer table of `docs/laminar_motif/ALLEN_V1_REPLICA_SPEC.md` section 1 (um:
  mid-depths 263, 431, 585, 747; thickness 779). That spec is RETIRED, and its source is a mouse V1 model. Transfer to the
  macaque LANSA areas is an assumption, UNKNOWN in validity. The protocol uses it as a prespecified condition only.
- sigma = 0.15 is labelled an assumption in code (`scripts/allen_v1_figure.py` line 36; `scripts/e2_motif_figure.py` SIGMA).
  The repo holds no source for its value.

## 8. Provenance

- Unit-spectrum records are read from the committed cowork HEAD (21b2697) with `git show`, not from the dirty working tree.
- The T1 motif receipts equal HEAD (57 of 57). The `.npz` files are present in the working tree (57).
- Each run record states the SHA, config, seed and held-out area.

## 9. Prior result

`paper/jomission_lansa/sections/results.tex` lines 481-488: the depth placement was selected on the observed prototype, and
its "held-out" runs are scored against the same prototype. Not held out. `docs/laminar_motif/CLAIM_DEPENDENCY_MATRIX_2026-11-24.md`
lists it as C12, UNSUPPORTED-PROVENANCE. Under R1 this protocol's output is exploratory, not a replacement for it.

## 10. Before any go

- OWNER: the route (R1 or R2); sigma_alt; confirmation of D = L5 primary and L6 sensitivity; the median-r_gm rule.
- A manifest entry that authorises Phase 1, since the current next authorised task is `SCI-AGSDR-CALIBRATION`.
- The seal (this file's final version) committed and pushed before any run (AGENTS.md).
- Phase 2 remains unauthorised.
