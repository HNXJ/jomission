# LANSA decisions, 2026-11-25: record

Status: owner decisions recorded 2026-11-25. Done here: the depth convention verified, the T1 audit corrected, a
recount receipt made, the geometry lineage drafted. No experiment run. No sealed spec or verdict changed. D1
(abstract) deferred. The T1 FAIL and the E1 INCONCLUSIVE stay in the record.

## 1. Geometry lineage: approved conditionally

Draft: `docs/laminar_motif/GEOMETRY_LINEAGE_DRAFT_2026-11-25.md` (unsealed). The owner's conditions are written into it:
T1 FAIL preserved; held-out by area (leave-one-area-out); geometry-only (Phase 1) separated from dynamics (Phase 2);
flat-spectrum and shuffled controls. Open: the OWNER items in the draft, before any seal.

## 2. Depth convention: verified; deep threshold open

- Verified (observed in the repo): LANSA bins are cortical-depth fractions about mid-L4, negative superficial
  (`docs/laminar_motif/data/lansa_density_X.json`, `units`; `docs/laminar_motif/PLAN_PAPER_JOMISSION_LANSA.md` line 29).
- Verified: the Allen replica layer table (`docs/laminar_motif/ALLEN_V1_REPLICA_SPEC.md` section 1, um): L4 357-505,
  mid-L4 431, mid-depths L2/3 263, L5 585, L6 747; thickness 779 um (`scripts/allen_v1_figure.py` THICK_UM). On the
  LANSA scale: L4 spans -0.095 to +0.095; L2/3 mid -0.216; L5 mid +0.198; L6 mid +0.406.
- Verified: the 3-unit model depths S -0.40, M 0.00, D +0.30 are hard-coded (`jomission/laminar/caminos_stack.py`
  line 246), listed as "agent choices" (`PLAN_PAPER_JOMISSION_LANSA.md` line 135). Nothing in the repo derives them from
  anatomy. On the Allen scale S -0.40 is at 119 um, inside L1 (50-169 um).
- Conflict: the LANSA module rule (|z| <= 0.175; `SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md`, Laminar modules row)
  does not match the Allen L4 bounds (+/-0.095). That row also says the boundaries are "not the same" and the V1 L4
  convention is UNKNOWN.
- Unknown: whether the V1 Allen boundaries transfer to the 11 LANSA areas.
- The decision text as received has no numeric deep threshold ("A peak at does not establish deep-layer dominance").
  Provisional candidate used here: alpha-beta deep if its peak z > +0.095 (the L4 lower boundary). OWNER to confirm.

## 3. Qualitative recount under the frozen anatomical rule

The 21 of 57 is not 21 replications. Rules were frozen before the run:

| Rule | Gamma superficial | Alpha-beta deep | All three | All three and crossing_2se |
|---|---|---|---|---|
| O: 2026-11-25 count (peak < 0, peak > 0) | 55 | 25 | 21 | 15 |
| A: anatomical, L4 boundaries (primary, provisional) | 55 | 18 | 15 | 9 |
| B: module boundary +/-0.175 (sensitivity) | 55 | 18 | 15 | 9 |
| C: deep at or beyond L5 mid (sensitivity) | 55 | 18 | 15 | 9 |

- Rules A, B and C coincide because no peak bin lies between +/-0.095 and +/-0.175. The alpha-beta peak bins are
  -0.40 (15 runs), 0.00 (17), +0.05 (7), +0.25 (4), +0.30 (14). Under A, the +0.05 and 0.00 peaks are inside L4, not deep.
- The endpoint-sign test (ab - gm < 0 at the first bin, > 0 at the last) is weak and is kept as the old rule had it.
- Receipt: `docs/receipts/lansa_t1_anatomy_recount_2026-11-25.tsv`; script `scripts/lansa_t1_anatomy_recount.py`.
  The script stops unless rule O gives 21 (reproduction) and the observed map passes rule A (known positive).

## 4. The geometry result already in the manuscript draft

- `paper/jomission_lansa/sections/results.tex` (HEAD, lines 481-488): the depth placement was selected on the 11-area
  mean prototype. The "held-out" runs are other model runs scored against that same prototype. Nothing is held out by
  area. The paper marks it UNRESOLVED. Required before any geometry claim: a leave-one-area-out redo (G3 in the draft),
  or the paragraph reworded as exploratory with the selection stated.
- Observation only: the selected depths (-0.25, -0.05, +0.20) lie within 0.05 of the Allen mid-depths (-0.216, 0.000,
  +0.198). The selection was on the prototype, so this is not evidence for anatomy.

## 5. T2-T4: priority recorded, not executed

| Priority | Experiment | Sealed spec | Modifier | Note |
|---|---|---|---|---|
| 1 | SOM/PV ratio scaling | `SEAL_T2_T3.md`, T3 | `rhosX` | keeps the ramp shape and each unit's inhibitory total |
| 2 | GABA weight scaling | `SEAL_T2_T3.md`, T2 | `giX` | every GABA weight times g |
| 3 | PV/SOM silencing | `SEAL_T4.md`, T4 | `ipvUV` / `isomUV` | dose calibration first (`t4cal`) |

- Nothing run. Specs unchanged.
- Verified: the sealed reference is "the frozen T1 winner" (kd 1.41, sst 15, sstd 21), which fails the gamma-shape test.
  The spec allows this. OWNER to decide whether it stays the reference.
- Needs an owner order. Whether T2-T4 are tier-3 (AGENTS.md: tier-3 runs only inside an authorising TODO): UNKNOWN until
  checked against `manifests/test_tiers.json`.

## 6. Scientific interpretation

- Supported: a ramp-dependent control of the signed laminar contrast M. Removing or reversing the SOM/PV ramp flips
  M's sign and removes deep alpha-beta placement (O38 and T8 receipts). The sign follows superficial alpha-beta: density
  arms strengthen M by suppressing superficial alpha-beta (S_ab 0.005-0.016 against 0.07-0.08 in the reference).
- Not supported: complete reproduction of the empirical gamma profile. T1 fails the gamma-shape criterion in 57 of 57
  runs (max r_gm 0.569 against 0.6).
- Unresolved: whether the pattern is readout-generated or circuit-generated. The single-kernel diagnostic, the
  depth-placement selection, the sigma = 0.15 assumption and the unit placement all bear on this. The manuscript says a
  geometry-only account is not excluded (shift null p = 0.19).
- Not claimed: weakening. No graded perturbation was run, and "diminished" has no threshold.

## 7. Open for the owner

1. Confirm the deep definition: rule A (provisional), rule B or rule C. On this data they give the same count.
2. Confirm that the V1 Allen layer depths may stand in for the LANSA areas, or that the geometry lineage must test
   them rather than assume them.
3. Approve the geometry draft's OWNER items: the D target (L5 mid or L6 mid), the median r_gm rule, G3, and whether
   Phase 2 is authorised.
4. T2-T4: whether the reference stays the failed T1 winner, and the order to authorise them.
5. The manuscript geometry paragraph: keep as exploratory with the selection stated, or hold it until the LOAO redo.
