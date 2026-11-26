# LANSA decisions, 2026-11-25: record (revision 2)

Status: owner decisions of 2026-11-25 and the geometry checkpoint ruling. Done: depth convention checked; recount receipt
regenerated; independent review (REVISE, addressed in this revision); Phase 1 protocol text frozen for review.
Not done and not authorised: merge into dev, any scientific run, any change to T1 criteria, verdicts or receipts.
D1 (abstract) deferred. The T1 FAIL and the E1 INCONCLUSIVE stay in the record.

## 1. Ruling applied

| Item | Ruling | Applied here |
|---|---|---|
| Deep threshold | Anatomical L4 lower boundary, for the new anatomical recount; original T1 classification unchanged | z > +0.095. The transmitted ruling has blank values; +0.095 is the value in section 2. OWNER to confirm. |
| D target | L5 mid primary; L6 sensitivity | L5 mid +0.198 primary; L6 mid +0.406 sensitivity |
| Gamma criterion | Keep the existing shape criterion; do not replace it with the qualitative test | r_gm >= 0.6, unchanged |
| Geometry grid | Smallest grid that separates placement, width and interaction | 2 x 2 plus interaction (protocol, section 5) |
| Phase 2 | Not authorised; complete readout-only Phase 1 first | Not authorised |
| T2-T4 reference | Keep the sealed T1 reference for comparability | Kept |
| Checkpoint 53a3e45 | Retain as an unaccepted review checkpoint; no merge | Retained; this revision is a new commit on a new branch |

## 2. Depth convention: coordinate verified; layer source is a mouse model

- Verified: LANSA bins are cortical-depth fractions about mid-L4, negative superficial (`docs/laminar_motif/data/lansa_density_X.json`,
  `units`; `docs/laminar_motif/PLAN_PAPER_JOMISSION_LANSA.md` line 29).
- Verified by recomputation: z = (um - 431) / 779 gives L4 -0.095 to +0.095; L2/3 mid -0.216; L5 mid +0.198; L6 mid +0.406.
  `scripts/allen_v1_figure.py` line 35 (THICK_UM = 779) agrees.
- Verified: the layer table is in `docs/laminar_motif/ALLEN_V1_REPLICA_SPEC.md` section 1. That spec is RETIRED (owner,
  2026-11-17; banner, lines 3-7). Its source is a mouse V1 model: `AllenInstitute/oscillations-biorealistic-mouse-v1-model`
  (source tag A, spec line 22; `docs/laminar_motif/OPEN38_ALT_ALLEN_INVENTORY.md`, title "mouse V1 model"). The spec prose
  never says "mouse"; the species is taken from the repo name and the inventory title.
- Not verified: the P2 inventory is said to re-verify the source table, but it does not repeat the layer depths (no match
  for 169, 357, 505 or 779). The depths were not re-checked against the model's files.
- UNKNOWN: whether any mouse-model depth applies to the macaque LANSA areas. The repo holds no macaque L4 boundary; the
  import scripts carry bin centres only.
- Observed: the 3-unit depths S -0.40, M 0.00, D +0.30 are hard-coded (`jomission/laminar/caminos_stack.py` line 246) and
  listed as "agent choices" (`PLAN_PAPER_JOMISSION_LANSA.md` lines 134-135). On the mouse-model scale S -0.40 is 119 um,
  inside L1 (50-169 um).
- Conflict: the LANSA module rule (|z| <= 0.175; `SPEC_OPEN38_SPECTRUM_SENSITIVITY_2026-11-16.md` line 164) does not match
  the +/-0.095 L4 bounds. That row says the boundaries are "not the same" and the L4 convention is UNKNOWN.

## 3. Recount under the anatomical rule (receipt `docs/receipts/lansa_t1_anatomy_recount_2026-11-25.tsv`)

| Rule | Gamma superficial | Alpha-beta deep | All three | All three and crossing_2se |
|---|---|---|---|---|
| O: 2026-11-25 chat count (peak < 0, peak > 0) | 55 | 25 | 21 | 15 |
| A: L4 boundaries (applied threshold, provisional) | 55 | 18 | 15 | 9 |
| B: module boundary +/-0.175 | 55 | 18 | 15 | 9 |
| C: deep at or beyond L5 mid | 55 | 18 | 15 | 9 |

- A, B and C give identical counts. No alpha-beta peak bin lies between +0.095 and +0.25, and peaks sit on a 0.05 grid.
  They are threshold checks, not a sensitivity analysis. The earlier word "sensitivity" is withdrawn.
- The rules were written on 2026-11-25 and not committed before the run. The earlier phrase "frozen before the run" is
  withdrawn.
- Reviewer check: the script output from the checkpoint is identical to the receipt except the header, which read
  `*_qx_all3` but held the all-three flag. The header now reads `*_all3`. Rule O reproduces 21; the observed map passes rule A.
- Result: 15 of 57 runs meet the anatomical qualitative pattern; 9 also pass crossing_2se. These are descriptive counts,
  not confirmations of the empirical motif.
- Alpha-beta peak bins: -0.40 (15 runs), 0.00 (17), +0.05 (7), +0.25 (4), +0.30 (14).

## 4. Independence and leakage

- The existing spectra are not independent of the observed prototype Y.
  - sst 15 ms is the T1 axis-2 winner by seed-mean r_proto against the 11-area mean prototype
    (`docs/laminar_motif/FINDINGS_2026-11-10.md` line 105). The frozen values kd 1.41, sst 15, sstd 21 are at line 137.
    `SEAL_T1.md` itself gives the grid and the selection rule, not the values.
  - The reference configuration `sw_n4000_dens_k1_kd1_sst15` and the finite-size receipts use sst 15. Whether sst 15 had a
    pre-Y origin (the E1 ruling) is not verified here.
- Depth placement (`paper/jomission_lansa/sections/results.tex` lines 481-488; `scripts/e2_depth_placement.py`):
  `select_on` = `sw_n4000_dens_k1_kd1_sst15_s1_T8`. The "held_out" entries in `depth_placement.json` are the `lfs_*` runs,
  scored against the same Y. Nothing is held out by area.
- The paper's UNRESOLVED comment (results.tex, about line 485) flags the untracked `depth_placement.json`. It does not flag
  the held-out defect, which is unflagged in the text.
- Consequence: no existing receipt set supports a leakage-free area-held-out claim. Leave-one-area-out keeps the geometry
  choice out of the held-out area, but the spectra carry Y-based selection of sst, kd and sstd. The anatomical condition G1
  has no fitted geometry and is the least exposed, not exposure-free.
- Observation only: the selected depths (-0.25, -0.05, +0.20) lie within 0.05 of the mouse-model mids (-0.216, 0.000, +0.198).
  The selection used Y, so this is not evidence for anatomy.

## 5. Provenance (cowork, `/mnt/d/cowork/jomission`)

- The 57 T1 motif JSON files equal HEAD 21b2697 byte-for-byte (57 of 57, sha256). Their scratch copies also match.
  Verdict JSON files were not compared to HEAD.
- Working tree: 51 tracked T1 paths are deleted (36 `.json`, 6 `.log`, 9 `.png`). All 51 are symlinks at HEAD (mode 120000,
  checked for all 51). 33 targets exist in the working tree. The other 18 are `t1_sst15_*` links, themselves deleted, which
  resolve through a second hop to `sw_n4000_dens_k1_kd1_sst15_*` files at HEAD. None of the 57 search receipts references a
  deleted path. Nothing was restored, because the working tree belongs to another agent. Restoring or committing the links
  is owner decision 6 in section 9, the same as handoff section 5, item 5 (the naming and provenance choice). Whether any script resolves the `t1_*` names is UNKNOWN.
- Phase 1 would read run records through `unit_spectra` (units, dt, depth). It must read them from the committed HEAD
  (`git show`), not from the dirty working tree.
- JaxFNE, from the per-run `jaxfne_version` fields: T1 axis 1 and 2 (36 records) are 0.5.8. T1 axis 3 (18), the three
  confirmation runs, the reference `sw_*` records and all 47 O38 records on dev are 0.5.9. E1's per-run records are 0.5.9.
  `docs/AUTHORITY_RULING_2026-11-24.md` admits 0.5.9 only for E0 and E1b (lines 15-16); line 16 marks every other family NOT COVERED. The E1 seal (`E1_SEAL.md` lines 5 and 13) and `HANDOUT_GEMINI.md` line 25 name 0.5.9 for E1, which is not E1b. That conflict is stated here so that no reader takes the E1 seal as admitting 0.5.9. The axis-1 reference level mixes versions (seed 1 at 0.5.9, seeds 2 and 3 at 0.5.8). OPEN-37 is still open (`docs/OPEN.md` line 63). The T1 and `sw_*` counts come from cowork HEAD 21b2697 only. Owner decision 9 in section 9, now resolved by the 2026-11-25 ruling (`docs/AUTHORITY_RULING_2026-11-25_JAXFNE059.md`, at 3739b75): admitted by family scope with the 2026-11-24 conditions; registered at d057609 (`manifests/current_state.json`, `docs/OPEN.md` row 8); the owner word is relayed from jxm-dev's session, not seen here. T8 not covered.

## 6. T2-T4

| Priority | Experiment | Sealed spec | Modifier | Note |
|---|---|---|---|---|
| 1 | SOM/PV ratio scaling | `SEAL_T2_T3.md`, T3 | `rhosX` | keeps the ramp shape and each unit's inhibitory total |
| 2 | GABA weight scaling | `SEAL_T2_T3.md`, T2 | `giX` | every GABA weight times g |
| 3 | PV/SOM silencing | `SEAL_T4.md`, T4 | `ipvUV` / `isomUV` | dose calibration first (`t4cal`) |

- Nothing run; specs unchanged. The reference stays the sealed T1 reference, per the ruling.
- Test tier: OPEN. `manifests/test_tiers.json` (about lines 98-116) places `tests/test_t2_readiness.py` and
  `tests/test_t4_t5_analysis.py` under the heading "simulating, no results writes (tier 2 covers; cost outside tier-1
  budget)". That heading names tier 2 for this group. The tier of the T2-T4 experiments themselves is not stated, so it
  stays open.

## 7. Scientific interpretation

- Supported: ramp-dependent control of the signed contrast M. Removing or reversing the ramp flips the sign and removes deep
  alpha-beta placement (O38 and T8 receipts). The sign follows superficial alpha-beta. From `docs/receipts/lansa_M_audit_2026-11-25.tsv`
  (S_ab means): O38 reference 0.079 and T8 reference 0.075; density flattened 0.016, reversed 0.005; ramp flattened 0.59,
  ramp reversed 0.58, composition 0.58 to 0.60. Density arms strengthen M by suppressing superficial alpha-beta.
- Not supported: complete reproduction of the empirical gamma profile. T1 fails the gamma criterion in 57 of 57 runs
  (max r_gm 0.569 against 0.6).
- Unresolved: readout-generated versus circuit-generated, and the independence question in section 4.
- Not claimed: weakening. No graded perturbation was run, and "diminished" has no threshold.

## 8. Gates and review (checkpoint 53a3e45)

- `scripts/project_check.py --offline --no-tests` does not pass in either interpreter tried:
  - system python: `jaxfne` missing.
  - `/home/hamm/.venvs/jaxfne`: `PROJECT_CHECK_FAIL: jaxfne identity: installed 0.5.9, execution authority 0.5.8`; the
    `jomission` package is not importable there (the check asks for an editable install). The manifest's next authorised
    task is `SCI-AGSDR-CALIBRATION`, and its current gate `V2_LOCAL_OPERATION` is FAIL. None of this comes from the
    checkpoint's files, and nothing was changed to get past it.
- Later, 2026-11-25: gate fixed by option A' (venv `/home/hamm/.venvs/jomission058`, jaxfne 0.5.8 from tag v0.5.8). Quick
  contract PASS (rc 0) at ef0a766, reported by jxm-dev and re-run by the reviewer; `V2_LOCAL_OPERATION` still FAIL. The
  bullets above describe the checkpoint, not the current state. Registration of the ruling is at d057609. A clean gate run at
  d057609 needs `/mnt/d/jomission` (Hamm's call); the clone run fails only on the editable install, as expected.
- Independent review: REVISE, minor items, no wrong numbers. Addressed in revision 2: the retired source and mouse species;
  A, B and C identity; the "frozen" wording; the misattributed UNRESOLVED comment; the header; the PLAN citation; the tie rule.
  Still open: the T2-T4 tier.
- Independent review of revision 2 (1591c18), by the jxm-dev reviewer: ACCEPT, two minor findings (rules B and C labelled
  "sensitivity"; the T2-T4 tier wording), both fixed in d4aa932 (`docs/receipts/lansa_geom_rev2_review_2026-11-25.md`).
  Not checked by that reviewer: the protocol's claim of 57 `.npz` files in cowork, and the subset counts in
  EVIDENCE_AUDIT section 3 (55, 38, 32, 31, 7).

## 9. Owner decisions needed

1. Independence route. R1: accept the spectrum-level leakage and label Phase 1 exploratory (no held-out claim). R2: new
   spectra with parameters fixed before any comparison with Y (a new lineage, with authorisation and a run).
2. Phase 1 registration. The manifest's next authorised task does not name it. Execution needs an authorising entry.
3. Macaque layer boundaries. A macaque source (the LANSA histology methods, not in the repo) is needed. Until then the
   recount is a mouse-model-convention recount.
4. Confirm the values applied from the blank ruling: +0.095, +0.198, +0.406.
5. sigma_alt for the 2 x 2 design. The protocol proposes 0.25, with no data basis. OWNER to choose.
6. Cowork deletions: restore, or commit the records. Owner decides; not done here.
7. The T2-T4 test tier.
8. The manuscript geometry paragraph: reword as exploratory, or hold it.
9. JaxFNE admissibility, resolved by the 2026-11-25 ruling (`docs/AUTHORITY_RULING_2026-11-25_JAXFNE059.md`, at 3739b75).
   T1 axis 3, the three T1 confirmation runs (the FAIL), the reference `sw_*` records, all 47 O38 records on dev and E1's
   per-run records are JaxFNE 0.5.9; the ruling admits them by family scope with the 2026-11-24 conditions. The owner word
   is relayed from jxm-dev's session; the owner word ("confirmed, register the ruling") is relayed from jxm-dev's session and was not seen here; the ruling is registered at d057609. Still open: whether the axis-1 reference mix
   (seed 1 at 0.5.9, seeds 2-3 at 0.5.8) matters scientifically, and whether the E1 seal and handout wording (0.5.9 for E1)
   needs correcting. T8 not covered.
