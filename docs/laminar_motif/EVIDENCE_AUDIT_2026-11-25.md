# LANSA evidence audit of existing perturbation results (2026-11-25)

Status: AUDIT of existing receipts. No simulation, no new seal, no change to any verdict, criterion or the abstract. Abstract wording (D1) stays pending, per the owner.
Base: `dev@a826ce6`. Thesis: owner, 2026-11-25 (mechanism explains the superficial–deep spectral motif; parameter exploration is the method).
Receipts marked **cowork** are in `/mnt/d/cowork/jomission/results/` and are not in this repo. Their checksums are in `docs/receipts/lansa_M_audit_2026-11-25_t8_sha256.txt`.
Labels: OBSERVED (read from a file), DERIVED (computed here from a file), INFERRED, UNKNOWN.

## 0. Correction (2026-11-25)

- §3 overstated the gamma-only failure. Corrected in place below.
- The qualitative count of 21 runs (peak gamma superficial, peak alpha-beta deep, endpoint crossing) used peak > 0 as
  "deep". Under the frozen anatomical rule (L4 lower boundary, z > +0.095) the count is 15, and 9 also pass
  crossing_2se. These are not 21 replications. Recount: `docs/laminar_motif/DECISIONS_2026-11-25.md` section 3.
- Depth convention verified: `DECISIONS_2026-11-25.md` section 2.
- The geometry paragraph in `paper/jomission_lansa/sections/results.tex` (lines 481-488) selects depths on the observed
  prototype. It is not held-out evidence: `DECISIONS_2026-11-25.md` section 4.

## 1. Thesis layers against the record

| Layer | Claim | Status | Evidence |
|---|---|---|---|
| Foundation | one CAMINOS module reproduces its published rhythm | PASS (E0) | `results/laminar_e0/e0_result.json` |
| A: rate ingredient | higher SST/PV ratio lowers peak frequency, isolated module | UNDECIDED: E1 INCONCLUSIVE; E1b arm A INCONCLUSIVE, arm B PASS (confounded, §3) | `results/laminar_e1/e1_result.json`, `results/laminar_e1b/e1b_result.json` |
| B: composition, deep alpha-beta | the prescribed SOM/PV ramp places alpha-beta deep | SUPPORTED for the reference (r_ab 0.82–0.87 over 5 seeds); lost in every non-ramp arm (O38) | `results/laminar_e2/o38s_pc_ref_*`, `o38s_s6_analysis.json` |
| B: composition, superficial gamma | the motif's gamma shape | FAILED: T1 (r_gm ≤ 0.57 in all 57 runs; threshold 0.6; other criteria fail in subsets, §3) | cowork `results/laminar_e2/t1_*_verdict.json` |
| C: density | cell-count depth profile sets the motif | Flattening/reversal keeps or strengthens the canonical sign (T8 dflat, drev) | cowork `results/laminar_e2/t8_*_motif.json` |
| D: connectivity and inhibition | GABA weight, SOM/PV ratio scale, PV/SOM silencing, +50 pA | NOT RUN. Sealed (S5, S6), no output in the repo or cowork | `docs/laminar_motif/SEAL_T2_T3.md`, `SEAL_T4.md` |
| E: apical dendrites | SST inhibition of apical dendrites | NOT MODELLED. No compartments in the code | §5 item 5 |

## 2. Experiment inventory

| ID | Seal | Manipulation | Result (receipt) | Where | Proposed role |
|---|---|---|---|---|---|
| E0 | S1 | single module, baseline | PASS | repo | baseline |
| E1 | S2 | SST/PV ratio, isolated module, seeds 1–3 | INCONCLUSIVE (c1 PASS, c2 PASS, c3 UNDEFINED, c4 PASS) | repo | essential to A |
| E1b | S8 | fixed-PV arm B, and arm A | arm A INCONCLUSIVE, arm B PASS | repo | essential to A (post-hoc design, total inhibition varies in arm B) |
| T1 | S4 | motif fine-tune, 3 axes, 18 levels × 3 seeds + confirmation seeds 4–6 | FAIL: 57 of 57 verdicts `pass: false` | cowork (R3, R7 not in repo) | essential to B |
| T8 | S7 | density arms (dflat, drev) and ramp arms (rflat, rrev) | density keeps the sign; ramp reverses it (§4) | cowork (R11 not in repo) | essential to B and C |
| O38 | pre-reg, not in the paper | 8 composition arms (ramp flat, ramp reversed, four monkey-profile areas, shape, PV-CHC), seeds 1–5 | reference canonical; all 8 arms lose r_ab (d ≈ 1.56) and reverse M; analysis accepted by owner 2026-11-17 | repo (`results/laminar_e2/o38s_*`) | essential to B, not cited in the paper |
| L-FS | S3 | finite-size ladder | not audited this round | — | prerequisite |
| T2 | S5 | every GABA weight × g | sealed, not run | — | connectivity (D) |
| T3 | S5 | SOM/PV ratio × s per unit | sealed, not run | — | ratio (D) |
| T4 | S6 | PV or SOM silencing, +50 pA | sealed, not run | — | causal (D) |
| T5 | none | VIP | deferred (AdEx has no VIP) | — | secondary |

Ordering of essentiality is my proposal. The owner decides which predictions are essential to the central claim (§8).

## 3. E1 and T1 against the central claim

**E1 (layer A).** Observed: `f_pk` falls with rho over 0.25–2 in every seed (slopes −6.6, −5.9, −6.3 Hz per log2 rho; mean −6.27). At rho = 4, two seeds have excess below 4 (3.04, 3.26) and are non-oscillating under both readings. Seed 3 has excess 4.78 (≥ 4) but a stored flag `oscillates: false`; `E1B_SEAL.md` lines 8–10 say the sealed 20–80 Hz band flagged all three rho = 4 peaks at their 20 Hz edge. E1's own text defines oscillation as excess ≥ 4 (`E1_SEAL.md` line 26). So the stored flag applies a band-edge rule the E1 seal does not state.
- c3 (mean drop ≥ 3 Hz) is UNDEFINED because rho = 4 has no defined `f_pk` for the mean. Under either reading of seed 3, seeds 1 and 2 keep c3 undefined, so the verdict stays INCONCLUSIVE.
- The E1 seal's verdict rule has no branch for "c1 PASS, c3 UNDEFINED". The label is assigned by code, not by the sealed text. That is a rule gap, not a changed criterion.
- Reading: the rate ingredient is neither supported nor falsified. E1b arm B continues the decrease at rho = 4 (24–26 Hz, abstract), but total inhibitory count co-varies, so the effect is not attributable to composition.

**T1 (layer B, motif shape).** Verified in cowork: 54 search-level verdicts (18 non-reference levels × 3 seeds) and 3 confirmation verdicts (seeds 4–6, 12 s) are all `pass: false`. Confirmation r_proto = 0.377, 0.365, 0.377.
- The gamma criterion is the only one that every run fails: maximum r_gm 0.569 (threshold 0.6). It is necessary and
  never met, so it is sufficient to fail T1 on its own.
- The other criteria fail in subsets of the 57 runs (DERIVED from the cowork verdict files): r_proto not above
  max(smooth p95, identity control) in 55; identity rank not 1 in 38; crossing_2se false in 32; r_ab below 0.6 in 31;
  halves disagree in 7.
- Reference (O38 seed 1, `o38s_pc_ref_n4000_s1_T8_verdict.json`): crossing true, r_ab 0.86 PASS, r_gm 0.33 FAIL, r_proto 0.33 below smooth p95 0.38 FAIL, identity control −0.08 PASS, halves agree PASS.
- Corrected 2026-11-25: "the deep alpha-beta placement and the sign of the contrast are not the failing part" holds for
  the O38 reference only. Across the 57 T1 runs, r_ab and crossing fail in about half. Gamma shape is the necessary
  failure, not the only one.

**Consequence.** T1 is an essential failure of the motif claim (title and abstract). It cannot be bypassed by rewording the abstract. E1 is an essential, undecided test of the rate ingredient. D1 stays pending until both are reconciled with the claim, as the owner ruled.

## 4. The signed contrast M (proposed observable)

**Computability.** The pipeline already computes `contrast_M`, a depth slope of (alpha-beta − gamma) on the stored image-band relative-power profiles. I recomputed the proposed log-ratio form from the same profiles (S = depth < −0.175, D = depth > +0.175). Both are in `scripts/e2_m_audit.py`. The pipeline value is reproduced in 62 of 62 files (47 tracked O38, 15 cowork T8). The two forms agree in sign for every file. Magnitudes differ.

**Per-arm summary** (`docs/receipts/lansa_M_audit_2026-11-25.tsv`; means over seeds, DERIVED):

| Arm | n | S_gm | S_ab | D_gm | D_ab | M proposed | M pipeline |
|---|---|---|---|---|---|---|---|
| reference (O38 pc_ref) | 5 | 0.41 | 0.08 | 0.76 | 0.76 | +1.68 (+1.56 to +1.88) | +0.54 (+0.44 to +0.70) |
| reference (T8) | 3 | 0.41 | 0.07 | 0.76 | 0.82 | +1.78 | +0.62 |
| density flat (dflat) | 3 | 0.17 | 0.02 | 0.83 | 0.84 | +2.44 | +0.25 |
| density reversed (drev) | 3 | 0.07 | 0.01 | 0.82 | 0.83 | +2.65 | +0.10 |
| ramp flat (rflat, pc_flat) | 3–5 | 0.58 | 0.60 | 0.44 | 0.12–0.14 | −1.37 to −1.18 | −0.45 to −0.42 |
| ramp reversed (rrev, pc_rev) | 3–5 | 0.58 | 0.58 | 0.10–0.14 | 0.01 | −2.45 to −2.52 | −0.11 to −0.16 |
| composition arms v1, mt, mst, teo, chc, shape | 5 each | 0.58 | 0.57 | 0.23–0.38 | 0.07–0.16 | −0.44 to −1.51 | −0.15 to −0.47 |

**Denominator decomposition (the check you asked for).** The sign of M follows the superficial alpha-beta term:
- S_ab is 0.07–0.08 in the reference, and 0.005–0.016 in both density arms.
- S_ab is 0.57–0.60 in every ramp and composition arm.
- S_gm is about 0.58 in all ramp and composition arms, and 0.17 or below in the density arms.
- So the sign change is a switch in superficial alpha-beta. The gamma numerator does not rise to produce canonical dominance. The density arms gain M by suppressing S_ab, not by raising gamma.

**Graded or binary?** No intermediate values. Across the 46 ramp and composition runs (O38 flat, reversed and six composition arms; T8 rflat and rrev), every M is negative. The nearest to zero is v1 (pipeline max −0.08, proposed max −0.19). The density runs are positive, so they are not part of this statement. "Diminished" has no threshold in any file, so "weakened" is undefined until one is declared.

## 5. Conflicts and corrections to the current narrative

1. **Reproduction.** "The model reproduces the motif in particular regimes" conflicts with the abstract ("did not reproduce the primate motif") and with T1 FAIL. Reconciled: the sign of M and the deep alpha-beta placement are reproduced under the prescribed ramp, and the superficial gamma shape is not. The thesis needs that qualifier.
2. **Weakening and reversal.** "Perturbing ratios and connectivity … can produce weakening or reversal": the receipts show reversal under ramp arms (rflat, rrev, and the O38 flat and reversed arms) and strengthening under density arms. No graded weakening was observed. Connectivity and silencing (T2–T4) have no outputs.
3. **Abstract vs O38.** The abstract says flattening or reversing "did not shift single bands". The O38 analysis of the same ramp arms shows loss of deep alpha-beta placement (r_ab +0.85 → −0.73, d ≈ 1.56) and sign reversal of M. Both can hold, since they are different observables, but the paper cites only the band-level result. The O38 analysis is not in the paper.
4. **E1.** The sealed verdict rule has no branch for c3 UNDEFINED. The stored oscillation flag disagrees with the seal's excess ≥ 4 definition for seed 3, and E1b's seal attributes the flag to the 20 Hz band edge. The verdict stays INCONCLUSIVE under either reading.
5. **Apical dendrites.** "SST-mediated inhibition of apical dendrites" is not modelled. No compartments exist in `jomission/` or `scripts/`. The CSD forward model places the SST kernel at a fixed apical depth (`scripts/e2_csd.py:51–53`), which is an assumption, not a result.
6. **Name collision.** `manifests/t1_t7_*` are omission-paradigm targets T1–T7. They are not LANSA T1.
7. **Provenance.** T1 records (R3, R7), T8 records (R11) and most other run records (R4–R6, R8–R10, R12) are not in the repo. The paper's D6 hashes them, but T1 and T8 cannot be reproduced from this repo alone.
8. **Duplicates.** `o38s_pc_ref_n4000_s1`, `o38s_ref_n4000_s1` and `o38s_nc_n4000_s1` carry the same contrast value (M +1.574, pipeline +0.436). The 47 tracked files therefore hold at most 45 distinct runs. I checked the contrast value only, not every field.

## 6. Not done

- No runs, sweeps, seals or edits to the abstract, verdicts or criteria.
- L-FS (S3) not audited. The T8 density arms were not checked against the scorer. The cowork checkout's integrity was not verified (HEAD 21b2697, dirty).
- M uses the 21 stored bins with seed spread only. Manuscript bands (8–30, 31–150 Hz) are stored in the scores and were not decomposed.

## 7. Recommendations (no action taken)

1. Decide how E1 classifies "c1 PASS, c3 UNDEFINED". This is an owner decision. A change to the band would start a new lineage.
2. Decide whether M is a secondary observable, which form, and a threshold for "diminished", before any further use.
3. Any manuscript that reports M must also report the S_ab decomposition, since S_ab sets the sign.
4. Bring the T1 and T8 records into the repo, or seal their hashes with access terms, so T1 FAIL can be reproduced from this repo.
5. T2–T4 stay sealed and unrun. Running them needs an owner order, and it is a new step, not part of this audit.
6. Keep D1 pending. The O38 result and the E1 rule gap belong in the reconciliation, not outside the record.

## 8. Owner decisions needed

- Q1: Does the proposed log-ratio M replace the pipeline's slope-form `contrast_M`, or sit beside it?
- Q2: How should E1's undefined-c3 case be classified?
- Q3: Which predictions are essential to the central claim? Proposed: T1 gamma shape, the ramp necessity from O38, and the E1 rate ingredient.
