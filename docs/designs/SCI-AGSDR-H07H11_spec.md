# H07 (locality) and H11 (CL1-A hash audit) computation on the 24 stage-0 candidates (pre-registered; sealed on commit of this file)

Status: all seven owner definitions ruled (Q-a to Q-g, 2026-11-04) and folded in below; implementation, seal and run authorised (Q12, 2026-11-04). Runner `scripts/agsdr_h07h11.py`. Basis: amendment 3 (`SCI-AGSDR-CALIBRATION_amendment_3.md`): H07 and H11 remain hard gates, not yet computed. Definitions below are quoted from `manifests/agsdr_local_objectives.json` (`H07_locality`, `H11_CL1A_PASS`); ambiguities in those definitions were put to the owner and are recorded as rulings below.

## Question
For each of the 24 stage-0 candidates (`AGS_S0_00` to `AGS_S0_23`, seed 0), does H07 hold, does H11 hold, and what is the final hard-feasible set (H01 to H10d with the H05 ruling, plus H07 and H11)?

## No simulation needed
Both gates are structural: built from `build_candidate_model(cand, seed=0)` (as the runner builds it) without running dynamics. The 2x-gain saturation test is report-only (Q-d) and is the only part that would need simulations (48 short runs); it is not part of the hard gate. Candidates differ in motif gains, tonic scale and Poisson parameters (11 coordinates), so topology varies through within/between gains only if the candidate changes edges (expected: motif gains scale weights, not edge counts; to be shown by the computation, not assumed).

## H07 terms (hard threshold quoted: "sparse_local intact: spatial_sigma 0.08, max_in_degree 25, within edges ~10000, between 590, Q>0.2, locality_index>3, not densifying or saturating")
Computed per candidate from `model.params["edge_list"]`, `positions`, `neuron_table()` areas:
1. spatial_sigma 0.08 and max_in_degree 25: read from the builder call (`cfg.metadata.within_locality`). **Ruled (Q-g): max within-area in-degree split by layer pair, observed from edges:** per post neuron, in-degree pooled over its within-area edges from non-vertical layer pairs (everything except L4->L2/3 and L2/3->L5) <= 25 (the builder cap in `_apply_spatial_locality` is per post over pooled within-area edges); per post, in-degree from vertical layer pairs <= 40 (`vertical_max_in_degree`). Canonical model: 25 and 12 (all within-area: 34, so the literal 25 on all edges would fail the reference).
2. within-area edge count and between-area edge count; thresholds "about 10000" and "590". **Ruled (Q-a): plus or minus 10 percent**: within in [9000, 11000], between in [531, 649]. Canonical model total edges 10666 (E-presyn 6995, I-presyn 3671); the split is computed here, and the canonical model must pass (control 1).
3. Q (modularity) > 0.2. **Ruled (Q-b): the areas as the partition (the canonical model has 4: FEF, PFC, V1, V4; the draft's earlier "8" was wrong)**: Newman modularity of the undirected multigraph (each directed edge counted once) with the 4 areas as the partition (Q = sum over areas of e_ii - a_i^2), using edge counts as weights.
4. locality_index > 3 where locality_index = avg_random / avg_within. **Ruled (Q-c):** avg_random = mean pairwise distance of all within-area neuron pairs (random expectation) divided by the mean distance of within-area connected pairs; > 3 means connected pairs are 3x closer than random.
5. "not densifying": within-area edge count <= 1.1 x the canonical model's (10 percent), and mean within-area connection probability <= 1.1 x canonical (pre-seal change: the draft said <= canonical, which contradicts the 10 percent on counts; same tolerance used for both).
6. "not saturating" (2x gain test). **Ruled (Q-d): not part of the hard gate; report only**, as the manifest's soft_threshold places the numeric 5 percent. Computed and reported as |bar_r(2 x within_gain) - bar_r(within_gain)| / bar_r(within_gain) > 0.05 (2000 ms, seed 0, same drive; 2 simulations per candidate, 48 total), flagged informational.
- H07 passes iff terms 1 to 5 pass.

## H11 terms (hard threshold: "CL1-A PASS: 12-gain pseudogenome + jitter + lognormal + delays + Poisson + E mixture hash-visible unchanged beyond allowed theta_local box")
Audit of the built model's metadata (no dynamics):
1. `e_mixture_version` equals "M2", and motif_gain, delays, lognormal, Poisson fields present in `cfg.metadata`.
2. Delays unchanged: FF 8, FB 12, within 2 (as `builder.py:90-97`); lognormal CV 1.5, sigma 1.08566, mu -0.58933 (`builder.py:142-144`); Poisson 2 kHz amplitude 2.0 (`builder.py:49-53`) **Ruled (Q-e):** Poisson rate and amplitude are exempt from "unchanged" (declared search coordinates) and are checked to lie inside the declared candidate box; the other fields must equal canonical.
3. The 12 motif gains: **Ruled (Q-f): H11 does not re-test the 0.7 to 1.3 ratio box** (it is the H08 box, informational by amendment 3; re-testing would make H08 hard through H11). H11 checks only structural presence and unchanged non-searched fields.
- H11 passes iff terms 1 and 2 (as ruled) pass.

## Controls
1. Canonical model: H07 terms on the canonical model (`build_jomission_model(n_per_area=100, seed=0)`) pass with the proposed tolerances (positive control; a gate that fails its reference is miscalibrated).
2. Negative control: a deliberately globalized model (within-area edges replaced by uniformly random pairs, same count) fails locality_index and Q (a check that can fail).
3. H11: a model with one searched-away field (e.g. FB delay 12 changed to 6) fails the audit.
4. Final feasible set recomputed by combining stored H01 to H10d results (`docs/receipts/agsdr_local/` and `stage0_correction_v1.json`, with the H05 all-E ruling) with H07 and H11; the Stage-0 originals are not edited.

## Output
Additive receipts under `docs/receipts/agsdr_h07h11/`: per-candidate `H07` and `H11` records, `summary.json` (spec and script sha256, per-candidate pass, the final feasible set, relation to the 2-of-24 upper bound). Nothing sealed is edited.

## Stop rules
Any control fails: report, do not interpret. If the final feasible set is empty: stop and report (design stop rule); no stage-1 action.

## Not decided here
Stage-1 authorization; the OPEN-25 viable-regime verdict; reviewer authorization.

## Owner rulings (2026-11-04)
Q-a plus or minus 10 percent; Q-b areas (4 in the canonical model); Q-c mean pair distance all vs connected; Q-d saturation report-only; Q-e Poisson exempt within box; Q-f no ratio box in H11; Q-g in-degree split by layer pair (non-vertical <= 25, vertical <= 40). Implementation and seal authorised by Q12.

## Pre-seal findings (disclosed; found by running the controls before sealing, no candidate computed)
- Control 1 FAILS on term 4. The canonical model has locality_index 1.92 under the Q-c definition (threshold > 3). Areas are boxes of 0.5 x 0.5 x 1.0 position units; the kernel width 0.08 acts on candidate edges whose layers sit at different depths, so connected pairs are only about 2x closer than random pairs. Terms 1, 2, 3, 5 pass on the canonical model (non-vertical in-degree 25, vertical 12, within 10076, between 590, Q 0.695).
- Control 2 passes (globalized: Q -0.012, locality 1.00). Control 3 passes (canonical H11 passes; FB delay 6 fails).
- Consequence under the stop rule: H07 is NOT_DETERMINED. The run computes and records every H07 term and H11 for all 24 candidates; it reports the set after H11 as an upper bound, and no final hard-feasible set. Term 4's definition or threshold needs an owner ruling (OPEN-29).
- Term 1 statistic corrected before sealing from per (post, source layer) to per post pooled, matching the builder cap; canonical value 25 (was 22 under the first coding).
