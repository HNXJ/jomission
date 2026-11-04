# H07 (locality) and H11 (CL1-A hash audit) computation on the 24 stage-0 candidates (DRAFT; not sealed; no run authorized)

Status: draft for owner review (Q11, 2026-11-04: spec first, no run yet). Basis: amendment 3 (`SCI-AGSDR-CALIBRATION_amendment_3.md`): H07 and H11 remain hard gates, not yet computed. Definitions below are quoted from `manifests/agsdr_local_objectives.json` (`H07_locality`, `H11_CL1A_PASS`); items marked **[OWNER]** are ambiguities in those definitions that I will not resolve silently.

## Question
For each of the 24 stage-0 candidates (`AGS_S0_01` to `AGS_S0_24`, seed 0), does H07 hold, does H11 hold, and what is the final hard-feasible set (H01 to H10d with the H05 ruling, plus H07 and H11)?

## No simulation needed
Both gates are structural: built from `build_candidate_model(cand, seed=0)` (as the runner builds it) without running dynamics, plus one 2x-gain saturation check (H07, last term) that needs short simulations. Candidates differ in motif gains, tonic scale and Poisson parameters (11 coordinates), so topology varies through within/between gains only if the candidate changes edges (expected: motif gains scale weights, not edge counts; to be shown by the computation, not assumed).

## H07 terms (hard threshold quoted: "sparse_local intact: spatial_sigma 0.08, max_in_degree 25, within edges ~10000, between 590, Q>0.2, locality_index>3, not densifying or saturating")
Computed per candidate from `model.params["edge_list"]`, `positions`, `neuron_table()` areas:
1. spatial_sigma 0.08 and max_in_degree 25: read from the builder call; **max in-degree within area** computed from edges must be <= 25 (observed, not just the argument).
2. within-area edge count and between-area edge count; thresholds "about 10000" and "590" **[OWNER Q-a]: tolerance?** Observed on the canonical model: total edges 10666 (E-presyn 6995, I-presyn 3671); the within/between split is computed here. Proposed tolerance: within in [9000, 11000], between in [531, 649] (plus or minus 10 percent), stated before seeing candidate values.
3. Q (modularity) > 0.2 **[OWNER Q-b]: partition?** Proposed: Newman modularity of the undirected weighted graph with the 8 areas as the partition (Q = sum over areas of e_ii - a_i^2), using edge counts as weights.
4. locality_index > 3 where locality_index = avg_random / avg_within **[OWNER Q-c]: definition of "avg".** Proposed: mean pairwise distance of all within-area neuron pairs (random expectation) divided by the mean distance of within-area connected pairs; > 3 means connected pairs are 3x closer than random.
5. "not densifying": within-area edge count <= 1.1 x the canonical model's (10 percent), and mean within-area connection probability <= canonical.
6. "not saturating" (2x gain test, soft_threshold lists Delta_r > 5 percent per 2x gain): **[OWNER Q-d]: is the saturation test part of the hard gate?** The hard_threshold text says "not densifying or saturating"; the soft_threshold carries the numeric 5 percent. Proposed: part of H07, computed as |bar_r(2 x within_gain) - bar_r(within_gain)| / bar_r(within_gain) > 0.05 (2000 ms, seed 0, same drive); needs 2 simulations per candidate (48 total).
- H07 passes iff terms 1 to 6 pass.

## H11 terms (hard threshold: "CL1-A PASS: 12-gain pseudogenome + jitter + lognormal + delays + Poisson + E mixture hash-visible unchanged beyond allowed theta_local box")
Audit of the built model's metadata (no dynamics):
1. `e_mixture_version` equals "M2", and motif_gain, delays, lognormal, Poisson fields present in `cfg.metadata`.
2. Delays unchanged: FF 8, FB 12, within 2 (as `builder.py:90-97`); lognormal CV 1.5, sigma 1.08566, mu -0.58933 (`builder.py:142-144`); Poisson 2 kHz amplitude 2.0 (`builder.py:49-53`) **[OWNER Q-e]:** candidates vary Poisson rate and amplitude (coordinates), so "unchanged" cannot hold literally for them. Proposed: Poisson rate and amplitude are exempt (they are declared search coordinates) and are checked to lie inside the declared candidate box; the other fields must equal canonical.
3. The 12 motif gains: ratios to canonical lie inside the box 0.7 to 1.3 **[OWNER Q-f]:** this is exactly the H08 box, which amendment 3 made informational (0 of 24 inside). Proposed: H11 does not re-test the ratio box (it would make H08 hard through H11); H11 checks only structural presence and unchanged non-searched fields.
- H11 passes iff terms 1 to 3 (as ruled) pass.

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

## Owner questions (6)
Q-a tolerances for within/between edge counts; Q-b modularity partition; Q-c definition of avg_random/avg_within; Q-d saturation test in the hard gate; Q-e Poisson exemption in H11; Q-f ratio box in H11.
