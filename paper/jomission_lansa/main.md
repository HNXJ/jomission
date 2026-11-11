# Manuscript plan: jomission-LANSA (flow-article, 2026-11-10)

Venue: bioRxiv preprint, New Results, Neuroscience; computational neuroscience reader (cortical circuits, LFP).
Layout after Herrera et al. 2026 (bioRxiv 10.64898/2026.11.04.763717, landing page read 2026-11-10: one-paragraph
abstract of about 190 words, context -> approach -> findings -> implication; competing-interest statement and funders
declared; full text not read). bioRxiv sets no word limit; targets used here: abstract <= 250 words, one paragraph;
sections Introduction, Results (declarative headings), Discussion, Methods, Data and Code availability,
Acknowledgements, Author contributions, Competing interests, Funding, AI-use statement, References, Supplement;
continuous line numbers. Declarations and the author list are [MISSING] until the owner supplies them.

Title (calibrated to the evidence; owner's working title "Laminar-cellular architecture sets the spectrolaminar
motif..." is not supported by a passing run and is kept as the question, not the claim):
"Testing Cell-Type Density as a Circuit Mechanism of Primate Spectrolaminar Oscillations" (owner's wording
2026-11-10, framed as a test because no run passed and the mechanism is not causally identified).

Message: a three-unit AdEx E/PV/SOM column, built from LANSA depth densities and the validated CAMINOS module,
places alpha-beta deep but leaves relative gamma weak between -0.2 and 0; size, in-degree, SOM kinetics and
per-unit weights do not close the gap, and readout controls show that depth geometry alone can mimic a fit.

## Key claims and evidence

| # | Claim | Evidence | Counter / limit |
|---|---|---|---|
| C1 | The CAMINOS module reproduces the JaxFNE reference (f 34.0 Hz, lag 7.16 ms, rates in margin) | results/laminar_e0/e0_result.json (5 seeds, PASS) | pipeline equality, not a test of the motif |
| C2 | Raising SST/PV lowers gamma frequency (46 -> 27 Hz over rho 0.25-2; slope -6.27 Hz/oct; perm p < 1e-4) | results/laminar_e1/e1_result.json | verdict INCONCLUSIVE: rho 4 does not oscillate (c3 undefined) |
| C3 | No E2 run passes the six-part scorer; T1 confirmation 0/3 (sealed rule 3/3) -> T1 FAIL | FINDINGS §2, 2b-2d; results/laminar_e2/t1_confirm_*_s{4,5,6}_T12_verdict.json | thresholds reachable: ceilings 0.854/0.693 (readout_ceiling.json) |
| C4 | Deep alpha-beta for the reference on its selection seeds (in-sample r_ab 0.83-0.86 at n4000); lost by the T1 winner (0.32-0.47 selection, 0.35-0.47 fresh seeds); r_gm < 0.6 in every T1 and ladder run | FINDINGS §2 table, §2b-2c; sstd verdicts | r_ab trades with r_gm along kd |
| C5 | Mid-depth gamma is weak: relative gamma at -0.2..0 is 0.19-0.31 (12 runs) vs 0.85 observed | data/mid_gamma.json (scripts/e2_mid_gamma.py); per-unit power §2b | missing generator is INFERRED; INFERRED |
| C6 | Middle-unit weights trade gamma against alpha-beta (exploratory, 1 seed) | FINDINGS §2d table | hand-chosen, needs sealed confirmation |
| C7 | Depth placement alone reaches r_proto 0.72, but a spectral-shift null is not rejected (p = (3+1)/21 on r_proto; 12 of 20 shifted draws had no passing placement) | FINDINGS §7, depth_placement.json | a negative control, not support |
| C8 | Event delivery: bitwise-equal outputs, 9.9x faster at T1 length | FINDINGS §5 | one seed, one layout |
| C9 | The RWS readout rests on the linear volume-conductor solution with validated weights and an assumed separable depth kernel; mid-depth deficit holds at sigma 0.10-0.25 (0.44-0.62 at 0.25); cross terms 0-12%, normalizing max at S | Methods derivation; data/readout_sensitivity.json (scripts/e2_readout_sensitivity.py, 12 runs) | r_gm is sigma-dependent; full scorer not rerun per sigma |
| C10 | Readout capacity (no simulation), sigma 0.15: max r with the observed gamma band-mean profile 0.853 (full C), 0.657 (no cross terms; S+M, no D) -> r_gm >= 0.6 reachable without coherence, so the model's r_gm failure is in its spectra; mid-depth level ceiling 0.853 at one frequency, at an optimum with S-M coherence 1.00 and D near antiphase (about -0.98; optimum not unique); r-maximizing profiles have mid 0.49 (auto) and 0.72 (full); leave-one-area-out (n 11): held-out r 0.81 +- 0.05 full (11/11 >= 0.6), 0.62 +- 0.09 auto (7/11), template 0.93 (data/readout_capacity_loao.json) | data/readout_capacity.json (scripts/e2_readout_capacity.py; self-recovery r >= 0.99999, negative r 0.07-0.14, order asserts) | limits of the observation model, not of macaque sources; an L2 fit (r 0.45) understated capacity and was corrected 2026-11-10 |
| C12 | T8 (sealed, confirmatory screen, n 3 seeds, uncorrected): primary r_ab, ref minus control: density flat +0.004 (no consistent difference), density S<->D +0.13, SOM/PV flat +1.56, reversed +1.58 (ref higher); density arms r_gm -0.83/-0.85, r_proto -0.37/-0.44; mid-depth gamma lower in all controls; one unit dominates every control map | data/t8_summary.json (scripts/e2_t8_summary.py), data/t8_engine_check.json (22/22 equal x 3 seeds), results/viz/t8/V3_t8_atlas.png | arms change sizes, in-degrees and rates with the named rule; dominant unit not the highest-rate one; launcher ran 15 at once (results bitwise equal for ref); V1 drawn after one verdict was read |
| C11 | Joint constraint, sigma 0.15: max r with mid within 0.02 of 0.848 is 0.845 with cross terms (S 46%, M 49%, D 5%, coherent), infeasible without (diagonal mid ceiling 0.665, grid-checked); 0.946 infeasible at 0.15. Circuit vs optimum: circuit gamma S 63-78%, M 17-19%, Re coherence -0.06..0.27; r_gm/r_proto equal verdict files | data/readout_joint.json (scripts/e2_readout_joint.py) | observation-model property; LOAO tests the target fit, not the mechanism; density not causally tested (needs fixed-total-count flatten/permute) |

Counter-evidence: Mackey et al. 2026 (Nat Neurosci) dispute the motif's ubiquity; our target is the 11-area
LANSA mean and V1, so the claim is conditional on that map.

## Sections

- Abstract (249 words): question, model, scorer, C1-C5, C12, C7, C10, implication.
- Introduction: laminar LFP literature (Maier 2010, Xing 2012, Buffalo 2011, van Kerkoerle 2014, Haegens 2015,
  Bastos 2015, 2020); motif and dispute; densities (LANSA); cells to rhythms (Tremblay, Borgers, Cardin, Sohal, Veit,
  CAMINOS, AdEx); currents to field (Buzsaki 2012, Einevoll 2013); Billeh/Herrera; gap; question; C12 in the summary.
- Results: R1 data and circuit (Fig 1); R2 module (C1, C2; Table 1); R3 reference stack (Figs 2-3, ceilings; readout capacity, joint constraint, circuit vs optimum);
  R4 size ladder and seeds (Table 2); R5 T1 axes and confirmation (Fig 4, Table 3); R6 per-unit weights (Fig 5);
  R6b T8 controls (C12; Table t8, Fig t8); R7 readout controls: depth placement, probe geometry, CSD (Fig 6);
  R8 engine speed (C8 extended by the T8 engine check).
- Discussion: what is shown; T8 separates ratio ramp (deep alpha-beta) from density (gamma profile); middle-unit
  coupling; limits; next sealed test (rate-matched M-gamma intervention).
- Back matter: Data and Code availability, Acknowledgements, Author contributions, Competing interests, Funding,
  AI use: all [MISSING] except the parts sourced here (JaxFNE DOI 10.5281/zenodo.21448155 from its CITATION.cff).
- Methods: existing text, connectivity filled, scorer, T1 seal, weight modifier, event delivery, figures.
- Supplement: per-model blocks, ladder maps, T1 atlases, sealed-not-run table, records table (code, seals, data, run records by identifier; no paths in the text).

## Figures

| Fig | Message | Source |
|---|---|---|
| 1 | Data and model as built (no simulation) | results/viz/schema/t1_kd1p41_n4000_s1_T8_schema.png |
| 2 | Reference run: architecture/raster/spectra/maps | results/viz/paper_w3/..._block.png |
| 3 | Scorer view of the T1 axis-1 winner | spectrolaminar/runs/t1_kd1p41_n4000_s1_T8.png |
| 4 | T1 axes: no level reaches r_gm 0.6 | results/viz/t1/V3_{kd,sst,sstd}_n4000_atlas.png |
| 5 | Weight trade-off: most-even and best-r_proto exploratory runs | spectrolaminar/runs/e2x_*.png |
| T8 | Controls: ratio ramp sets deep alpha-beta; density sets the gamma profile | results/viz/t8/V3_t8_atlas.png (scripts/e2_t8_atlas.py) |
| 6 | Readout controls: probe geometry summary; CSD (redrawn at 6.8 in, 7 pt) | paper_w5/probe_summary.png; paper_w4/..._csd.png (scripts/e2_csd.py) |
| S | seeds, ladder maps, per-run panels | paper_w6/seed_diff.png, laminar_e2/*_motif.png |
