# Manuscript plan: jomission-LANSA (flow-article, 2026-11-10)

Venue: bioRxiv preprint, computational neuroscience reader (cortical circuits, LFP). Layout after Herrera et al.
2026 (SLM), data and conventions of LANSA (Mulvey et al. 2024).

Title (calibrated to the evidence; owner's working title "Laminar-cellular architecture sets the spectrolaminar
motif..." is not supported by a passing run and is kept as the question, not the claim):
"A density-informed three-unit E/PV/SOM column reproduces part of the primate spectrolaminar
motif but leaves mid-depth gamma weak".

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

Counter-evidence: Mackey et al. 2026 (Nat Neurosci) dispute the motif's ubiquity; our target is the 11-area
LANSA mean and V1, so the claim is conditional on that map.

## Sections

- Abstract: question, model, scorer, C3/C4/C5, C7, implication.
- Introduction: motif and dispute; densities (LANSA); models (CAMINOS, Herrera); gap; question.
- Results: R1 data and circuit (Fig 1); R2 module (C1, C2; Table 1); R3 reference stack (Fig 2, ceilings);
  R4 size ladder and seeds (Table 2); R5 T1 axes and confirmation (Fig 3, Table 3); R6 per-unit weights (Table 4);
  R7 readout controls: depth placement, probe geometry, CSD (Fig 4); R8 engine speed.
- Discussion: what is shown; middle-unit coupling; limits; next sealed test (rate-matched M-gamma intervention).
- Methods: existing text, connectivity filled, scorer, T1 seal, weight modifier, event delivery, figures.
- Supplement: per-model blocks, ladder maps, T1 atlases, sealed-not-run table.

## Figures

| Fig | Message | Source |
|---|---|---|
| 1 | Data and model as built (no simulation) | results/viz/schema/t1_kd1p41_n4000_s1_T8_schema.png |
| 2 | Reference run: architecture/raster/spectra/maps; observed vs model | results/viz/paper_w3/..._block.png; spectrolaminar/runs/t1_kd1p41... |
| 3 | T1 axes: no level reaches r_gm 0.6 | results/viz/t1/V3_{kd,sst,sstd}_n4000_atlas.png |
| 4 | Weight trade-off: most-even and best-r_proto exploratory runs | spectrolaminar/runs/e2x_*.png |
| 5 | Readout controls: probe geometry summary; CSD | paper_w5/probe_summary.png; paper_w4/..._csd.png |
| S | seeds, ladder maps, per-run panels | paper_w6/seed_diff.png, laminar_e2/*_motif.png |
