# OPEN-38 composition table, monkey #1 (portal sample `monkey_f001`), 2026-11-16

Status: provisional, one animal (owner ruling 2026-11-16: provisional circuit modeling, animal-level test deferred).
Not a confirmatory result; the 119 sections and 87 clusters are not replicates. 7A stays CONFLICT.

## Build

`python scripts/open38_laminar_composition.py CLUSTERS ALL_CELLS OUT` (commit on `_temp_lansa-provenance`).
Inputs (outside the repo, `D:/_sources_open38/chen2023_portal/`): 87 per-cluster Cell Summation exports and the
all-cells export `cell_summation_monkey_f001_ALL.csv` (sha256 c61d0d96…, identical to the owner's upload).
Output `derived/composition_f001.csv` (sha256 33545e31cbeb6355de2a8ce5efe83bb0ce9321c08953583991b91f10658d10d3)
and `composition_f001.receipt.json` (per-file sha256, layer coverage). Not committed: ND licence (owner to decide).

## Gate: complete, internally consistent table — PASS

| Check | Result |
|---|---|
| clusters per subclass = Table S2 | SST 32, PVALB 32, PV_CHC 5, VIP 9, VIP_RELN 9 (87) |
| one region set in all 88 exports | 1002 regions each |
| unique identity | no two nonzero exports share bytes or counts |
| zero exports | exactly 8, all declared: SST.23, SST.25, PVALB.19, PVALB.28, PVALB.30, PVALB.32, PV_CHC.5, VIP.9. These are the 8 GABA types missing from Chen's per-region density table (owner-supplied `cell_density.csv`, 226 types, sha256 7316eddc…): typed in snRNA-seq, not assigned in Stereo-seq |
| cluster and subclass ≤ all cells, every region | holds |
| areas | 143 (= Table S1) |
| layer coverage (Σ layer rows / area row, all cells) | 1.000 in 142 areas; V4 0.407 (PENDING area; its area row holds cells outside its layer rows) |
| missing modules | M missing in 47 areas (agranular, e.g. F1-F7, 24, 25, 32, insula, entorhinal); kept missing, never 0 |

Portal label for chandelier cells is `GABA PV_CHC.n` (Table S2 abbreviation `GABA CHC.n`).

## Four mapped areas (AREA_LABEL_TABLE PASS)

R = N_SST / N_PVALB per module (S = l2+l3, M = l4, D = l5+l6); Δ = log2(R_D / R_S). Counting-noise 95% intervals
are in the CSV (width ≤ ±7% of R); they carry no between-animal uncertainty. "+CHC": PV_CHC added to the PV pool.
fVIP: (VIP + VIP_RELN) / all cells.

| Area | R_S | R_M | R_D | Δ | Δ +CHC | fVIP S | fVIP M | fVIP D |
|---|---|---|---|---|---|---|---|---|
| V1 | 1.688 | 0.558 | 1.584 | −0.093 | −0.065 | 0.0217 | 0.0143 | 0.0056 |
| MT | 1.280 | 0.599 | 1.014 | −0.336 | −0.305 | 0.0295 | 0.0126 | 0.0071 |
| MST | 1.220 | 0.615 | 1.103 | −0.146 | −0.133 | 0.0300 | 0.0118 | 0.0076 |
| TEO | 1.409 | 0.695 | 1.153 | −0.289 | −0.259 | 0.0384 | 0.0127 | 0.0067 |
| LANSA prescribed (RHO) | 0.25 | 1.0 | 2.0 | +3.0 | | | | |

Observed in this animal: the profile is U-shaped (M lowest in all four areas), S ≥ D, |Δ| < δ_min = 1 in every area.
The prescribed monotone ramp (S 0.25 → D 2, eightfold) is not seen in monkey #1. Inferred: L4's PVALB excess sets M.

## Mapping into the circuit (provisional)

- Population proportions only: per-unit N_SST / N_PV = R via `rhouUX` (`caminos_stack.py`), e.g. V1:
  `e2_3unit_rhouS1p688_rhouM0p558_rhouD1p584`. PV + SOM per unit and E/I 8000:1500 unchanged.
- PV pool = PVALB (prereg A2); +CHC is a sensitivity variant.
- Kept as separate parameters, unchanged by the data: synaptic weights (giX, wRUX), connectivity (p, k, kd),
  kinetics (sst, sstd), drive.
- Not constrained by these data: E/I (the all-cells denominator includes non-neuronal cells; no glutamatergic
  export), VIP (not in the E-PV-SST model; fractions recorded only).
- PENDING areas enter only as per-candidate sensitivity, never pooled; 7A not used.
