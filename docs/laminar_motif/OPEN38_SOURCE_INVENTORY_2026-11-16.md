# OPEN-38 source inventory: macaque SST / PVALB / VIP by layer — 2026-11-16

Status: **inventory, not data.** No values imported; no number below is a model constraint until extracted from
the primary text or data with area, assay, denominator and uncertainty. Bibliographic details come from web
search on 2026-11-16 and are marked where the full text was not read. Owner order 2026-11-16: compare observed
distributions against the prescribed 0.25/1/2 ramp, without fitting anatomy to the spectral result.

## Candidate sources

| # | Source | Markers | Areas / layers | Assay, denominator | Status |
|---|---|---|---|---|---|
| 1 | Chen et al., *Cell* 2023, "Single-cell spatial transcriptome reveals cell-type organization in the macaque cortex", doi:10.1016/j.cell.2023.06.009; processed data https://macaque.digital-brain.cn/spatial-omics, raw CNGB CNP0002035 | SST, PVALB, VIP (and LAMP5, RELN) inhibitory supertypes | 143 regions, layer-resolved (figure: volume-normalized layer distribution of GABAergic types) | Stereo-seq + snRNA-seq cell typing; per-cell calls, so SST/PV from one section shares one denominator. *M. fascicularis*, 3 animals (press release; verify) | **Only candidate found with all three markers by layer and area.** Full text and portal not yet read; per-layer values, animal count and uncertainty UNKNOWN |
| 2 | LANSA preprint, bioRxiv 10.1101/2024.03.27.587084 (2024), "The laminar organization of cell types in macaque cortex and its relationship to neuronal oscillations" | PV, CB, CR, NRGN | 11 areas, 21 bins | IHC densities; already imported as `lansa_density_X.json` | In repo. **No SST, no VIP**; CB and CR are not substitutes (owner ruling 2026-11-06) |
| 3 | *J Neurosci* 1987; 7(4):1133, "Regional heterogeneity in the distribution of somatostatin-28- and somatostatin-28(1-12)-immunoreactive profiles in monkey neocortex" (PMC6568990) | prosomatostatin-derived peptides | V1, V2, AIT, area 24 and others; laminar distribution | IHC, *M. fascicularis*; denominator UNKNOWN | Authors not verified (PMC page blocked). Reported: V1 fewest SST profiles, V2 about 2x, AIT about 3x V1. Per-layer numbers not extracted |
| 4 | Kelly, García-Marín, Rudy, Hawken 2019, macaque V1 GABA/PV densities | GABA, PV | V1, layers 1-6 | IHC; reported GABA 11% of neurons, PV 52% of GABA | Venue not verified. No SST/VIP; supplies a GABA denominator for V1 |
| 5 | Kooijmans et al., mouse vs macaque V1 calcium-binding proteins (PMC8152890) | PV, CB, CR | V1 by layer | IHC, automated counts; CBPs about 95% of V1 inhibitory cells | Year to verify. No SST/VIP |
| 6 | Torres-Gomez et al., *Cereb Cortex* 2020; 30(8):4544-4562 | PV, CR (SST: UNKNOWN) | MT/MST vs LPFC, layer 2/3 | IHC | L2/3 only; PV falls and CR rises toward LPFC |

No quantitative macaque VIP laminar source other than #1 was found. Mouse VIP data are out of scope.

## Check on the in-repo CB/PV claim

`PLAN_LANSA_E2.md:30` states macaque CB/PV about 3.0 superficial and 0.45-0.6 deep. LANSA X module means
(modules as in `scripts/lansa_module_density.py`; CB/PV of mean relative density; 11 areas) give medians
S 1.58, M 0.54, D 0.86 (range S 0.95-3.28, D 0.10-1.21). X does not reproduce the claim as stated; its source
stays UNKNOWN. CB/PV is not SST/PV in any case.

## Next steps (OPEN-38)

1. Read Chen 2023 (main text, STAR Methods, supplementary tables) and the portal: per-layer SST and PVALB cell
   counts per area, animal count, section count, layer registration, uncertainty. Record exact files and sha256.
2. Declare the comparison before reading values: per area, observed SST/PVALB by layer (L2/3, L4, L5/6 mapped to
   S/M/D) against the prescribed 0.25/1/2 ramp. Estimand: direction of the S-to-D change and its ratio; unit of
   inference animal or area, not cell; no fit to Y.
3. Read #3 for an IHC cross-check of SST laminar shape in V1/V2 (assay independent of #1).
4. Flag VIP as unavailable unless #1 provides it.
