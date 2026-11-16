# OPEN-38 source manifest

Source: Chen et al., "Single-cell spatial transcriptome reveals cell-type organization in the macaque cortex",
*Cell* 186(17):3726-3743 (2023), doi:10.1016/j.cell.2023.06.009. Species *Macaca fascicularis* (per press
release; confirm from STAR Methods). Assay: Stereo-seq spatial transcriptomics with snRNA-seq cell typing
(transcript-based abundance, not protein-positive density).

Downloaded 2026-11-16 from the Elsevier supplement server (owner-approved, Vanderbilt network), URL pattern
`https://ars.els-cdn.com/content/image/1-s2.0-S0092867423006797-mmcN.xlsx`. Local copies (outside the repo):
`D:/_sources_open38/chen2023/`. Not committed: they are inputs, recoverable from the URLs and verified by hash.

| File | Table | sha256 | Bytes |
|---|---|---|---|
| mmc1.xlsx | S1 Full names and cortical levels of cortical regions | 0e49911de60b2d4aa6b4695c69d3d75e36ae82c9b4d746deff17fbc52b50ec9a | 23010 |
| mmc2.xlsx | S2 Full names and marker genes of cell types (taxonomy) | 2a6a504d110d7be0a504ebf5afa9cb5fa79bc44d2d78cbcbaaf31c56da5ab416 | 7116672 |
| mmc3.xlsx | S3 Metadata for Stereo-seq sections | 9312684d3178b88b23547635b06e0075f117910dec6566259d987252b19396ce | 14841 |
| mmc4.xlsx | S4 Image training and validation | c0b2b1a37db6ea8864a110de58122dc835dd00322e435a5eb78d8a69f1110f0d | 16404 |
| mmc5.xlsx | S5 Density of cell types in each cortical region | 7903754359423734665db961af564b747c103afb6bb75ffd6759a79cd4102fa3 | 1230131 |
| mmc6.xlsx | S6 Neighborhood glutamatergic neurons | 3e70b59aa293a8f5be059606f09012703dc8edc78eb29503e142de51add870d7 | 1273029 |
| mmc7.xlsx | S7 Cross-species pair-wise comparison | 7a246bf318320bdc69e82bc90263df9d3fc5310f91e27415b20c6a266e80e687 | 396011 |

Article PDF: not obtained (ScienceDirect and cell.com return a bot check to scripted requests).

## Metadata read (no SST/PVALB value read)

- S3: 161 Stereo-seq sections, Section ID, Monkey, EBZ (mm), Number of cells. Sections per monkey:
  #1 119, #2 19, #3 23. Monkey ID exists per section.
- S5: columns Region, Class, Cell type, Cell_density; sheets glutamatergic, GABAergic, non-neuronal.
  **No layer column and no monkey column**: densities are per region, pooled over layers and animals.
- S1: region hierarchy (Level 1-3) for 143 regions; used for the §4 area-label table.
- S2: class, subclass, full name and abbreviation of 264 types; used for the A2 type assignment.

## Feasibility verdict

The supplement cannot carry the primary test: S5 has neither layer nor animal resolution. The preregistered
contrast needs per-cell (or per-section × layer) records with region, layer, section and monkey labels: the
processed Stereo-seq data on the portal or the raw data at CNGB CNP0002035. Animal coverage is unbalanced
(#1 has 119 of 161 sections); areas sampled in one monkey only will be INCONCLUSIVE (n = 1), and if no area has
≥ 2 animals passing the floor the outcome is NOT TESTABLE (prereg A2/A4).
