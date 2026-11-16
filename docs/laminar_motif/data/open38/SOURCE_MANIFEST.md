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

## Official processed export (portal "Data download", 2026-11-16)

The portal (`macaque.digital-brain.cn/spatial-omics`) menu item "Data download" opens the Brain Science Data
Center (CAS) dataset "Cell type annotation dataset for the spatial transcriptome of the macaque cortex",
Yidi Sun (2023), DOI 10.12412/BSDC.1685411291.30001, CSTR 33145.11.BSDC.1685409749.1663093488869060610,
page `https://www.braindatacenter.cn/datacenter/web/#/dataSet/details?id=1663381185152036865`.
License **CC BY-NC-ND 4.0**; 174 files, 564.16 GB; contact ydsun@ion.ac.cn.

Files listed on the page (first page of the list; sizes as shown):

| File | Size | Likely content (from the name; unverified) |
|---|---|---|
| /README-update20251117.md | 4 KB | schema |
| /ST_result/ST.CellID.CellType.3monkeys.all.tsv | 1.58 GB | per-cell type calls, 3 monkeys |
| /ST_result/ST.CellID.quality.3monkeys.all.tsv | 1.63 GB | per-cell quality |
| /ST_result/SliceID_EBZ_info.xlsx | 13 KB | section-to-monkey map (cf. S3) |
| /ST_result/cytpe_region_ration.xlsx | 425 KB | type-by-region ratios |
| /ST_result/regions-macaque1.csv, -macaque2.csv, -macaque3.csv (+ a macaque2 contour2 variant) | 31-538 KB | region (possibly layer) contours per monkey |
| /ST_result/20230503-macaque1-sct/total_gene_T*.type...contour2.round.txt | 1.5-3.9 GB each | per-section gene counts with type and contour |

**Access restriction:** every Download button, including the 4 KB README, redirects to the BSDC login page
(`braindatacenter.cn/iam/admin/#/login`). No file has been downloaded; whether a cortical-layer field exists is
UNKNOWN until the README is read. The ND term restricts sharing adapted material: extracted per-layer count
tables may not be redistributable in this repository without the authors' permission (owner to decide).

## Public route without login: portal "Cell Summation" CSV (2026-11-16)

Portal Help ("Show Gene Expression in Certain Layer of Brain Area"; Cell Summation panel: "The Num. of cells of
all selected cell types in differ regions", with "Download as CSV") documents a per-slice export. Probe on the
default view (Sample `monkey_f001`, Project Neocortex, slice 64/119, default cell-type selection), via the UI's
own Download as CSV, **no login**:

- Columns: `region`, `Cell Count`; 1002 rows.
- Region labels are hierarchical and **layer-resolved**: lobe (`occipital`, `parietal`, ...), area (`V1`,
  `V2`, `V3A`, `V4v`, `7op`, `AIP`, `DP`, ...), and area-layer rows `<area>-l1` ... `<area>-l6`.
- Only labels were read; no count was read. The probe file stays in scratch, not in the repo.

Fields of the minimal table (animal, area, layer, SST_count, PVALB_count, sampled cells, assay):

| Field | Source | Status |
|---|---|---|
| animal | Sample selector (`monkey_f001`; S3 lists monkeys #1-#3) | other sample names not yet read (selector not opened by the probe) |
| section | slice selector (n/119 for monkey #1; S3 Section IDs) | available |
| area, layer | CSV `region` (`<area>-l1..l6`) | available; L2 and L3 separate (A1 merge rule: sum); **V1 L4 sublaminae not separated** (A1 V1 sublaminar report not possible) |
| SST, PVALB count | CSV `Cell Count` after selecting the SST or PVALB subclass in "Select cell type" | filter behaviour not yet confirmed |
| sampled cells | same export with all types selected | available |
| assay | Stereo-seq cell typing (Spatial-ID) | transcriptomic abundance |

Cost: one CSV per animal × slice × type (161 slices × 2 types = 322 exports), through the supported UI.
Not a confirmatory dataset until the area-label table (§4/A1) is committed and the filter is confirmed.

## Acquisition status (owner ruling 2026-11-16, Option 3)

| Route | Priority | Owner action | State |
|---|---|---|---|
| Author request (`AUTHOR_REQUEST_DRAFT_2026-11-16.md`) | 1 | send (unsent until explicitly authorized) | waiting on owner |
| BSDC per-cell TSV + README | 2 | register and download into `D:/_sources_open38/chen2023_bsdc/` | waiting on owner |

On arrival, before any count is read: verify animal, section, region, layer and cell-type identifiers are
recoverable; check section-to-animal correspondence against Table S3; layer assignment method; 7A nomenclature;
and, if both routes deliver, whether they are the same observations (reconcile provenance and duplicates
before combining). Confirmatory analysis resumes only with ≥ 2 independent animals for a comparison; sections
and portal `-test` copies are never animals. No monkey #1 exploratory pilot meanwhile.

## Feasibility verdict

The supplement cannot carry the primary test: S5 has neither layer nor animal resolution. The preregistered
contrast needs per-cell (or per-section × layer) records with region, layer, section and monkey labels: the
processed Stereo-seq data on the portal or the raw data at CNGB CNP0002035. Animal coverage is unbalanced
(#1 has 119 of 161 sections); areas sampled in one monkey only will be INCONCLUSIVE (n = 1), and if no area has
≥ 2 animals passing the floor the outcome is NOT TESTABLE (prereg A2/A4).
