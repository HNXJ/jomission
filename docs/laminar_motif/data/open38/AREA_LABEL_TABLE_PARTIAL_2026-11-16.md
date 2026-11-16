# OPEN-38 area-label table — PARTIAL (4 of 11 areas), 2026-11-16

Status: **partial mapping, committed separately by owner order 2026-11-16.** The complete 11-area table stays
uncommitted until its acceptance criteria pass. Sampling sites were not assigned from anatomical plausibility;
an owner statement is not evidence of LANSA's sampling locations.

Sources: portal Cell Summation CSV region labels (probe: sample `monkey_f001`, slice 64/119, default
selection; labels only) and Chen 2023 Table S1 (`mmc1.xlsx`, sha256 0e49911d...). Rule: prereg A1.

| LANSA | Status | Portal label(s) | S1 row(s) | Portal layers |
|---|---|---|---|---|
| V1 | PASS (exact) | V1 | 1 | l1-l6 |
| MT | PASS (exact) | MT | 36 | l1-l6 |
| MST | PASS (exact) | MST | 35 | l1-l6 |
| TEO | PASS (exact) | TEO | 37 | l1-l6 |
| V3 | PENDING | candidates V3d, V3v (parent V3 has no own rows) | 3, 4 | l1-l6 |
| V4 | PENDING | candidates V4 (S1: "visual area 4 (dorsal part)"), V4v | 6, 7 | l1-l6 |
| LIP | PENDING | candidates LIPd, LIPv | 14, 15 | l1-l6 |
| PMD | PENDING | candidates F2, F7 (parent F2/F7) | 103, 104 | l1, l2, l3, l5, l6 (agranular: M structurally missing) |
| 8A | PENDING | candidates 8Ad, 8Av (parent 8A/FEF) | 120, 121 | l1-l6 |
| PFC | PENDING | no "PFC" label; candidates include 46d, 46v | 115, 116 | l1-l6 |
| 7A | CONFLICT | portal: 7A = {Opt, PG}, 7B = {PF, PFG}; S1 Level 1: 7A = {PF, PFG}, 7B = {PG, Opt} | 10-13 | l1-l6 |

- PENDING and CONFLICT areas are excluded from every quantitative comparison (never zero).
- 7A stays unresolved until source evidence settles which subareas form 7A in each source.
- Recovery: sampling sites from the LANSA manuscript (bioRxiv 10.1101/2024.03.27.587084, Methods) with page
  references; a site that does not identify one atlas subdivision stays MISSING.

## Portal samples vs animals (decision-gate evidence)

All 14 portal samples (`monkey_f001`, `monkey_f002-test` ... `monkey_f0014-test`) list the same 119 section
IDs, T25-T155, which are monkey #1's sections in Table S3. Table S3's monkey #2 (T573-T592, 19 sections) and
monkey #3 (T825-T875, 23 sections) do not appear on the portal. The `-test` samples are copies of monkey #1.

Gate result: the portal export carries **one animal**. Under prereg A2/A4 no area can have ≥ 2 animals, so
the confirmatory test is **NOT TESTABLE from the portal** (a statement about this source, not a finding about
SST/PVALB). Portal counts could support only an exploratory n = 1 description. Animal-linked data for monkeys #2
and #3 exist in the BSDC export (`ST.CellID.CellType.3monkeys.all.tsv`, login required) or from the authors.
