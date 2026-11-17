# OPEN-38 licensing assessment: may the derived composition table be committed? (2026-11-16)

Owner decision 2026-11-16 (conservative): **not committed.** The full table stays outside the repository; the repo
holds independently authored code, provenance, validation results and aggregate findings that do not reconstruct
the source data. A successful portal export is not redistribution permission. This is an engineering reading of the
licence text, not legal advice; the open questions go to the authors (draft request, unsent) or counsel.

## Sources and their terms

| Source | Terms | Status |
|---|---|---|
| BSDC dataset, DOI 10.12412/BSDC.1685411291.30001 | CC BY-NC-ND 4.0 (dataset page) | not downloaded (login) |
| Portal `macaque.digital-brain.cn/spatial-omics` (source of the 88 exports) | terms of use not checked; UNKNOWN whether the CC licence of the BSDC dataset extends to the portal | the actual source of our counts |
| Chen et al. 2023, *Cell* supplements (mmc1-7) | publisher terms of the article (not read for this note): UNKNOWN | local copies, hashes in `SOURCE_MANIFEST.md` |

## CC BY-NC-ND 4.0, relevant clauses (legalcode.en, read 2026-11-16; quotes are short excerpts)

- §1 "Adapted Material means material subject to Copyright and Similar Rights that is derived from or based upon"
  the Licensed Material and altered, arranged or modified in a way that needs the Licensor's permission.
- §2(a)(1): share the Licensed Material "in whole or in part, for NonCommercial purposes only"; produce Adapted
  Material, but not share it.
- §4 (Sui Generis Database Rights): extract, reuse, reproduce and share all or a substantial portion of database
  contents for NonCommercial purposes, "provided You do not Share Adapted Material"; a database of one's own that
  includes such contents is itself Adapted Material (its individual contents are not); sharing a substantial
  portion requires §3(a) attribution.
- §8(a): the licence does not "reduce, limit, restrict, or impose conditions on any use" that needs no permission.

## Reading

1. Our table (area × layer module × subclass counts, all-cells denominators, fractions) is a summation and
   rearrangement of per-cluster counts from the portal. Whether that is "Adapted Material" depends on whether the
   counts are protected at all (facts are not copyrightable in many jurisdictions; database rights exist in the
   EU and some others) and on whether the arrangement needs permission. Not decidable from the text alone.
2. If the BSDC licence governs the portal data: §4 permits non-commercial sharing of a substantial portion of the
   contents with attribution, but our compiled table could be a database "that includes" those contents and so
   itself Adapted Material, which ND forbids sharing. That is the material risk the owner named.
3. If the portal has separate or no terms: permission is UNKNOWN, which is no permission.
4. Aggregate findings (a few ratios per area in prose, as a paper cites a source's numbers) do not reconstruct
   the dataset; the four-area ratio table in `COMPOSITION_F001_2026-11-16.md` (12 ratios, 8 VIP fractions) is
   of that kind. **[owner]** confirm this boundary, or remove those numbers and keep only the qualitative finding.

## What the repository holds and does not hold

| Item | In repo | Why |
|---|---|---|
| `scripts/open38_laminar_composition.py`, tests | yes | own code |
| sha256 of every input and of the output table; receipt fields | yes (`COMPOSITION_F001…`, `SOURCE_MANIFEST.md`) | provenance; hashes do not reconstruct data |
| validation results (gate checks, cross-check against the density table) | yes | own analysis |
| per-area ratio summary, 4 areas | yes, pending the [owner] item above | aggregate finding |
| 88 portal exports, `composition_f001.csv`, receipt JSON | no (`D:/_sources_open38/chen2023_portal/`) | reconstructs the source data |
| model layouts carrying 3-decimal ratios (`rhouS1p688…`) | in the spectrum spec | 3 numbers per area, the same aggregate class as item 4 |

Attribution to carry wherever a number appears: Chen A. et al., *Cell* 186:3726-3743 (2023),
doi:10.1016/j.cell.2023.06.009; portal macaque.digital-brain.cn (sample `monkey_f001`, exported 2026-11-16);
BSDC dataset DOI above, CC BY-NC-ND 4.0.

## Route to permission

Add one line to the unsent author request (`AUTHOR_REQUEST_DRAFT_2026-11-16.md`): permission to deposit the
derived per-area, per-layer subclass count table (with attribution) in a non-commercial research repository.
Sending stays the owner's action.
