# A1 declarations: fill-in template

For Hamm. Each field is a decision only you can make; nothing here is inferred, and every value below is
`[MISSING: ...]` until you supply it. Fill the box, and I will put the text into `main.tex` and
`sections/backmatter.tex` unchanged, then re-run the build and the declaration count.

Two rules this template keeps:

- **No generated declarations.** No name, affiliation, ORCID, funder, grant number, conflict, contribution or AI-use
  statement is drafted here, not even a plausible placeholder. A wrong author list is worse than a missing one.
- **No access-rights claim.** The LANSA wording is conditional on your ruling. It does not say the data are
  redistributed, and it does not say they are available on request; both would be claims about someone else's
  terms.

---

## 1. Authors, affiliations, ORCID, corresponding author — `main.tex:30`

```latex
\author{[MISSING: authors, affiliations, ORCID and corresponding author]}
```

Please give, for each author in order:

| # | Full name as it should print | Affiliation(s), in the order they print | ORCID | Corresponding? |
|---|---|---|---|---|
| 1 | `MISSING` | `MISSING` | `MISSING` | `MISSING` |

Also needed: the corresponding author's email address, and whether you want affiliations numbered or written out.

## 2. Author contributions (CRediT) — `backmatter.tex:14`

```latex
[MISSING: author contributions (CRediT).]
```

Give the role list per author, e.g. `Conceptualization: A.B.; Methodology: A.B., C.D.; ...` — or state that you will
use a different scheme. `MISSING` until then.

## 3. Funding — `backmatter.tex:22`

```latex
[MISSING: funding sources and grant numbers.]
```

| Funder, as it must be named | Grant number | Authors supported |
|---|---|---|
| `MISSING` | `MISSING` | `MISSING` |

If there was no specific funding, say so explicitly ("This work received no specific grant...") — that is a fact I
will not assume.

## 4. Competing interests — `backmatter.tex:18`

```latex
[MISSING: competing-interest statement.]
```

State the declaration verbatim, or confirm you want the standard "The authors declare no competing interests."
**I will not insert that sentence on my own**; it is an assertion about your interests.

## 5. Acknowledgements — `backmatter.tex:12`

```latex
[MISSING: acknowledgements.]
```

Text as it should print, or an explicit "None."

## 6. Use of AI tools — `backmatter.tex:24`

```latex
[MISSING: the authors' statement on the use of AI tools.]
```

This is a factual record of what was used and for what. Only you can state it. Three decisions it needs:

1. Which tools, named as you want them named (the repository history records agent-assisted work; the journal's
   policy sets what must be disclosed).
2. Which tasks they were used for (drafting, code, analysis, figures, review).
3. Whether the statement asserts that authors reviewed and take responsibility for all output.

## 7. Data availability — `backmatter.tex:3–7`

Current text, with two gaps:

```latex
The laminar densities and LFP spectra are those of \citet{Mulvey2024}. [MISSING: how the LANSA files were obtained
and their access terms.] Derived data and run records are listed in Table~\ref{tab:records}; their SHA-256 hashes
are in record \rec{D6}. [MISSING: archive and DOI for the run records, or the owner's statement that they are
available on request.]
```

Needs from you:

1. **How the LANSA files were obtained** — the route and any agreement that governs them.
2. **Their access terms**, and specifically what may be said in print. Choose one and I will word it exactly:
   - (a) cite Mulvey 2024 and state that the derived arrays are available while the source data are not redistributed;
   - (b) state the source's own access route (repository, request to the original authors, or a stated licence);
   - (c) another wording you specify.
3. **The run-record archive**: Zenodo is the proposal on the table, not a decision. If you approve it, the wording
   becomes a DOI; if not, the statement must say where the records are. Nothing is deposited until you say so.

Proposed wording shape, for approval, with the DOI filled from the archive you authorize:

> The laminar densities and LFP spectra are those of \citet{Mulvey2024}, obtained `[MISSING: route]` under
> `[MISSING: terms]`. Derived data and the run records listed in Table S6 are archived at `[MISSING: repository]`
> under DOI `[MISSING: DOI]`. The LANSA source data are not redistributed here; they remain under the terms above.

**Not to be published before your ruling:** any sentence implying the LANSA files are redistributable.

## 8. Code availability — `backmatter.tex:8–10`

Current text ends:

```latex
[MISSING: archive and DOI for the jomission release used here.]
```

`refs.bib` currently carries `note = {Software repository; no archive DOI yet}` for `Jomission`, which is accurate
and stays until a deposit exists. If you approve a Zenodo release of the code at a frozen SHA, I will fill the DOI and
update that note; if not, it stays as is.

## 9. The archive itself (A2), for approval — not an action yet

Proposal as you stated it, restated so the decision is explicit:

| Item | Proposal | Needs from you |
|---|---|---|
| Code | Zenodo deposit at a frozen SHA of this repository | go/no-go, and whether `dev`'s final SHA or a `main` merge |
| Derived simulation records | Zenodo, same deposit or a second one | go/no-go; R1–R11 plus the `results/` trees the paper cites |
| D6 checksums | included in the deposit, re-verified from the downloaded archive | nothing; I will run the re-hash and report mismatches |
| LANSA source data | **not deposited, not redistributed** | your ruling on the wording in §7 |
| Identifier | DOI preferable for both deposits | your choice of host if not Zenodo |

The check I would run after a deposit: download it fresh, re-hash every file listed in D6, and report
`0 mismatches` or the exact mismatches. That receipt goes into the record that the DOI exists.

---

## What happens when this comes back filled

1. I insert your text verbatim into `main.tex` and `backmatter.tex`; no rewording.
2. `grep -c MISSING sections/*.tex main.tex` must be 0.
3. Rebuild: 35 pages (page count may change with a longer author block), 0 errors, 0 overfull, 0 float-too-large.
4. A1's acceptance is then author-approved text with no placeholders — which is your signature, not mine to claim.

## Deliberately not in this template

- Authorship or ordering, even as a suggestion.
- Any funder, grant, conflict, contribution or AI-use sentence, even in draft form.
- Any statement about LANSA redistribution.
- Phase B–D work. Those are new studies, not prerequisites for this preprint.
