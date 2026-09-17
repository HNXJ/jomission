# TFNE v2 migration decisions

**Date:** 2026-09-17
**Authority:** owner (Hamm), answering the five questions in `04_SOURCE_MIGRATION.md`.
**Authoritative algebra:** `TFNE_ALGEBRA_PROJECT_SOURCE.md`.
**Status of this branch:** design lineage only. No parser, no compiler, no scientific authority.

## Pipeline endpoint

$$
x:\mathcal A:y \;\xrightarrow{\text{normalize}}\; NF(\mathcal A)
\;\xrightarrow[\rho]{R,\ \text{realize}}\; (s,h_0,I)
\;\xrightarrow{x_t}\; (h_t,y_t)
$$

with $R$ the realization rules, $\rho$ the declared allocation, and $I$ the index/address map.

## Decisions

| # | Question | Decision |
|---|---|---|
| 1 | `Q` | **Retired.** Architectural primitives are `O` (ordered) and `X` (non-ordered/cross). Lateral is the specialization `A X[lateral] B`. The v1 `O`/`Q` interchange law is not migrated; a recurring lateral topology is defined inside the relevant `X[k]` rule, never as universal grammar. |
| 2 | Canonical cortical unit | **`CTX`, a definition, not a reserved name.** `CTX[k] := [...]` defines a cortical-unit family; `V1 := CTX[v1]`, `V4 := CTX[v4]` are instances. D4a–D4f become properties of a candidate `CTX`. `H` stays exclusively the H-state tensor. `LGN`, `SC`, spinal segments, ganglia and single cells likewise require definitions, not primitives. Cortex is a TFNE definition, not TFNE grammar. |
| 3 | Validity layer | **Retained conceptually, migrated after the adversarial corpus.** Stage order `parse → type → normalize → validate → realize`. The minimum error vocabulary is derived from observed v2 failure classes; the 20 v1 codes are historical in the migration ledger, not automatically permanent. Identities such as `h_T = hash(NF(A))` are kept in principle and redesigned after the v2 normal form is fixed; v1 hash semantics do not constrain v2. |
| 4 | SL0 / SL1 / SL2 | **Retained, outside the fundamental algebra.** They are qualification requirements on a candidate `CTX`: `CTX_candidate --SL0--> native generators`; `F_field --SL1--> qualified physical forward model`; `SL0 + SL1 --SL2--> spectrolaminar phenotype`. The prohibition stands: `CTX` is never tuned to make a phenomenological proxy reproduce SL2. |
| 5 | Fixtures and probes | **Preserved historically, not mechanically translated.** `01_SYNTHETIC_CONFORMANCE.md` is marked `V1_FIXTURE`, superseded for v2 semantics. The 18 pinned JaxFNE values stay valid compatibility receipts because `scripts/tfne/probes/probe_verify.py` and `probe_density.py` execute the engine and never parse TFNE notation. Build the v2 adversarial corpus first; migrate fixture concepts afterwards. |

## Refinement: the index map `I`

`I` is a **derived realization object**, not a fundamental user-language symbol.

$$
I:\ \text{TFNE address} \leftrightarrow \text{realized index/range},\qquad I[\mathrm{V1.L4.E}]=[i_0,i_1)
$$

- Deterministic for fixed $NF(\mathcal A)$, $R$ and $\rho$: the same three inputs always give the same ranges.
- Load-bearing for observability, intervention targeting, checkpoint interpretation, and comparing two realizations without losing biological identity in flattening.
- Not a user-facing primitive: nothing in an expression names `I`; it is produced by realization.

## Sealing criterion

> New biology requires new definitions, not new grammar.

The universality test: a single HH cell, a nonlaminar nucleus, a six-layer cortical `CTX`,
a replicated spinal chain, and a heterogeneous multi-area nervous system, all expressible
without adding primitives. Worked in `06_V2_ADVERSARIAL_CORPUS.md`.

## Next action (this lineage only)

Corpus, invalid probes, expected structural and type properties, manual derivations. No
parser implementation. One replacement permanent spec, and retirement of the v1 documents,
only after the corpus review.
