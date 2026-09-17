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

## Normalization decisions (second pass, 2026-09-17)

Answering the seven findings of `06_V2_ADVERSARIAL_CORPUS.md`. No new architectural
primitive: every item is specification of normalization, not of nervous systems.

| # | Finding | Decision |
|---|---|---|
| R1 | replica composition (F1) | Replication stays connectivity-free: `A^n = {A_1, …, A_n}`. Composition over a replicated family is the derived application form `O[k](A^n)`, meaning `A_1 O[k] A_2 O[k] … O[k] A_n`. It is an application of an existing rule, not a new relation. `A^n` alone stays disconnected. |
| R2 | instance addressing (F2) | Numeric path components: `SEG.3`, and `SEG.3.C.E` is ordinary addressing. `[]` is not overloaded; `.` stays address-only. |
| R3 | composite frontiers (F3) | Frontiers are never inferred from grouping. Every composite has derived interfaces `in(A)`, `out(A)`, optionally overridden in its definition. For `{V2 X[lateral] V3}`, `in({·}) = in(V2) ∪ in(V3)`. A rule applies from `out(lhs)` to the operand's declared or derived `in`. Not uniquely derivable → `E_FRONTIER_UNRESOLVED`; no guessing. |
| R4 | rule operands (F4) | Inside a rule body, the metavariables `$L` and `$R` bind the left and right operands of that invocation: `O[ff] := [ $L.out >[AMPA] $R.in ]`. For `X[k]` the same metavariables carry syntactic identity only, with no implied order. Rules therefore generate explicit `>`, `<`, `<>` without naming objects. |
| R5 | associativity (F5) | `A O[k] B O[k] C ≡ (A O[k] B) O[k] C` for an unbraced homogeneous `O[k]` chain; its normal form is the ordered sequence `[A, B, C]` with `k` applied between adjacent elements. Braces create an object, so `{A O B} O C` is not generally equivalent. `X` is not globally associative: `A X[k] B X[k] C` must be grouped unless rule `k` declares an associative composition policy. |
| R6 | index traversal (F6) | Canonical traversal, not declaration order (source reordering must not change realization identity): structural path, replicated numeric index, canonical child key, population/cell identity, local realization index. Addresses sort lexicographically under the declared canonical key order. Where biological order matters it belongs in the object definition and therefore enters `NF`. |
| R7 | exclusions (F7) | Canonical expansion yields `G_0`; every exclusion resolves to projection identities `E_-` with `E_- ⊆ G_0`, and `G = G_0 \ E_-`. An exclusion matching nothing is `E_EXCLUSION_UNKNOWN`, so a stale exclusion cannot survive silently. Explicit additions form `E_+` with `E_+ ∩ G_0 = ∅`. |

**Allocation correction.** Non-integer products are not themselves invalid: `N = 101` with
`P = (0.8, 0.2)` is realized by the declared policy `R`, for example `(80.8, 20.2) → (81, 20)`
with `Σ_c N_c = 101`. The failure is *no realization/allocation policy available*. `ρ` is the
realization RNG identity, not the allocation rule; a deterministic `R` needs no RNG.

**Pipeline.**

```text
source → parse → type/address resolution → replication expansion → structural normalization
       → frontier resolution → O/X-rule expansion → projection generation G_0
       → exception resolution → NF(A) --R, ρ--> (s, h_0, I)
```

Each stage fails locally.

**Stopping test.** Can the adversarial corpus break deterministic normalization without
requiring new biology-specific grammar? If not, TFNE v2 is ready for one permanent spec.

## Sealing criterion

> New biology requires new definitions, not new grammar.

The universality test: a single HH cell, a nonlaminar nucleus, a six-layer cortical `CTX`,
a replicated spinal chain, and a heterogeneous multi-area nervous system, all expressible
without adding primitives. Worked in `06_V2_ADVERSARIAL_CORPUS.md`.

## Next action (this lineage only)

Corpus, invalid probes, expected structural and type properties, manual derivations. No
parser implementation. One replacement permanent spec, and retirement of the v1 documents,
only after the corpus review.
