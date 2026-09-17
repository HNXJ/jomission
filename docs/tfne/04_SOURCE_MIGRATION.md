# TFNE v1 to project source: conflicts and unspecified items

> **SUPERSEDED by `TFNE_V2_SPEC.md` (tfne/2, 2026-09-17).** Kept as the migration record.

**Date:** 2026-09-17
**Authoritative:** `TFNE_ALGEBRA_PROJECT_SOURCE.md` (supplied by Hamm, final).
**Superseded where they differ:** `00_TFNE_ALGEBRA_SPEC.md` (algebra version `tfne-algebra/1`, commit `cfd1e16`), and the fixtures and probes built on its spelling.

This record is additive. No v1 text was edited except the status banner of
`00_TFNE_ALGEBRA_SPEC.md`, which now points here. Nothing here decides anything: it
lists what the source changes, and what v1 carried that the source does not mention.

## 1. Changed meanings

| Item | v1 | Project source | Consequence |
|---|---|---|---|
| Core form | `x:A`, input application only | `x : A : y`, typed upstream and downstream boundaries | v1 has no downstream boundary and no `y` type; every v1 expression is a source expression with `y` undeclared |
| `H` | the canonical hierarchical cortical unit, `H = { P_{ℓ,c}, C_local, Θ_cell, Θ_syn, G }` | H-state tensor only; units are ordinary named objects (`V1 := CTX[v1]`) | the same letter now names a different thing. v1's unit content is not carried by any source name; `P`, `N`, `G`, `C` and `model` cover parts of it |
| `Q` | lateral composition primitive: symmetric, commutative, sorted operands, level-wise on chains, interchange law with `O` | absent; `X` is "nonserial, cross, or lateral" | v1 laterality rules (frontier union, `E_SHAPE`, interchange) have no source counterpart |
| `X` | architectural bypass relation, undirected until specialized | cross/lateral architectural composition | v1's "bypass" reading and `E_X_UNDIRECTED` are not in the source |
| Projection | `→`, `↛`, `↔` | `>`, `<`, `<>`, `≠>`, `≠<` | spelling change plus a new right-to-left primitive; v1 principle 1 (semantics canonical, spelling provisional) makes this a spelling migration only where the relations coincide |
| Grouping | `()`, required when operators mix; groups carry no identity | `{}`, a composite tensor that is a preserved semantic boundary; group sensitivity is explicit | in v1 grouping only disambiguates; in the source it changes the object |
| Replication | `H Q^n` couples every unordered pair; `H O^n` builds a chain | `A^n` creates indexed instances with no connectivity implied | v1 replication implied topology; the source forbids that reading |
| Declaration | units declared only by architecture statements; references never create units | `:=` defines any object or rule; `;` separates entries | the source has an explicit definition operator; v1's "references never create units" rule is not restated |
| Addressing | `unit.layer.class` | `.` is hierarchical addressing at any depth (`A.L4.E.soma`), never composition; also used in the compact dependency form `y = s.h.x` | source addressing is deeper and scale-free; the layer level is optional |
| Hierarchy | cortical layers assumed throughout | optional hierarchy, degenerate validity at cardinality one | a nonlaminar nucleus or a single cell is a valid model with no new grammar |

## 2. In the source, absent from v1

- Realization split `A → (s, h_0, I)` and the execution normal form `(h_t, x_t; s) ↦ (h_{t+1}, y_t)`.
- The realization index map `I` and its required inspection directions.
- Mutable-state decomposition `h = {h_dyn, h_H, h_plastic, h_history, h_rng}`, with plasticity on any declared mutable parameter.
- Static/mutable separation, continuation sufficiency, deterministic RNG semantics.
- `P`, `N`, `G`, `C` maps with proportion normalization and exact integer cardinality through a declared allocation map.
- Biological identity separated from the assigned dynamical `model`.
- Factoring and flattening stages, and the JaxFNE compilation boundary.

## 3. In v1, not mentioned by the source

These are not retired by the source; it simply does not speak about them. They need a
decision before any parser or compiler is built.

| v1 material | Where | Status now |
|---|---|---|
| Normalization pipeline (parse, expand, flatten, sort, match, validate, expand motifs, apply exceptions) | 00 §4.1 | the source requires "one deterministic expansion" but specifies no pipeline |
| Structural and projection normal forms, hashes `h_S`, `h_T`, `h_P`, `h_M` | 00 §4.2–4.4 | no model-identity layer in the source |
| Error catalogue (20 codes, including `E_PROJECTION_REDUNDANT`, `E_MECHANISM_NOT_PERMITTED`, `E_MATCH`, `E_RELATION_CONFLICT`) | 00 §5 | no validity layer in the source |
| Projection identity `(source population, target population, mechanism)`; parameters are not identity | 00 §3.9 | unstated; the source's rule independence is compatible with it |
| Exception algebra `G = (G0 \ E−) ∪ E+`, transactional statements | 00 §3.12–3.13 | the source has exclusion operators but no exception algebra |
| `allowed_mechanisms(route)` with `M_canonical ⊆ M_allowed` | 00 §3.8 | unstated |
| One architectural relation per unit pair | 00 §3.6 | unstated |
| Open biology D1, D2, D3, D4a–D4f | 00 §6.1 | unstated; D2 (lateral motif) is now attached to `X`, not `Q` |
| `H_SL` candidate and the SL0 / SL1 / SL2 gates, including "H is never tuned to make a proxy reproduce SL2" | 00 §6.2 | unstated; still the governing rule for any spectrolaminar work unless withdrawn |
| Reference spelling, grammar, map syntax, worked example | 00 §7–8 | superseded spelling; the worked example uses `Q` and `→` |
| Synthetic conformance fixtures and tests (P1–P15, E1–E7c) | `01_SYNTHETIC_CONFORMANCE.md` | written against v1 spelling and error codes |
| Structural probes | `scripts/tfne/probes/probe_verify.py`, `probe_density.py` | exercise v1 spelling; `probe_verify --check` pins 18 recorded values |
| JaxFNE findings A–D, normalization receipt, §5 spectrolaminar tooling F1–F5 | `02_JAXFNE_COMPATIBILITY.md` | unaffected: engine facts, not algebra |

## 4. Questions for the owner

> **Answered 2026-09-17:** `05_V2_MIGRATION_DECISIONS.md`. Q retired into `X[lateral]`;
> the cortical unit becomes the definition `CTX`; the validity layer is rebuilt after the
> corpus; SL0-SL2 move onto a candidate `CTX`; fixtures kept as `V1_FIXTURE`.

1. **`Q`.** Retired into `X`, or kept as a separate operator that the source omits? If retired, the v1 laterality rules (frontier union, level-wise expansion on chains, the `O`/`Q` interchange law) either move onto `X[k]` rules or are dropped.
2. **Unit content.** Does the canonical cortical unit survive as a named object (for example `CTX := [...]` with `P`, `C`, `G`, `model`), leaving D4a–D4f open against that object? The letter `H` is no longer available for it.
3. **Validity layer.** Do the v1 pipeline, normal forms, hashes and error codes carry over as the implementation of "deterministic normalization", or are they withdrawn pending a parser?
4. **Spectrolaminar gates.** Do SL0 / SL1 / SL2 and the no-tuning-against-a-proxy rule stay in force, and in which document?
5. **Fixtures.** Migrate `01_SYNTHETIC_CONFORMANCE.md` and the probes to the source spelling, or freeze them as a v1 record?

Until these are answered, `00_TFNE_ALGEBRA_SPEC.md` stays in the branch as the v1
record, and no parser or compiler work is justified.
