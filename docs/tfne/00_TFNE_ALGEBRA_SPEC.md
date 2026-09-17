# TFNE Algebra v1: semantics

```
TFNE status: DESIGN LANGUAGE
Parser: NOT IMPLEMENTED
Biological H: OPEN
Canonical O motif: OPEN
Canonical Q motif: OPEN
Canonical X catalogue: OPEN
Structural conformance: SPECIFIED
TFNE -> JaxFNE structural compilation: DESIGN
TFNE -> JDNA parameter compilation: UNQUALIFIED
Scientific execution authority: NONE
```

**Owner:** Jomission (Hamm)
**Algebra version:** `tfne-algebra/1`
**Date:** 2026-09-16
**Lineage:** independent language/tooling lineage (branch `tfne-algebra`); not part of the V2 scientific lineage.
**Scope:** language only, no simulator content. Test fixtures: `01_SYNTHETIC_CONFORMANCE.md`. Realization in JaxFNE: `02_JAXFNE_COMPATIBILITY.md`.

## 0. Status tags

| Tag | Meaning |
|---|---|
| **SETTLED** | Decided by Hamm. |
| **PROVISIONAL SPELLING** | Reference text syntax. Any future spelling must normalize to the same TFNE graph. |
| **OPEN-BIOLOGY** | A biological specification. Must not be decided just to unblock implementation. |
| **OPEN-LANGUAGE** | Undefined until a real expression needs it. |
| **PROPOSED** | Suggested for implementability; awaiting confirmation. |

## 1. Principles [SETTLED]

1. **Semantics are canonical; textual spelling is provisional.** The reference spelling (section 7) may change, but every spelling of an expression must normalize to the same TFNE graph.
2. **Architecture identity exists independently of any realization.** A TFNE graph and its hashes are defined without reference to any simulator. Compiling into JaxFNE is a separate step that keeps configured, realized and effective values distinct. No simulator default, normalization rule or example config defines TFNE biology.
3. **Scope is architecture and composition.** Neuron dynamics, geometry, HDP/plasticity, stimulation protocol, RNG, duration and solver enter only as explicit parameters `[θ]`.
4. **Growth is driven by real expressions.** The primitive set is frozen; add a primitive only when a recurring architectural concept cannot be expressed by composition.
5. **Layer separation:**

| Layer | Owns |
|---|---|
| `H` | unit semantics |
| `O`, `Q`, `X` | composition semantics |
| `→`, `↛` | projection semantics |
| JDNA / JaxFNE | realization |

## 2. Primitives [SETTLED]

| Primitive | Name | Meaning |
|---|---|---|
| `H` | unit | One canonical hierarchical cortical unit; named instances (`V1`, `V4_L`, …) specialize it. |
| `:` | input application | `x:A` applies input `x` to the input frontier of `A`. |
| `O` | hierarchical composition | Serial composition with the canonical hierarchical motif. |
| `Q` | lateral composition | Parallel composition at one level; symmetric. |
| `X` | bypass relation | Architectural noncanonical cross/bypass relation. |
| `^n` | indexed replication | `H Q^n`: n parallel copies. `H O^n`: n serial copies. |
| `.` | addressing | Selection inside a unit (`V1.L23.E`); composes nothing. |
| `[θ]` | parameterization | On units, populations, operators, replication, projections, exclusions. |
| `→` | explicit projection | A directed projection at any addressable level. |
| `↛` | exclusion | Removes projections from the canonical expansion. |
| `()` | grouping | Required whenever O, Q and X mix. |

## 3. Semantics

### 3.1 Units and identity [SETTLED]

- A unit is one H instance, identified by its identifier. The base is the part before `_`; the identity label is the part after. For `V1_L`, the base is `V1` and the label is `L`.
- Identity is by label. Operand order carries no topology meaning, except that the left operand of O is the lower unit.
- **Units are declared only by architecture statements.** A bare `V1` is a valid statement. References never create units: projections, exclusions and addresses must name declared units, so `V1; PFC; V1 -> PFCl` raises `E_ADDRESS_UNKNOWN`.
- **[PROPOSED]** Every realized unit id is unique. The compiler never renames; a collision raises `E_ID_COLLISION`.

### 3.2 Replication [SETTLED]

n is total multiplicity:

- `H Q^n` creates parallel units `H_1 … H_n`.
- `H O^n` creates the chain `H_1 O H_2 O … O H_n`.

**[OPEN-LANGUAGE]** An exponent on a composite, such as `(A O B) Q^2`, raises `E_REP_COMPOSITE`.

### 3.3 Frontiers and O [SETTLED]

```
in(H) = out(H) = {H}
in(A O B) = in(A)          out(A O B) = out(B)
in(A Q B) = in(A) ∪ in(B)  out(A Q B) = out(A) ∪ out(B)
```

`A O B` connects `out(A)` to `in(B)` under the canonical O motif, matched by identity label, never by position:

- Two singleton frontiers match.
- Otherwise each unit matches exactly one unit with the same label on the other side.
- If labels cannot be matched uniquely, the expression is invalid (`E_MATCH`) unless `O[map=...]` states the mapping.
- Crossed links exist only when added with X. All-to-all is never inferred.

Example: `(V1_L Q V1_R) O (V4_R Q V4_L)` gives `V1_L O V4_L` and `V1_R O V4_R`.

O is associative. Consequence: `(H1 Q^n1) O (H2 Q^n2)` is valid only if `n1 = n2` or a map is given.

### 3.4 Q [SETTLED]

- Symmetric lateral/homologous coupling between corresponding populations under the canonical lateral motif. That motif is OPEN-BIOLOGY (D2).
- Associative and commutative. Operands are sorted by canonical label before matching.
- `H Q^n` couples every unordered pair `{H_i, H_j}` with `i < j`, making it a parallel bank. Sparse coupling needs `Q^n[topology=...]`.
- Level-wise between chains: `(A O B) Q (C O D)` expands to both chains plus `A Q C` and `B Q D`. Depths must be equal and labels compatible; otherwise `E_SHAPE` unless `Q[map=...]`.
- Interchange law, when shapes and labels are compatible: `(A Q B) O (C Q D) ≡ (A O C) Q (B O D)`. O spans depth and Q spans parallel organization.

### 3.5 X [SETTLED]

- An architectural noncanonical relation, optionally with a named bypass motif (catalogue: D3).
- Bare `A X B` states existence only and is not executable (`E_X_UNDIRECTED` at projection level).
- `A X[->] B` and `A X[<->] B` specialize it. X never inherits O's motif.
- **[OPEN-LANGUAGE]** X between composites raises `E_X_COMPOSITE`.

### 3.6 One architectural relation per unit pair [SETTLED]

```
R(A, B) ∈ {O, Q, X, ∅}
```

If an architecture needs two biologically distinct relation families between the same units, one is the canonical relation and the others are explicit projections or motifs. A second relation raises `E_RELATION_CONFLICT`. This prevents ambiguous double expansion.

Explicit projections `→` do not count toward R. They are still validated against G0 for redundancy and conflict (3.12).

### 3.7 Addressing [SETTLED]

`unit.layer.class`, e.g. `V1.L5.PV`. Addresses select; they never compose. Layer names and aliases (such as `L23`) come from the unit definition (D4a). An alias expands to the Cartesian product of its members. An unknown address raises `E_ADDRESS_UNKNOWN`.

### 3.8 Explicit projections `→` [SETTLED]

- **Valid at any addressable level.**
  - At unit level, `V1 → PFC` is an explicit directed projection specification, not an architectural relation. It is legal, and becomes executable only through an explicit or default projection motif (D3); otherwise `E_MOTIF_UNDEFINED`.
  - At population level, `V1.L23.E → PFC.L4.PV` is fully addressed.
- X is an architectural bypass relation; `→` is an explicit directed projection. `A X[->] B` is not defined as identical to `A → B`.
- **Exactly one mechanism.** Every resolved explicit projection obtains exactly one mechanism, either from the arrow (`→[mech=AMPA]`) or from the motif or operator that generates it.

| Case | Result |
|---|---|
| no mechanism from the arrow or a generating motif | `E_MECHANISM_UNRESOLVED` |
| arrow mechanism differs from the generating motif's | `E_MECHANISM_CONFLICT` |

| arrow mechanism equals the generating motif's | `E_PROJECTION_REDUNDANT` |

- **One spelling per normal form.** Restating a mechanism the generating motif already supplies adds a second spelling of the same graph and is invalid. This follows the same principle as `E+ ∩ G0 = ∅` (3.12). An explicit fully addressed projection that repeats an identity already in G0 raises `E_ADD_IN_G0`.
- **Same endpoints, different mechanism.** A fully addressed projection whose endpoints match a G0 projection but whose mechanism differs has a distinct identity (3.9). It is valid unless the generating motif restricts that route to its own mechanisms (`E_MECHANISM_NOT_PERMITTED`; registry field `extra_mechanisms`, PROPOSED).

### 3.9 Projection identity [SETTLED]

```
ID_P = (source population, target population, mechanism)
```

- Two projections between the same populations with different mechanisms are distinct projections.
- Weights, delays, density, time constants and plasticity are **parameters** of a projection, not part of its identity.

### 3.10 Parameters `[θ]` [SETTLED placement; keys PROPOSED]

- **Placement:** a parameter binds to the object or operator it follows.

| Spelling | Parameterizes |
|---|---|
| `H[θ]`, `V1[N=2000]` | unit |
| `PFC.L4.PV[θ]` | population |
| `A O[θ] B`, `Q[θ]`, `X[θ]` | operator / its motif |
| `H Q^n[topology=...]` | replication |
| `A →[θ] B` | projection |
| `A ↛[mech=NMDA] B` | exclusion selector |

  `PFC.L4.PV[θ]` never parameterizes a projection.
- **Density** is a motif parameter, either `O[p=ρ]` or inside the registered motif. TFNE does not inherit any simulator's density limitation.
- **Synaptic time constant and conduction delay are distinct keys.** `tau_syn_ms` is the synaptic time constant; `delay_ms` is the axonal conduction delay. A compiler never maps one onto the other, and never reads an engine field as a delay because of its name (JaxFNE `dT_ms` is realized as a synaptic time constant; `02_JAXFNE_COMPATIBILITY.md` finding B).
- **[PROPOSED]** Keys and types are declared per target in the motif registry; an unknown key raises `E_PARAM_UNKNOWN`. Values are coerced to their declared type and defaults are resolved **before** hashing, so `N=1000` equals `N=1000.0`, and `O[p=1.0]` equals `O` when the registry default is 1.0.

### 3.11 Input [SETTLED]

- `x:(A Q B)` ≡ `(x:A) Q (x:B)`.
- `(x:A) Q B` gives `x` to A only.
- `x:(A O B)` ≡ `(x:A) O B`.
- Receiving populations come from the unit's canonical input frontier (D4c).
- The input prefix extends to the end of its enclosing group and is not a composition operator.

### 3.12 Exceptions [SETTLED]

```
G0 = expand(TFNE)
G  = (G0 \ E−) ∪ E+
```

E− and E+ are unordered sets.

- An exclusion selects by address and optional mechanism:
  - `V4 ↛ V1` removes every generated V4→V1 projection.
  - `V4.L5.E ↛ V1.L1.E` removes all mechanisms between those populations.
  - `V4.L5.E ↛[mech=NMDA] V1.L1.E` removes only the NMDA projection.
- Every exception resolves to explicit projection identities, then:

| Constraint | Error |
|---|---|
| `E− ∩ E+ = ∅` | `E_EXC_CONFLICT` |
| `E− ⊆ G0`, and each exclusion selects at least one projection | `E_EXCL_NOT_IN_G0` |
| `E+ ∩ G0 = ∅` | `E_ADD_IN_G0` |

- The first constraint follows from the other two. It is kept for its clearer message, and both errors are reported when a projection is in both sets.
- Because identity includes mechanism, adding NMDA between populations that already share an AMPA projection is a valid addition, and removing one receptor leaves the others.
- `A O B; B ↛ A` is valid when the intent is exactly "canonical O minus its feedback". There is no `O_FF`.

### 3.13 Transactional statements [SETTLED]

A statement is validated over all of its resolved projection identities. If any one violates a constraint, the statement contributes nothing to G: no partial expansion. Example: `V1.L23.E ->[mech=AMPA] V4.L4` under `V1 O V4` resolves to E targets already in G0 and PV targets that are new; the whole statement raises `E_ADD_IN_G0` and the PV projections are not added.

## 4. Normal forms and hashes

### 4.1 Pipeline [SETTLED]

1. parse
2. expand indexed replication
3. flatten associative structures
4. sort Q operands by canonical label
5. resolve O/Q structural matching
6. validate declarations, mappings and shapes
7. expand O, Q, X and projection motifs; resolve mechanisms
8. expand exceptions to explicit projection identities
9. validate E−, E+ per statement (3.13)
10. apply exceptions
11. produce G_TFNE

### 4.2 Structural normal form (steps 1–6) [PROPOSED]

A motif-free unit-relation graph:

- units (base, label, definition id, θ)
- O relations (lower → higher)
- Q unordered pairs
- X relations with direction or `unspecified`
- explicit projections and exclusions as written, with canonical addresses
- input bindings to units

It is computable with no motif registry. Hash: `h_S`.

### 4.3 Projection normal form (steps 1–11) [SETTLED pipeline]

The explicit projection set after exceptions: each projection's identity (3.9) with its parameters, plus input bindings. It requires a motif registry.

- With the canonical registry it is the biological TFNE graph, which is **blocked by D1–D4**.
- With a registry marked synthetic it is a conformance graph, available now (`01_SYNTHETIC_CONFORMANCE.md`).

### 4.4 Hashes [SETTLED]

| Hash | Covers |
|---|---|
| `h_T = hash(NF_topology)` | unit set, architectural relations, and the set of projection identities after exceptions |
| `h_P = hash(NF_parameters)` | every parameter keyed by unit, population or projection identity: `N`, weights, `delay_ms`, density, `tau_syn_ms`, plasticity |
| `h_M = hash(h_T, h_P, motif-registry version, algebra version)` | the full model identity |
| `h_S` | structural NF (4.2) |

How to read changes:

| Change | Meaning |
|---|---|
| `Δh_T = 0`, `Δh_P ≠ 0` | parameter-only change |
| `Δh_T ≠ 0` | topology changed |
| `Δh_T = Δh_P = 0`, `Δh_M ≠ 0` | registry or algebra version changed with no effect on this graph |

Invariants:

```
h_S(A) = h_S(B)  ⇒  h_T(A) = h_T(B)  and  h_P(A) = h_P(B)     (same registry)
h_T(A) = h_T(B)  ⟺  A and B specify the same TFNE topology     (same registry)
```

The converse of the first line fails. `V4 ↛ V1` and the equivalent population-level exclusions have different `h_S` but equal `h_T`.

Realization receipts (normalization factors, lost fields, realized edges) never enter `h_T`, `h_P` or `h_M`.

**[PROPOSED] Encoding:** canonical JSON (sorted keys, `(",", ":")` separators, every set as a sorted list, ASCII arrows), UTF-8, sha256. Each NF payload carries its schema id.

## 5. Validity [PROPOSED codes]

Errors carry a code, a message, and the smallest offending subexpression with its span. Validation stops at the first failing pipeline stage and reports every violation found there. Codes carry the `E_` prefix; `E_MECHANISM_UNRESOLVED` and `E_MECHANISM_CONFLICT` are the settled `MECHANISM_UNRESOLVED` and `MECHANISM_CONFLICT`.

| Code | Stage | Condition |
|---|---|---|
| `E_TOKEN` | 1 | undelimited operator, bad identifier, reserved name |
| `E_UNGROUPED_MIX` | 1 | different operators in one group |
| `E_REP_COMPOSITE` | 2 | exponent on a composite (OPEN-LANGUAGE) |
| `E_REP_INT` | 2 | n < 1 |
| `E_ID_COLLISION` | 2 | duplicate realized unit id |
| `E_MATCH` | 5 | O frontier labels not uniquely matchable, no map |
| `E_SHAPE` | 5 | Q operands of unequal depth or incompatible labels, no map |
| `E_MAP_INCOMPLETE` | 6 | map misses a frontier unit or names a unit outside it |
| `E_HIERARCHY_CYCLE` | 6 | O relations form a cycle |
| `E_RELATION_CONFLICT` | 6 | more than one architectural relation on a unit pair |
| `E_DUPLICATE` | 6 | a relation or statement declared twice |
| `E_X_COMPOSITE` | 6 | X operand is a composite (OPEN-LANGUAGE) |
| `E_ADDRESS_UNKNOWN` | 6 / 7 | undeclared unit (stage 6); unknown layer, alias or class (stage 7, needs the registry) |
| `E_PARAM_UNKNOWN` | 6 | θ key not declared for its target |
| `E_MECHANISM_UNRESOLVED` | 7 | resolved projection with no mechanism from the arrow or a generating motif |
| `E_MECHANISM_CONFLICT` | 7 | arrow mechanism differs from the generating motif's |
| `E_PROJECTION_REDUNDANT` | 7 | arrow mechanism equals the generating motif's |
| `E_MECHANISM_NOT_PERMITTED` | 9 | explicit projection adds a mechanism on a G0 route whose motif forbids extra mechanisms |
| `E_X_UNDIRECTED` | 7 | projection expansion of bare X |
| `E_MOTIF_UNDEFINED` | 7 | a motif or default projection motif used is not `defined` in the registry |
| `E_EXCL_NOT_IN_G0` | 9 | exclusion selects nothing in G0 |
| `E_ADD_IN_G0` | 9 | explicit projection identity already in G0 |
| `E_EXC_CONFLICT` | 9 | the same identity in E− and E+ |

## 6. Open items

### 6.1 Biological specifications [OPEN-BIOLOGY]

| ID | Specification | Status |
|---|---|---|
| D1 | O projection motif: classes per laminar route, mechanisms, parameters, `L56 → L56` pairing | OPEN |
| D2 | Q lateral motif | OPEN |
| D3 | X bypass motif catalogue, including the default projection motif for unit-level `→` | OPEN |
| D4a | layer-class proportions `p_{ℓ,c} ≥ 0`, including absent populations, and layer aliases | OPEN |
| D4b | scaling `H(N)` | OPEN |
| D4c | canonical input and output frontiers | OPEN |
| D4d | local canonical connectivity `C_local` | OPEN |
| D4e | cell and intrinsic parameter distributions `Θ_cell`, synaptic parameters `Θ_syn` | OPEN |
| D4f | spatial/morphological representation `G` required for observables | OPEN |

**Unit content.** H is not defined as the product `∏_{ℓ=1..6} {E, PV, SST, VIP}`. It is

```
H = { P_{ℓ,c}, C_local, Θ_cell, Θ_syn, G },   p_{ℓ,c} ≥ 0
```

TFNE fixes the class vocabulary `{E, PV, SST, VIP}`. D4a decides which layer-class populations exist and in what proportions. A class may be absent from a layer; L1 needs no E population because E is in the vocabulary. An address to an absent population raises `E_ADDRESS_UNKNOWN`.

Settled so far for D1: laminar routing only. FF runs lower `L23` → higher `L4`; FB runs higher `L56` → lower `L1` and `L56`.

**Biologically canonical projection expansion is blocked by D1–D4.** Structural normalization and projection-level synthetic conformance are not blocked.

### 6.2 Candidate direction for D4: `H_SL` [OPEN-BIOLOGY; candidate, not solved]

`H_SL` is a candidate canonical cortical unit, defined by an output invariant rather than by wiring:

```
H_SL = candidate canonical cortical unit
phenotype:  argmax_z gamma power      → L2/3
            argmax_z alpha-beta power → L5/6
            crossover                 → L4
```

- **Condition.** The motif must appear in the unit alone, without sensory stimulation, during finite irregular operation of the populations D4a declares. Construct the smallest H that produces it before testing `H O H`.
- The empirical target is the laminar landmark set. Its neuronal generators are unknown; superficial gamma, deep alpha-beta and superficial-deep interaction are candidate mechanisms, not settled ones.
- **Three gates**, each necessary and none implying another:

| Gate | Question | Tests |
|---|---|---|
| SL0 network generator | Do H's native neural/current dynamics contain superficial high-frequency and deep alpha-beta activity with the required laminar ordering? | the circuit |
| SL1 physical forward model | Given transmembrane/synaptic currents and cell geometry, does `I_m(r, t) → φ(z, t)` produce a laminar depth-frequency map, qualified against a known-answer case? | the biophysics |
| SL2 empirical phenotype | Does the composition `H →(SL0) currents →(SL1) LFP` give γ max ∼ L2/3, αβ max ∼ L5/6, crossover ∼ L4? | the phenotype |

- Only SL2 is compared against the empirical phenotype. A spike-derived proxy cannot establish SL1; its outputs are `lfp_proxy` / `csd_proxy`.
- A readout operator must not create the laminar motif. A depth-dependent filter, injected oscillatory source, or band/normalization choice belongs to the forward model and needs its own null control (`02_JAXFNE_COMPATIBILITY.md` §5).
- **H parameters are never tuned against a phenomenological proxy readout.** Doing so fits the readout operator rather than explaining the physiology.
- `H_SL` needs D4a–D4f; it settles none of them. SL1 depends on D4f.
- **Lineage.** The V2 scientific lineage may later yield a qualified H. TFNE does not assume that outcome, and TFNE work grants no V2 authority.

### 6.3 Language [OPEN-LANGUAGE]

- L1: exponent on composites
- L2: X between composites
- L3: settled (3.8): an arrow mechanism equal to the generating motif's raises `E_PROJECTION_REDUNDANT`

### 6.4 Awaiting confirmation [PROPOSED]

- unique unit ids, with no renaming
- map syntax and completeness
- Q operand sort key: units by id, composites by sorted member ids; integer labels numerically, otherwise as strings
- `E_HIERARCHY_CYCLE`, `E_DUPLICATE`
- parameter keys and types in the registry; defaults resolved before hashing
- registry field `extra_mechanisms: allowed | forbidden` per motif route, and `E_MECHANISM_NOT_PERMITTED`
- error codes and the encoding in 4.4

## 7. Reference spelling [PROVISIONAL SPELLING]

### 7.1 Tokens

- **Operators** `O`, `Q`, `X` are standalone tokens delimited by whitespace, `(`, `)` or `[`. They are reserved and cannot be unit names. For example, `V1 O OFC` has two units, while `V1OOFC` is one identifier.
- **Identifiers:** `[A-Za-z][A-Za-z0-9]*('_'[A-Za-z0-9]+)?`. A level belongs in the base: write `H1_a`, not `H_1a`.
- **Arrows:**

| Unicode | ASCII |
|---|---|
| `→` | `->` |
| `↛` | `-/->` |
| `↔` | `<->` |

  The normal form uses ASCII.
- **Programs** are statements joined by `;` with no order semantics. The same identifier in two statements is the same unit.

### 7.2 Grammar

```ebnf
program     = stmt { ";" stmt } ;
stmt        = arch | projection | exclusion ;
arch        = [ input ] expr ;                    (* the only statement that declares units *)
input       = IDENT ":" ;                         (* scope: to the end of the enclosing group *)
expr        = term { op term } ;                  (* one operator letter per group *)
op          = ( "O" | "Q" | "X" ) [ theta ] ;
term        = unit [ rep ] | "(" arch ")" ;
unit        = IDENT [ theta ] ;
rep         = ( "O" | "Q" ) "^" INT [ theta ] ;   (* postfix on a unit *)
projection  = address ( "->" | "→" ) [ theta ] address ;    (* theta on the arrow *)
exclusion   = address ( "-/->" | "↛" ) [ theta ] address ;
address     = IDENT { "." IDENT } [ theta ] ;              (* theta here parameterizes the population *)
theta       = "[" item { "," item } "]" ;
item        = KEY "=" VALUE | "->" | "→" | "<->" | "↔" ;
```

### 7.3 Map syntax

`map` is a set of unit-id pairs; fan-in and fan-out are allowed.

```
(H1 Q^2) O[map=(H1_1:H2_1, H1_1:H2_2, H1_2:H2_3, H1_2:H2_4)] (H2 Q^4)
```

## 8. Worked example (structural)

**Program:**

```
x:(V1_L Q V1_R) O (V4_L Q V4_R) O (FEF_L Q FEF_R) O (PFC_L Q PFC_R); V1_L X[->] PFC_L; V1_R X[->] PFC_R
```

**Equivalent spelling:**

```
V1_R X[->] PFC_R; x:(V1_R O V4_R O FEF_R O PFC_R) Q (V1_L O V4_L O FEF_L O PFC_L); V1_L X[->] PFC_L
```

Both give the same structural NF:

```json
{
  "schema": "tfne_structural_nf_v1",
  "algebra_version": "tfne-algebra/1",
  "units": ["FEF_L", "FEF_R", "PFC_L", "PFC_R", "V1_L", "V1_R", "V4_L", "V4_R"],
  "O": [["FEF_L","PFC_L"], ["FEF_R","PFC_R"], ["V1_L","V4_L"], ["V1_R","V4_R"], ["V4_L","FEF_L"], ["V4_R","FEF_R"]],
  "Q": [["FEF_L","FEF_R"], ["PFC_L","PFC_R"], ["V1_L","V1_R"], ["V4_L","V4_R"]],
  "X": [{"from": "V1_L", "to": "PFC_L", "dir": "->"}, {"from": "V1_R", "to": "PFC_R", "dir": "->"}],
  "inputs": [{"input": "x", "unit": "V1_L"}, {"input": "x", "unit": "V1_R"}],
  "projections": [],
  "exclusions": []
}
```

The units list is abridged to ids. The canonical projection NF of this program is blocked by D1–D4; its synthetic conformance NF is defined in `01_SYNTHETIC_CONFORMANCE.md`.
